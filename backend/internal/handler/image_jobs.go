package handler

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// imageJobsState 是 OpenAIGatewayHandler 的懒加载字段。
type imageJobsState struct {
	Store         *service.ImageJobStore
	TTL           time.Duration
	RunTimeout    time.Duration // 客户端不传 ?timeout_seconds 时的默认值
	MaxRunTimeout time.Duration // ?timeout_seconds 允许的上限
	MaxTotalBytes int64
	Interval      time.Duration
}

func (h *OpenAIGatewayHandler) ensureImageJobsState() (*imageJobsState, error) {
	h.imageJobsStateMu.Lock()
	defer h.imageJobsStateMu.Unlock()
	if h.imageJobsState != nil {
		return h.imageJobsState, nil
	}
	cfg := h.cfg.Gateway.ImageJobs
	rootDir := strings.TrimSpace(cfg.RootDir)
	if rootDir == "" {
		base := strings.TrimSpace(h.cfg.Pricing.DataDir)
		if base == "" {
			base = "./data"
		}
		rootDir = filepath.Join(base, "jobs", "images")
	}
	store, err := service.NewImageJobStore(rootDir)
	if err != nil {
		return nil, err
	}
	state := &imageJobsState{
		Store:         store,
		TTL:           time.Duration(cfg.TTLSeconds) * time.Second,
		RunTimeout:    time.Duration(cfg.RunTimeoutSeconds) * time.Second,
		MaxRunTimeout: time.Duration(cfg.MaxRunTimeoutSeconds) * time.Second,
		MaxTotalBytes: int64(cfg.MaxTotalDiskMB) << 20,
		Interval:      time.Duration(cfg.CleanupIntervalSeconds) * time.Second,
	}
	if state.RunTimeout <= 0 {
		state.RunTimeout = 5 * time.Minute
	}
	if state.MaxRunTimeout <= 0 {
		state.MaxRunTimeout = 10 * time.Minute
	}
	if state.MaxRunTimeout < state.RunTimeout {
		state.MaxRunTimeout = state.RunTimeout
	}
	if state.Interval <= 0 {
		state.Interval = 2 * time.Minute
	}
	h.imageJobsState = state
	return state, nil
}

// StartImageJobsCleaner 由构造函数启动一次，传入根 ctx，绑进程生命周期。
func (h *OpenAIGatewayHandler) StartImageJobsCleaner(ctx context.Context) {
	state, err := h.ensureImageJobsState()
	if err != nil {
		logger.LegacyPrintf("handler.image_jobs", "image jobs cleaner init failed: %v", err)
		return
	}
	service.StartImageJobCleaner(ctx, service.ImageJobCleanerConfig{
		Store:           state.Store,
		TTL:             state.TTL,
		MaxTotalBytes:   state.MaxTotalBytes,
		Interval:        state.Interval,
		ZombieThreshold: state.MaxRunTimeout + 2*time.Minute,
	})
}

// SubmitImageJob 返回一个 handler，处理 POST /v1/jobs/images/{generations,edits}。
// endpoint 必须是 service.openAIImagesGenerationsEndpoint 或 service.openAIImagesEditsEndpoint。
func (h *OpenAIGatewayHandler) SubmitImageJob(endpoint string) gin.HandlerFunc {
	return func(c *gin.Context) {
		state, err := h.ensureImageJobsState()
		if err != nil {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "Image jobs storage unavailable")
			return
		}
		apiKey, ok := middleware2.GetAPIKeyFromContext(c)
		if !ok {
			h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
			return
		}
		subject, ok := middleware2.GetAuthSubjectFromContext(c)
		if !ok {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "User context not found")
			return
		}
		if !service.GroupAllowsImageGeneration(apiKey.Group) {
			h.errorResponse(c, http.StatusForbidden, "permission_error", service.ImageGenerationPermissionMessage())
			return
		}

		// 让下游解析以及 GetInboundEndpoint 等基于 URL.Path 的 fallback 看到正确的端点。
		c.Request.URL.Path = endpoint

		body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
		if err != nil {
			if maxErr, ok := extractMaxBytesError(err); ok {
				h.errorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
				return
			}
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
			return
		}
		if len(body) == 0 {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
			return
		}

		// 提交期就解析一次：让明显畸形的请求体在 202 之前 400 出去。
		parsed, err := h.gatewayService.ParseOpenAIImagesRequest(c, body)
		if err != nil {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		// 异步任务本就是流式接口的替代品，stream=true 在 jobs 接口下语义矛盾，
		// 而且会让 RunTimeout 失效（ForwardImages 对 stream 请求会 detach 上下文）。
		// 直接 400 拒绝，让调用方明确改成 stream=false。
		if parsed.Stream {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error",
				"stream=true is not supported on /v1/jobs/images/*; submit with stream=false and poll the job")
			return
		}

		// 解析可选的 ?timeout_seconds=N：不传 → 用全局默认 state.RunTimeout；
		// 传 → 在 [1, MaxRunTimeout] 内校验，超过上限直接 400 不静默 clamp。
		runTimeout := state.RunTimeout
		if raw := strings.TrimSpace(c.Query("timeout_seconds")); raw != "" {
			n, err := strconv.Atoi(raw)
			if err != nil || n <= 0 {
				h.errorResponse(c, http.StatusBadRequest, "invalid_request_error",
					"timeout_seconds must be a positive integer")
				return
			}
			maxSec := int(state.MaxRunTimeout / time.Second)
			if n > maxSec {
				h.errorResponse(c, http.StatusBadRequest, "invalid_request_error",
					fmt.Sprintf("timeout_seconds exceeds maximum (%ds)", maxSec))
				return
			}
			runTimeout = time.Duration(n) * time.Second
		}

		contentType := c.GetHeader("Content-Type")

		jobID, err := generateImageJobID()
		if err != nil {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to allocate job id")
			return
		}
		meta := service.NewImageJobMeta(jobID, endpoint, contentType, parsed.Model, apiKey.ID, subject.UserID)
		meta.RunTimeoutSeconds = int(runTimeout / time.Second)
		if err := state.Store.Create(meta, body); err != nil {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to persist image job")
			return
		}

		runCtx := imageJobRunContext{
			APIKey:       apiKey,
			Subject:      subject,
			ContentType:  contentType,
			Endpoint:     endpoint,
			UserAgent:    c.GetHeader("User-Agent"),
			ClientIP:     c.ClientIP(),
			ForwardedFor: c.GetHeader("X-Forwarded-For"),
			RealIP:       c.GetHeader("X-Real-Ip"),
			RequestID:    c.GetHeader("X-Request-Id"),
		}
		if sub, ok := middleware2.GetSubscriptionFromContext(c); ok {
			runCtx.Subscription = sub
		}

		go h.runImageJob(jobID, runCtx, state, body)

		c.JSON(http.StatusAccepted, gin.H{
			"job_id":     jobID,
			"status":     string(service.ImageJobStatusQueued),
			"created_at": meta.CreatedAt,
		})
	}
}

// imageJobRunContext 是从原始提交请求里捕获、传给 goroutine 的上下文快照。
type imageJobRunContext struct {
	APIKey       *service.APIKey
	Subject      middleware2.AuthSubject
	Subscription *service.UserSubscription
	ContentType  string
	Endpoint     string
	UserAgent    string
	ClientIP     string
	ForwardedFor string
	RealIP       string
	RequestID    string
}

// GetImageJob 处理 GET /v1/jobs/:job_id。
func (h *OpenAIGatewayHandler) GetImageJob(c *gin.Context) {
	state, err := h.ensureImageJobsState()
	if err != nil {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "Image jobs storage unavailable")
		return
	}
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}

	jobID := strings.TrimSpace(c.Param("job_id"))
	if !service.IsValidImageJobID(jobID) {
		h.errorResponse(c, http.StatusNotFound, "not_found_error", "Job not found")
		return
	}
	meta, err := state.Store.LoadMeta(jobID)
	if err != nil {
		h.errorResponse(c, http.StatusNotFound, "not_found_error", "Job not found")
		return
	}
	if meta.APIKeyID != apiKey.ID {
		// 不暴露存在性。
		h.errorResponse(c, http.StatusNotFound, "not_found_error", "Job not found")
		return
	}

	switch meta.Status {
	case service.ImageJobStatusQueued, service.ImageJobStatusProcessing:
		c.Header("Retry-After", "3")
		c.JSON(http.StatusOK, gin.H{
			"job_id":     meta.JobID,
			"status":     string(meta.Status),
			"created_at": meta.CreatedAt,
		})
		return
	case service.ImageJobStatusFailed:
		c.JSON(http.StatusOK, gin.H{
			"job_id":       meta.JobID,
			"status":       string(meta.Status),
			"created_at":   meta.CreatedAt,
			"completed_at": meta.CompletedAt,
			"http_status":  meta.HTTPStatus,
			"error": gin.H{
				"type":    meta.ErrorType,
				"message": meta.ErrorMessage,
			},
		})
		_ = state.Store.Delete(jobID)
		return
	case service.ImageJobStatusSucceeded:
		body, err := state.Store.LoadResult(jobID)
		if err != nil {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to read job result")
			return
		}
		wrapped, err := wrapImageJobSuccess(meta, body)
		if err != nil {
			h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to encode job result")
			return
		}
		c.Data(http.StatusOK, "application/json; charset=utf-8", wrapped)
		_ = state.Store.Delete(jobID)
		return
	}

	h.errorResponse(c, http.StatusInternalServerError, "api_error", "Unknown job status")
}

// wrapImageJobSuccess 将上游 images JSON 与外层包装字段合并成扁平 JSON。
// 若键冲突，外层字段优先（实际上 OpenAI images 响应没有 job_id/status/completed_at/created_at 字段）。
func wrapImageJobSuccess(meta *service.ImageJobMeta, inner []byte) ([]byte, error) {
	innerMap := map[string]json.RawMessage{}
	if len(inner) > 0 {
		if err := json.Unmarshal(inner, &innerMap); err != nil {
			return nil, err
		}
	}
	wrapper := map[string]any{
		"job_id":       meta.JobID,
		"status":       string(meta.Status),
		"created_at":   meta.CreatedAt,
		"completed_at": meta.CompletedAt,
	}
	for k, v := range wrapper {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		innerMap[k] = raw
	}
	return json.Marshal(innerMap)
}

// runImageJob 在独立 goroutine 中执行任务：合成一个 *gin.Context 跑一遍现有的 Images() 处理流程，
// 把 recorder 收到的状态码 + 响应正文写回 store。
func (h *OpenAIGatewayHandler) runImageJob(jobID string, runCtx imageJobRunContext, state *imageJobsState, body []byte) {
	defer func() {
		if r := recover(); r != nil {
			logger.LegacyPrintf("handler.image_jobs", "panic in image job runner: %v", r)
			_, _ = state.Store.UpdateMeta(jobID, func(meta *service.ImageJobMeta) error {
				meta.Status = service.ImageJobStatusFailed
				meta.HTTPStatus = http.StatusInternalServerError
				meta.ErrorType = "internal_error"
				meta.ErrorMessage = fmt.Sprintf("internal panic: %v", r)
				meta.CompletedAt = time.Now().Unix()
				return nil
			})
		}
	}()

	if _, err := state.Store.UpdateMeta(jobID, func(meta *service.ImageJobMeta) error {
		meta.Status = service.ImageJobStatusProcessing
		meta.StartedAt = time.Now().Unix()
		return nil
	}); err != nil {
		logger.LegacyPrintf("handler.image_jobs", "image job %s mark processing failed: %v", jobID, err)
		return
	}

	// 单任务超时：优先用提交期写入 meta 的值（来自 ?timeout_seconds），缺省则用全局默认。
	runTimeout := state.RunTimeout
	if metaSnapshot, err := state.Store.LoadMeta(jobID); err == nil && metaSnapshot.RunTimeoutSeconds > 0 {
		runTimeout = time.Duration(metaSnapshot.RunTimeoutSeconds) * time.Second
	}
	jobCtx, cancel := context.WithTimeout(context.Background(), runTimeout)
	defer cancel()

	recorder := newImageJobRecorder()
	c, _ := gin.CreateTestContext(recorder)

	req, err := http.NewRequestWithContext(jobCtx, http.MethodPost, runCtx.Endpoint, bytes.NewReader(body))
	if err != nil {
		h.persistImageJobError(jobID, state, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}
	if runCtx.ContentType != "" {
		req.Header.Set("Content-Type", runCtx.ContentType)
	}
	req.ContentLength = int64(len(body))
	if runCtx.UserAgent != "" {
		req.Header.Set("User-Agent", runCtx.UserAgent)
	}
	if runCtx.ForwardedFor != "" {
		req.Header.Set("X-Forwarded-For", runCtx.ForwardedFor)
	} else if runCtx.ClientIP != "" {
		req.Header.Set("X-Forwarded-For", runCtx.ClientIP)
	}
	if runCtx.RealIP != "" {
		req.Header.Set("X-Real-Ip", runCtx.RealIP)
	}
	if runCtx.RequestID != "" {
		req.Header.Set("X-Request-Id", runCtx.RequestID)
	}
	if runCtx.ClientIP != "" {
		req.RemoteAddr = runCtx.ClientIP + ":0"
	}

	c.Request = req
	c.Set(string(middleware2.ContextKeyAPIKey), runCtx.APIKey)
	c.Set(string(middleware2.ContextKeyUser), runCtx.Subject)
	if runCtx.Subscription != nil {
		c.Set(string(middleware2.ContextKeySubscription), runCtx.Subscription)
	}
	c.Set(ctxKeyInboundEndpoint, runCtx.Endpoint)

	// 跑现有的同步 Images() —— 内部 defer 会负责并发槽归还、计费记录。
	h.Images(c)

	statusCode := recorder.statusCode
	if statusCode == 0 {
		statusCode = http.StatusInternalServerError
	}
	respBody := recorder.Body.Bytes()

	if statusCode >= 200 && statusCode < 300 {
		if err := state.Store.WriteResult(jobID, respBody); err != nil {
			h.persistImageJobError(jobID, state, http.StatusInternalServerError, "internal_error", "Failed to write job result")
			return
		}
		_, _ = state.Store.UpdateMeta(jobID, func(meta *service.ImageJobMeta) error {
			meta.Status = service.ImageJobStatusSucceeded
			meta.HTTPStatus = statusCode
			meta.CompletedAt = time.Now().Unix()
			return nil
		})
		return
	}

	errType, errMsg := extractImageJobError(respBody)
	if errType == "" {
		errType = "upstream_error"
	}
	if errMsg == "" {
		errMsg = http.StatusText(statusCode)
	}
	_, _ = state.Store.UpdateMeta(jobID, func(meta *service.ImageJobMeta) error {
		meta.Status = service.ImageJobStatusFailed
		meta.HTTPStatus = statusCode
		meta.CompletedAt = time.Now().Unix()
		meta.ErrorType = errType
		meta.ErrorMessage = errMsg
		return nil
	})
}

func extractImageJobError(body []byte) (string, string) {
	if len(body) == 0 {
		return "", ""
	}
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err == nil {
		return envelope.Error.Type, envelope.Error.Message
	}
	return "", ""
}

func (h *OpenAIGatewayHandler) persistImageJobError(jobID string, state *imageJobsState, statusCode int, errType, errMsg string) {
	_, _ = state.Store.UpdateMeta(jobID, func(meta *service.ImageJobMeta) error {
		meta.Status = service.ImageJobStatusFailed
		meta.HTTPStatus = statusCode
		meta.CompletedAt = time.Now().Unix()
		meta.ErrorType = errType
		meta.ErrorMessage = errMsg
		return nil
	})
}

// generateImageJobID 生成形如 job_<32 hex> 的任务 ID。
func generateImageJobID() (string, error) {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", err
	}
	return "job_" + hex.EncodeToString(buf[:]), nil
}

// imageJobRecorder 实现 http.ResponseWriter + http.Flusher，被 gin.CreateTestContext 包装后供合成上下文使用。
// gin.responseWriter.Flush 会断言底层实现 http.Flusher，因此 Flush 必须存在。
type imageJobRecorder struct {
	statusCode int
	header     http.Header
	Body       *bytes.Buffer
}

func newImageJobRecorder() *imageJobRecorder {
	return &imageJobRecorder{
		header: http.Header{},
		Body:   bytes.NewBuffer(nil),
	}
}

func (r *imageJobRecorder) Header() http.Header { return r.header }

func (r *imageJobRecorder) Write(b []byte) (int, error) {
	if r.statusCode == 0 {
		r.statusCode = http.StatusOK
	}
	return r.Body.Write(b)
}

func (r *imageJobRecorder) WriteHeader(code int) {
	r.statusCode = code
}

func (r *imageJobRecorder) Flush() {}