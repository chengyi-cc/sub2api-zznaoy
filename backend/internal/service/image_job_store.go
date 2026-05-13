package service

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ImageJobStatus represents the lifecycle state of an async image job.
type ImageJobStatus string

const (
	ImageJobStatusQueued     ImageJobStatus = "queued"
	ImageJobStatusProcessing ImageJobStatus = "processing"
	ImageJobStatusSucceeded  ImageJobStatus = "succeeded"
	ImageJobStatusFailed     ImageJobStatus = "failed"
)

// ErrImageJobNotFound 由 LoadMeta / LoadResult / LoadRequestBody 在记录不存在时返回。
var ErrImageJobNotFound = errors.New("image job not found")

// ImageJobMeta 落盘到 meta.json 的任务元信息。
type ImageJobMeta struct {
	JobID             string         `json:"job_id"`
	Status            ImageJobStatus `json:"status"`
	Endpoint          string         `json:"endpoint"`
	ContentType       string         `json:"content_type"`
	Model             string         `json:"model"`
	APIKeyID          int64          `json:"api_key_id"`
	UserID            int64          `json:"user_id"`
	CreatedAt         int64          `json:"created_at"`
	StartedAt         int64          `json:"started_at,omitempty"`
	CompletedAt       int64          `json:"completed_at,omitempty"`
	HTTPStatus        int            `json:"http_status,omitempty"`
	ErrorType         string         `json:"error_type,omitempty"`
	ErrorMessage      string         `json:"error_message,omitempty"`
	RunTimeoutSeconds int            `json:"run_timeout_seconds,omitempty"`
}

// imageJobIDPattern 限制 job_id 形状为 "job_<32 hex>"，与 generateImageJobID 保持一致。
// 校验严格的目的是：cleaner 遍历 root_dir 时一旦遇到形状不符的目录就跳过 +
// 打 warn，绝不删除——避免用户误把 root_dir 指到包含其他数据的目录时被清掉。
var imageJobIDPattern = regexp.MustCompile(`^job_[A-Fa-f0-9]{32}$`)

// IsValidImageJobID 用于鉴定任意外部输入的 job_id 是否合法。
func IsValidImageJobID(jobID string) bool {
	return imageJobIDPattern.MatchString(jobID)
}

// ImageJobStore 是基于本地文件系统的简单任务存储。
//
// 目录结构：
//
//	<rootDir>/<job_id>/meta.json
//	<rootDir>/<job_id>/request.body  (原始请求体)
//	<rootDir>/<job_id>/result.json   (上游成功响应；失败任务不写)
type ImageJobStore struct {
	rootDir string
}

// NewImageJobStore 创建并初始化目录。
func NewImageJobStore(rootDir string) (*ImageJobStore, error) {
	rootDir = strings.TrimSpace(rootDir)
	if rootDir == "" {
		return nil, errors.New("image job store root dir is empty")
	}
	if err := os.MkdirAll(rootDir, 0o755); err != nil {
		return nil, fmt.Errorf("create image jobs dir %q: %w", rootDir, err)
	}
	return &ImageJobStore{rootDir: rootDir}, nil
}

// RootDir 返回 store 的根目录（仅用于诊断/日志）。
func (s *ImageJobStore) RootDir() string {
	return s.rootDir
}

func (s *ImageJobStore) jobDir(jobID string) string {
	return filepath.Join(s.rootDir, jobID)
}

// Create 写入初始 meta + request.body。
func (s *ImageJobStore) Create(meta *ImageJobMeta, requestBody []byte) error {
	if meta == nil || !IsValidImageJobID(meta.JobID) {
		return errors.New("invalid image job meta")
	}
	dir := s.jobDir(meta.JobID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "request.body"), requestBody, 0o600); err != nil {
		_ = os.RemoveAll(dir)
		return err
	}
	if err := s.writeMeta(meta); err != nil {
		_ = os.RemoveAll(dir)
		return err
	}
	return nil
}

func (s *ImageJobStore) writeMeta(meta *ImageJobMeta) error {
	raw, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	dir := s.jobDir(meta.JobID)
	tmp := filepath.Join(dir, "meta.json.tmp")
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, "meta.json"))
}

// UpdateMeta 读取-修改-回写。返回最新 meta。
func (s *ImageJobStore) UpdateMeta(jobID string, mutate func(*ImageJobMeta) error) (*ImageJobMeta, error) {
	meta, err := s.LoadMeta(jobID)
	if err != nil {
		return nil, err
	}
	if mutate != nil {
		if err := mutate(meta); err != nil {
			return nil, err
		}
	}
	if err := s.writeMeta(meta); err != nil {
		return nil, err
	}
	return meta, nil
}

// LoadMeta 读取 meta.json。
func (s *ImageJobStore) LoadMeta(jobID string) (*ImageJobMeta, error) {
	if !IsValidImageJobID(jobID) {
		return nil, ErrImageJobNotFound
	}
	raw, err := os.ReadFile(filepath.Join(s.jobDir(jobID), "meta.json"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrImageJobNotFound
		}
		return nil, err
	}
	var meta ImageJobMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("decode image job meta %q: %w", jobID, err)
	}
	return &meta, nil
}

// LoadRequestBody 读取原始请求字节。
func (s *ImageJobStore) LoadRequestBody(jobID string) ([]byte, error) {
	if !IsValidImageJobID(jobID) {
		return nil, ErrImageJobNotFound
	}
	raw, err := os.ReadFile(filepath.Join(s.jobDir(jobID), "request.body"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrImageJobNotFound
		}
		return nil, err
	}
	return raw, nil
}

// WriteResult 写入上游成功响应正文（JSON）。
func (s *ImageJobStore) WriteResult(jobID string, body []byte) error {
	if !IsValidImageJobID(jobID) {
		return ErrImageJobNotFound
	}
	return os.WriteFile(filepath.Join(s.jobDir(jobID), "result.json"), body, 0o600)
}

// LoadResult 读取上游成功响应正文。
func (s *ImageJobStore) LoadResult(jobID string) ([]byte, error) {
	if !IsValidImageJobID(jobID) {
		return nil, ErrImageJobNotFound
	}
	raw, err := os.ReadFile(filepath.Join(s.jobDir(jobID), "result.json"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrImageJobNotFound
		}
		return nil, err
	}
	return raw, nil
}

// Delete 移除整个任务目录。job_id 不合法时静默忽略。
func (s *ImageJobStore) Delete(jobID string) error {
	if !IsValidImageJobID(jobID) {
		return nil
	}
	return os.RemoveAll(s.jobDir(jobID))
}

// ListAll 枚举 store 下所有形如 "job_<32hex>" 的合法任务目录。
// 遇到形状不符的目录（用户可能把 root_dir 错配到了其它数据目录）一律 skip + warn，
// 绝不删——防御性默认。损坏的 *合法* 目录（meta.json 不可解析）才会被清掉。
func (s *ImageJobStore) ListAll() ([]*ImageJobMeta, error) {
	entries, err := os.ReadDir(s.rootDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []*ImageJobMeta
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		jobID := entry.Name()
		if !IsValidImageJobID(jobID) {
			slog.Warn("image_job_store.skip_unknown_dir",
				"root", s.rootDir,
				"name", jobID,
				"hint", "root_dir may be misconfigured; only job_<32hex> directories are managed",
			)
			continue
		}
		meta, err := s.LoadMeta(jobID)
		if err != nil {
			// 形状合法但 meta 损坏 → 视为孤儿，清掉。
			slog.Warn("image_job_store.delete_corrupt_job", "job_id", jobID, "error", err)
			_ = os.RemoveAll(filepath.Join(s.rootDir, jobID))
			continue
		}
		out = append(out, meta)
	}
	return out, nil
}

// TotalSizeBytes 统计根目录下**仅 job_<32hex> 目录**的总文件字节数。
// 和 ListAll 的管理范围严格一致——避免 root_dir 误配到含有其他数据的目录时，
// 非 job 数据撑爆配额统计、连带把合法 job 结果清掉。
func (s *ImageJobStore) TotalSizeBytes() (int64, error) {
	entries, err := os.ReadDir(s.rootDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	var total int64
	for _, entry := range entries {
		if !entry.IsDir() || !IsValidImageJobID(entry.Name()) {
			continue
		}
		jobDir := filepath.Join(s.rootDir, entry.Name())
		err := filepath.Walk(jobDir, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				if errors.Is(walkErr, os.ErrNotExist) {
					return filepath.SkipDir
				}
				return walkErr
			}
			if info != nil && !info.IsDir() {
				total += info.Size()
			}
			return nil
		})
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return total, err
		}
	}
	return total, nil
}

// NewImageJobMeta 是构造默认初始状态 meta 的便捷方法。
func NewImageJobMeta(jobID, endpoint, contentType, model string, apiKeyID, userID int64) *ImageJobMeta {
	return &ImageJobMeta{
		JobID:       jobID,
		Status:      ImageJobStatusQueued,
		Endpoint:    endpoint,
		ContentType: contentType,
		Model:       model,
		APIKeyID:    apiKeyID,
		UserID:      userID,
		CreatedAt:   time.Now().Unix(),
	}
}