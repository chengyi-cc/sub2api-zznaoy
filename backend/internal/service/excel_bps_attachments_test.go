package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func attachmentTestGateway(up HTTPUpstream) *OpenAIGatewayService {
	svc := openAIClientToolsTestService(nil)
	svc.httpUpstream = up
	svc.excelBPSImages = NewExcelBPSImageService(nil)
	return svc
}

type attachmentUpstream struct {
	HTTPUpstream
	mu              sync.Mutex
	uploads, models int
	uploadStatus    int
	uploadReply     string
	uploadsData     [][]byte
	urls, proxies   []string
	modelBodies     [][]byte
	t               *testing.T
}

func (u *attachmentUpstream) Do(req *http.Request, proxy string, accountID int64, concurrency int) (*http.Response, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.urls = append(u.urls, req.URL.String())
	u.proxies = append(u.proxies, proxy)
	require.True(u.t, HTTPUpstreamRedirectsDisabled(req.Context()))
	if req.URL.String() == excelBPSAttachmentsURL {
		u.uploads++
		require.Equal(u.t, "Bearer test-token", req.Header.Get("Authorization"))
		require.Equal(u.t, "test-account", req.Header.Get("Chatgpt-Account-Id"))
		require.Equal(u.t, "test-account", req.Header.Get("X-Openai-Account-Id"))
		require.Equal(u.t, "chatgpt", req.Header.Get("X-Basispoints-Auth-Mode"))
		require.Equal(u.t, "application/json", req.Header.Get("Accept"))
		kind, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
		require.NoError(u.t, err)
		require.Equal(u.t, "multipart/form-data", kind)
		raw, err := io.ReadAll(req.Body)
		require.NoError(u.t, err)
		require.Equal(u.t, int64(len(raw)), req.ContentLength)
		reader := multipart.NewReader(bytes.NewReader(raw), params["boundary"])
		part, err := reader.NextPart()
		require.NoError(u.t, err)
		require.Equal(u.t, "file", part.FormName())
		require.True(u.t, strings.HasPrefix(part.FileName(), "picture-"))
		require.Equal(u.t, "image/png", part.Header.Get("Content-Type"))
		data, err := io.ReadAll(part)
		require.NoError(u.t, err)
		u.uploadsData = append(u.uploadsData, data)
		_, err = reader.NextPart()
		require.ErrorIs(u.t, err, io.EOF)
		status := u.uploadStatus
		if status == 0 {
			status = 200
		}
		reply := u.uploadReply
		if reply == "" {
			reply = fmt.Sprintf(`{"openai_file_id":"file-test-%d"}`, u.uploads)
		}
		return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(reply))}, nil
	}
	require.Equal(u.t, basispoints.ResponsesURL, req.URL.String())
	u.models++
	body, err := io.ReadAll(req.Body)
	require.NoError(u.t, err)
	u.modelBodies = append(u.modelBodies, body)
	return excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_picture\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":20,\"output_tokens\":1}}}\n\n"), nil
}

func TestExcelBPSAttachmentsGatewayDeduplicatesAndIsolates(t *testing.T) {
	up := &attachmentUpstream{t: t}
	svc := attachmentTestGateway(up)
	svc.excelBPSImages = NewExcelBPSImageService(nil)
	inline := inlineTestImage(t, 1)
	body := inlineTestBody(t, inline, inline)
	original := bytes.Clone(body)
	account := excelAccount()
	for _, key := range []int64{1, 1, 2} {
		c, rec := imageGatewayContext()
		c.Set("api_key", &APIKey{ID: key})
		_, err := svc.Forward(context.Background(), c, account, body)
		require.NoError(t, err)
		require.Equal(t, 200, rec.Code)
	}
	require.Equal(t, 2, up.uploads)
	require.Equal(t, 3, up.models)
	require.Equal(t, up.modelBodies[0], up.modelBodies[1], "same image history must keep the entire upstream body stable")
	require.Equal(t, original, body)
	require.NotContains(t, string(up.modelBodies[0]), "data:image")
	require.NotContains(t, string(up.modelBodies[0]), "inline-image.invalid")
	require.NotContains(t, string(up.modelBodies[0]), "/v1/excel-images")
	for _, wire := range up.modelBodies {
		count := 0
		require.NoError(t, excelBPSImageParts(wire, func(_ string, part gjson.Result) error {
			require.True(t, validExcelBPSFileID(part.Get("file_id").String()))
			require.False(t, part.Get("image_url").Exists())
			require.Equal(t, "high", part.Get("detail").String())
			count++
			return nil
		}))
		require.Equal(t, 2, count)
	}
	account.ID++
	c, _ := imageGatewayContext()
	c.Set("api_key", &APIKey{ID: 1})
	_, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Equal(t, 3, up.uploads, "upstream account switches must not share account-owned file IDs")
}

func TestExcelBPSAttachmentsConcurrentUploadAndCancellation(t *testing.T) {
	svc := NewExcelBPSImageService(nil)
	key := excelBPSAttachmentKey{excelBPSAttachmentScope{1, 2, "upstream"}, sha256.Sum256([]byte("picture"))}
	started, release := make(chan struct{}), make(chan struct{})
	var count atomic.Int64
	upload := func() (string, error) {
		if count.Add(1) == 1 {
			close(started)
		}
		<-release
		return "file-shared", nil
	}
	var wg sync.WaitGroup
	results := make(chan string, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id, err := svc.fileID(context.Background(), key, upload)
			if err != nil {
				results <- "error"
			} else {
				results <- id
			}
		}()
	}
	<-started
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := svc.fileID(ctx, key, upload)
	require.ErrorIs(t, err, context.Canceled)
	close(release)
	wg.Wait()
	close(results)
	for id := range results {
		require.Equal(t, "file-shared", id)
	}
	require.Equal(t, int64(1), count.Load())
	require.Empty(t, svc.flights)
	require.Empty(t, svc.slots)
}

func TestExcelBPSAttachmentsFailuresNeverSendModelOrCacheFailure(t *testing.T) {
	for _, tc := range []struct {
		status int
		reply  string
	}{{401, `{"error":{"message":"secret-token"}}`}, {413, `{}`}, {302, `{}`}, {200, `{"openai_file_id":"https://bad.example"}`}, {200, `{}`}, {200, strings.Repeat("x", 65537)}} {
		up := &attachmentUpstream{t: t, uploadStatus: tc.status, uploadReply: tc.reply}
		svc := attachmentTestGateway(up)
		svc.excelBPSImages = NewExcelBPSImageService(nil)
		body := inlineTestBody(t, inlineTestImage(t, 2))
		c, rec := imageGatewayContext()
		_, err := svc.Forward(context.Background(), c, excelAccount(), body)
		require.Error(t, err)
		require.Equal(t, 502, rec.Code)
		require.Zero(t, up.models)
		require.NotContains(t, rec.Body.String(), "secret-token")
		require.Empty(t, svc.excelBPSImages.ids)
		up.uploadStatus = 200
		up.uploadReply = `{"openai_file_id":"file-recovered"}`
		c, _ = imageGatewayContext()
		_, err = svc.Forward(context.Background(), c, excelAccount(), body)
		require.NoError(t, err)
		require.Equal(t, 2, up.uploads)
		require.Equal(t, 1, up.models)
	}
}

func TestExcelBPSAttachmentsValidateBeforeSideEffectsAndNativeUnchanged(t *testing.T) {
	up := &attachmentUpstream{t: t}
	svc := attachmentTestGateway(up)
	svc.excelBPSImages = NewExcelBPSImageService(nil)
	body := inlineTestBody(t, inlineTestImage(t, 3))
	invalid := bytes.Replace(body, []byte(`"model":`), []byte(`"previous_response_id":"old","model":`), 1)
	c, _ := imageGatewayContext()
	_, err := svc.Forward(context.Background(), c, excelAccount(), invalid)
	require.Error(t, err)
	require.Zero(t, up.uploads)
	require.Zero(t, up.models)
	native := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_native\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Image received\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n")}
	svc = openAIClientToolsTestService(native)
	account := excelAccount()
	account.Extra["openai_excel_bps"] = false
	c, _ = imageGatewayContext()
	_, err = svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Contains(t, string(native.lastBody), "data:image/png;base64,")
}

func TestExcelBPSAttachmentsCacheEvictionAndInvalidation(t *testing.T) {
	svc := NewExcelBPSImageService(nil)
	scope := excelBPSAttachmentScope{1, 2, "upstream"}
	key := excelBPSAttachmentKey{scope, sha256.Sum256([]byte("first"))}
	_, err := svc.fileID(context.Background(), key, func() (string, error) { return "file-first", nil })
	require.NoError(t, err)
	wire := []byte(`{"input":[{"role":"user","content":[{"type":"input_image","file_id":"file-first"}]}]}`)
	svc.invalidateRejected(scope, wire, []byte(`{"error":{"code":"invalid_request"}}`))
	require.Len(t, svc.ids, 1)
	svc.invalidateRejected(scope, wire, []byte(`{"error":{"code":"file_not_found"}}`))
	require.Empty(t, svc.ids)
	for i := 0; i < excelBPSAttachmentCacheSize+1; i++ {
		key.digest = sha256.Sum256([]byte(fmt.Sprint(i)))
		_, err = svc.fileID(context.Background(), key, func() (string, error) { return fmt.Sprintf("file-%d", i), nil })
		require.NoError(t, err)
	}
	require.Len(t, svc.ids, excelBPSAttachmentCacheSize)
	key.digest = sha256.Sum256([]byte("0"))
	require.Nil(t, svc.ids[key])
}

func TestExcelBPSLegacyCleanupLeavesOtherFilesAlone(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	expired := filepath.Join(root, strings.Repeat("a", 64))
	active := filepath.Join(root, strings.Repeat("b", 64))
	other := filepath.Join(root, "meta.json")
	for _, path := range []string{expired, active, other} {
		require.NoError(t, os.WriteFile(path, []byte("old image"), 0600))
	}
	require.NoError(t, os.Chtimes(expired, now.Add(-6*time.Minute), now.Add(-6*time.Minute)))
	require.NoError(t, cleanupLegacyExcelImages(root, now))
	require.NoFileExists(t, expired)
	require.FileExists(t, active)
	require.FileExists(t, other)
	require.NoError(t, cleanupLegacyExcelImages(root, now.Add(6*time.Minute)))
	require.NoFileExists(t, active)
	require.FileExists(t, other)
	missing := filepath.Join(root, "missing")
	require.NoError(t, cleanupLegacyExcelImages(missing, now))
	require.NoDirExists(t, missing)
}

func TestExcelBPSAttachmentsToolOutputImages(t *testing.T) {
	up := &attachmentUpstream{t: t}
	svc := attachmentTestGateway(up)
	body := []byte(fmt.Sprintf(`{"model":"gpt-6-astra","tools":[{"type":"function","name":"view_image","parameters":{"type":"object","properties":{}}}],"input":[{"type":"function_call","call_id":"call_picture","name":"view_image","arguments":"{}"},{"type":"function_call_output","call_id":"call_picture","output":[{"type":"input_image","image_url":"%s","detail":"high"}]}]}`, inlineTestImage(t, 5)))
	c, _ := imageGatewayContext()
	_, err := svc.Forward(context.Background(), c, excelAccount(), body)
	require.NoError(t, err)
	require.Equal(t, 1, up.uploads)
	require.Equal(t, 1, up.models)
	require.NotContains(t, string(up.modelBodies[0]), "inline-image.invalid")
	require.Contains(t, string(up.modelBodies[0]), `"file_id":"file-test-1"`)
}
