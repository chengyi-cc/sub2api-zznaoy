package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestExcelBPSLocalHTTPLeaseAndRestart(t *testing.T) {
	s := imageTestService(t)
	now := time.Now().Truncate(time.Second)
	s.now = func() time.Time { return now }
	body, plan, err := prepareExcelBPSImages(inlineTestBody(t, inlineTestImage(t, 10)))
	require.NoError(t, err)
	wire, err := s.upload(context.Background(), body, plan, "key:1/thread:1", "site.example")
	require.NoError(t, err)
	link := gjson.GetBytes(wire, "input.0.content.1.image_url").String()
	parsed, err := url.Parse(link)
	require.NoError(t, err)
	token := parsed.Query().Get("token")
	cfg := &config.Config{}
	cfg.Gateway.ImageJobs.RootDir = strings.TrimSuffix(s.root, "-excel-inputs")
	cfg.JWT.Secret = s.secret
	restarted := NewExcelBPSImageService(cfg, nil)
	restarted.now = s.now
	gateway := &OpenAIGatewayService{excelBPSImages: restarted}
	request := func(method, link string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		gateway.ServeExcelBPSImage(rec, httptest.NewRequest(method, link, nil))
		return rec
	}
	get := request("GET", link)
	require.Equal(t, 200, get.Code)
	require.Equal(t, plan.images[0].data, get.Body.Bytes())
	require.Equal(t, "image/png", get.Header().Get("Content-Type"))
	require.Contains(t, get.Header().Get("Cache-Control"), "no-store")
	require.Equal(t, "nosniff", get.Header().Get("X-Content-Type-Options"))
	head := request("HEAD", link)
	require.Equal(t, 200, head.Code)
	require.Empty(t, head.Body.Bytes())
	require.Equal(t, get.Header().Get("Content-Length"), head.Header().Get("Content-Length"))
	require.Equal(t, 405, request("POST", link).Code)
	for _, invalid := range []string{"", "../meta.json", token[:63], "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"} {
		require.Equal(t, 404, request("GET", "https://site.example"+ExcelBPSImagePath+"?token="+url.QueryEscape(invalid)).Code)
	}
	resumed, err := restarted.upload(context.Background(), body, plan, "key:1/thread:1", "site.example")
	require.NoError(t, err)
	require.Equal(t, wire, resumed)
	now = now.Add(ExcelBPSImageTTL)
	require.Equal(t, 404, request("GET", link).Code, "expiry is enforced before the periodic cleaner")
	require.Equal(t, 404, request("HEAD", link).Code)
	require.FileExists(t, filepath.Join(s.root, token))
	restarted.Start()
	require.Eventually(t, func() bool { _, err := os.Stat(filepath.Join(s.root, token)); return os.IsNotExist(err) }, 3*time.Second, 10*time.Millisecond)
	restarted.Stop()
	restarted.Stop()
}

func TestExcelBPSLocalConcurrentRenewalAndQuota(t *testing.T) {
	s := imageTestService(t)
	body, plan, err := prepareExcelBPSImages(inlineTestBody(t, inlineTestImage(t, 1)))
	require.NoError(t, err)
	expected, err := s.upload(context.Background(), body, plan, "caller", "site.example")
	require.NoError(t, err)
	token := s.imageToken("caller", plan.images[0].data)
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := s.upload(context.Background(), body, plan, "caller", "site.example")
			if !assertLocalResult(t, err == nil && string(got) == string(expected), "concurrent renewal changed the request") {
				return
			}
			f, err := s.OpenImage(token)
			if !assertLocalResult(t, err == nil, "concurrent fetch failed") {
				return
			}
			data, err := io.ReadAll(f)
			_ = f.Close()
			assertLocalResult(t, err == nil && string(data) == string(plan.images[0].data), "image was truncated")
			s.mu.Lock()
			err = s.cleanupLocked()
			s.mu.Unlock()
			assertLocalResult(t, err == nil, "cleanup failed")
		}()
	}
	wg.Wait()
	require.Len(t, s.files, 1)
	require.Equal(t, int64(len(plan.images[0].data)), s.total)
	s.maxBytes = s.total
	_, err = s.upload(context.Background(), body, plan, "other-caller", "site.example")
	require.Error(t, err)
	require.Len(t, s.files, 1, "quota failures must not evict active conversation images")
	_, err = s.upload(context.Background(), body, plan, "caller", "site.example")
	require.NoError(t, err, "existing images can still be renewed when full")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = s.upload(ctx, body, plan, "caller", "site.example")
	require.Error(t, err)
}

func assertLocalResult(t *testing.T, ok bool, message string) bool {
	t.Helper()
	if !ok {
		t.Error(message)
	}
	return ok
}

func TestExcelBPSLocalPublicOriginAndDiagnostics(t *testing.T) {
	s := imageTestService(t)
	s.frontendURL = "https://configured.example/prefix/v1/"
	base, err := s.publicBaseURL(context.Background(), "untrusted.example")
	require.NoError(t, err)
	require.Equal(t, "https://configured.example/prefix"+ExcelBPSImagePath, base)
	s.frontendURL = ""
	base, err = s.publicBaseURL(context.Background(), "site.example:8443")
	require.NoError(t, err)
	require.Equal(t, "https://site.example:8443"+ExcelBPSImagePath, base)
	for _, host := range []string{"", "evil.example/?redirect=x", "name@evil.example", "evil.example\\path"} {
		_, err := s.publicBaseURL(context.Background(), host)
		require.Error(t, err)
	}
	for _, base := range []string{"http://site.example", "https://name:secret@site.example", "https://site.example?secret=value", "https://site.example/#fragment"} {
		s.frontendURL = base
		_, err := s.publicBaseURL(context.Background(), "site.example")
		require.Error(t, err)
	}
	raw := `{"error":{"message":"Could not fetch https://site.example/v1/excel-images?token=private-image-capability"}}`
	require.NotContains(t, excelBPSSanitizeErrorBody(raw, "account-token", excelAccount()), "private-image-capability")
}

func TestExcelBPSLocalEndpointOverHTTP(t *testing.T) {
	s := imageTestService(t)
	body, plan, err := prepareExcelBPSImages(inlineTestBody(t, inlineTestImage(t, 5)))
	require.NoError(t, err)
	_, err = s.upload(context.Background(), body, plan, "caller", "site.example")
	require.NoError(t, err)
	gateway := &OpenAIGatewayService{excelBPSImages: s}
	server := httptest.NewServer(http.HandlerFunc(gateway.ServeExcelBPSImage))
	defer server.Close()
	resp, err := server.Client().Get(server.URL + ExcelBPSImagePath + "?token=" + s.imageToken("caller", plan.images[0].data))
	require.NoError(t, err)
	defer resp.Body.Close()
	actual, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 200, resp.StatusCode)
	require.Equal(t, plan.images[0].data, actual)
}

func TestExcelBPSLocalInterruptedWritesAreCleanedAfterRestart(t *testing.T) {
	s := imageTestService(t)
	now := time.Now().Truncate(time.Second)
	s.now = func() time.Time { return now }
	require.NoError(t, os.MkdirAll(s.root, 0700))
	pending := filepath.Join(s.root, ".pending-interrupted")
	require.NoError(t, os.WriteFile(pending, []byte("partial image"), 0600))
	require.NoError(t, os.Chtimes(pending, now, now))
	require.NoError(t, s.cleanupLocked())
	require.FileExists(t, pending)
	now = now.Add(ExcelBPSImageTTL)
	require.NoError(t, s.cleanupLocked())
	require.NoFileExists(t, pending)
	require.Zero(t, s.total)
}

type excelImageSettingsFake struct {
	SettingRepository
	values map[string]string
}

func (r excelImageSettingsFake) GetValue(_ context.Context, key string) (string, error) {
	return r.values[key], nil
}

func TestExcelBPSLocalConfiguredOriginPriority(t *testing.T) {
	s := imageTestService(t)
	s.frontendURL = "https://config.example"
	values := map[string]string{SettingKeyAPIBaseURL: "https://api.example/v1", SettingKeyFrontendURL: "https://frontend.example"}
	s.settings = excelImageSettingsFake{values: values}
	base, err := s.publicBaseURL(context.Background(), "request.example")
	require.NoError(t, err)
	require.Equal(t, "https://api.example"+ExcelBPSImagePath, base)
	delete(values, SettingKeyAPIBaseURL)
	base, err = s.publicBaseURL(context.Background(), "request.example")
	require.NoError(t, err)
	require.Equal(t, "https://frontend.example"+ExcelBPSImagePath, base)
}

func TestExcelBPSLocalForwardedImageCanBeFetchedAcrossAccountSwitch(t *testing.T) {
	s := imageTestService(t)
	upstream := &httpUpstreamRecorder{}
	gateway := openAIClientToolsTestService(upstream)
	gateway.excelBPSImages = s
	server := httptest.NewTLSServer(http.HandlerFunc(gateway.ServeExcelBPSImage))
	defer server.Close()
	s.frontendURL = server.URL
	body := inlineTestBody(t, inlineTestImage(t, 8))
	_, plan, err := prepareExcelBPSImages(body)
	require.NoError(t, err)
	var previousLink string
	for _, accountID := range []int64{1, 2} {
		upstream.resp = excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_image\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n")
		c, rec := imageGatewayContext()
		c.Set("api_key", &APIKey{ID: 42})
		account := excelAccount()
		account.ID = accountID
		_, err := gateway.Forward(context.Background(), c, account, body)
		require.NoError(t, err)
		require.Equal(t, 200, rec.Code)
		var link string
		require.NoError(t, excelBPSImageParts(upstream.lastBody, func(_ string, part gjson.Result) error {
			link = part.Get("image_url").String()
			return nil
		}))
		require.NotEmpty(t, link)
		if previousLink != "" {
			require.Equal(t, previousLink, link)
		}
		previousLink = link
		resp, err := server.Client().Get(link)
		require.NoError(t, err)
		actual, err := io.ReadAll(resp.Body)
		require.NoError(t, resp.Body.Close())
		require.NoError(t, err)
		require.Equal(t, 200, resp.StatusCode)
		require.Equal(t, plan.images[0].data, actual)
	}
}
