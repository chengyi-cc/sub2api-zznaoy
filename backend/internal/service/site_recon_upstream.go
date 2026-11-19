package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/httpclient"
	"golang.org/x/sync/errgroup"
)

// 站点对账的上游取数：sub2api 上游（账号 key 调 /v1/usage）与 NewAPI 上游（访问令牌查消费日志）。
// 金额口径与原服务一致：sub2api 直接取 actual_cost；NewAPI 用 quota / quota_per_unit。

const (
	siteReconUA              = "sub2api-site-recon/1.0"
	siteReconHTTPTimeout     = 120 * time.Second
	siteReconDetectTimeout   = 20 * time.Second
	siteReconUpstreamWorkers = 30 // 上游翻页并发
	siteReconSub2APIMaxDays  = 90
	siteReconNewAPIPage      = 100
	siteReconMaxBodyBytes    = 32 << 20
)

var siteReconRetryStatus = map[int]bool{408: true, 425: true, 429: true, 500: true, 502: true, 503: true, 504: true,
	520: true, 521: true, 522: true, 523: true, 524: true}

var siteReconRetryWait = []time.Duration{1500 * time.Millisecond, 4 * time.Second, 9 * time.Second}

// 官方上游没有“按 key 查扣费”的接口，无法对账。
var siteReconOfficialHosts = map[string]bool{
	"api.openai.com": true, "api.anthropic.com": true, "generativelanguage.googleapis.com": true,
	"api.deepseek.com": true, "api.x.ai": true, "cli-chat-proxy.grok.com": true,
}

// siteReconHTTP 带重试的 JSON GET。
type siteReconHTTP struct {
	clientFor func(proxyURL string, timeout time.Duration) (*http.Client, error)
	sleep     func(ctx context.Context, d time.Duration) error
}

func newSiteReconHTTP() *siteReconHTTP {
	return &siteReconHTTP{clientFor: siteReconClient, sleep: siteReconSleep}
}

func siteReconSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// siteReconClient 复用共享连接池，但每次包一层自己的 Client：禁止跟随跳转，
// 防止携带令牌的请求被 30x 带到别的主机。
func siteReconClient(proxyURL string, timeout time.Duration) (*http.Client, error) {
	base, err := httpclient.GetClient(httpclient.Options{ProxyURL: proxyURL, ResponseHeaderTimeout: timeout})
	if err != nil {
		return nil, err
	}
	return &http.Client{
		Transport:     base.Transport,
		Timeout:       timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}, nil
}

// getJSON GET 一个 JSON。超时、断线、网关错误、限流会自动重试；4xx（key 无效等）立即报错。
// 错误信息只带不含 query 的地址，避免把查询参数里的东西写进日志。
func (h *siteReconHTTP) getJSON(ctx context.Context, rawURL, proxyURL string, headers map[string]string, timeout time.Duration, retries int, out any) error {
	where := rawURL
	if i := strings.IndexByte(where, '?'); i >= 0 {
		where = where[:i]
	}
	client, err := h.clientFor(proxyURL, timeout)
	if err != nil {
		return fmt.Errorf("代理不可用 %s: %v", where, err)
	}
	for attempt := 0; attempt <= retries; attempt++ {
		last := attempt == retries
		status, body, err := h.once(ctx, client, rawURL, headers)
		switch {
		case err == nil && status >= 200 && status < 300:
			if jerr := json.Unmarshal(body, out); jerr == nil {
				return nil
			} else if last {
				return fmt.Errorf("请求失败（已重试 %d 次）%s: 返回的不是 JSON", retries, where)
			}
		case err == nil:
			if !siteReconRetryStatus[status] || last {
				snippet := string(body)
				if len(snippet) > 300 {
					snippet = snippet[:300]
				}
				return fmt.Errorf("HTTP %d %s: %s", status, where, strings.TrimSpace(snippet))
			}
		default:
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if last {
				return fmt.Errorf("请求失败（已重试 %d 次）%s: %s", retries, where, siteReconNetReason(err))
			}
		}
		wait := siteReconRetryWait[min(attempt, len(siteReconRetryWait)-1)]
		if err := h.sleep(ctx, wait); err != nil {
			return err
		}
	}
	return fmt.Errorf("请求失败 %s", where)
}

func (h *siteReconHTTP) once(ctx context.Context, client *http.Client, rawURL string, headers map[string]string) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("User-Agent", siteReconUA)
	req.Header.Set("Accept", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, siteReconMaxBodyBytes))
	if err != nil {
		return 0, nil, err
	}
	return resp.StatusCode, body, nil
}

// siteReconNetReason 网络错误只保留原因，去掉 net/http 附带的完整 URL（可能含 query）。
func siteReconNetReason(err error) string {
	var uerr *url.Error
	if errors.As(err, &uerr) {
		err = uerr.Err
	}
	var nerr net.Error
	if errors.As(err, &nerr) && nerr.Timeout() {
		return "超时"
	}
	return err.Error()
}

// ---------------------------------------------------------------- 识别上游类型

type siteReconDetected struct {
	Kind   string // sub2api / newapi / official / unknown
	Status map[string]any
}

// siteReconDetector 按上游根地址识别程序类型。每次对账新建一个：同一上游一次对账只探测一次，
// 但失败结果不会留到下一次。
type siteReconDetector struct {
	h     *siteReconHTTP
	mu    sync.Mutex
	locks map[string]*sync.Mutex
	cache map[string]siteReconDetected
}

func newSiteReconDetector(h *siteReconHTTP) *siteReconDetector {
	return &siteReconDetector{h: h, locks: map[string]*sync.Mutex{}, cache: map[string]siteReconDetected{}}
}

func (d *siteReconDetector) detect(ctx context.Context, origin, proxyURL string) siteReconDetected {
	d.mu.Lock()
	lock := d.locks[origin]
	if lock == nil {
		lock = &sync.Mutex{}
		d.locks[origin] = lock
	}
	d.mu.Unlock()
	lock.Lock()
	defer lock.Unlock()
	d.mu.Lock()
	cached, ok := d.cache[origin]
	d.mu.Unlock()
	if ok {
		return cached
	}
	res := d.probe(ctx, origin, proxyURL)
	d.mu.Lock()
	d.cache[origin] = res
	d.mu.Unlock()
	return res
}

func (d *siteReconDetector) probe(ctx context.Context, origin, proxyURL string) siteReconDetected {
	if u, err := url.Parse(origin); err == nil && siteReconOfficialHosts[strings.ToLower(u.Host)] {
		return siteReconDetected{Kind: "official"}
	}
	var st struct {
		Data map[string]any `json:"data"`
	}
	if err := d.h.getJSON(ctx, origin+"/api/status", proxyURL, nil, siteReconDetectTimeout, 1, &st); err == nil && st.Data != nil {
		if _, ok := st.Data["quota_per_unit"]; ok {
			return siteReconDetected{Kind: "newapi", Status: st.Data}
		}
	}
	var pub struct {
		Code *int `json:"code"`
	}
	if err := d.h.getJSON(ctx, origin+"/api/v1/settings/public", proxyURL, nil, siteReconDetectTimeout, 1, &pub); err == nil && pub.Code != nil && *pub.Code == 0 {
		return siteReconDetected{Kind: "sub2api"}
	}
	return siteReconDetected{Kind: "unknown"}
}

// ---------------------------------------------------------------- 上游统一接口

type siteReconUpstream interface {
	// Daily 返回窗口内每天的上游汇总（超出上游可查范围的日子不在 map 里）和上游余额（拿不到为 nil）。
	Daily(ctx context.Context, days []string) (map[string]siteReconUpDay, *float64, error)
	// DayModels 某天按模型汇总。
	DayModels(ctx context.Context, day string) (map[string]siteReconUpModel, error)
	Describe() string
}

// siteReconSub2APIUpstream 上游是 sub2api：用账号的 API key 调 /v1/usage，无需登录。只有按天 / 按模型汇总。
type siteReconSub2APIUpstream struct {
	h        *siteReconHTTP
	origin   string
	apiKey   string
	proxyURL string
	tz       string
	loc      *time.Location
	now      func() time.Time
}

func (u *siteReconSub2APIUpstream) Describe() string { return "上游 sub2api" }

func (u *siteReconSub2APIUpstream) usage(ctx context.Context, q url.Values, out any) error {
	return u.h.getJSON(ctx, u.origin+"/v1/usage?"+q.Encode(), u.proxyURL,
		map[string]string{"Authorization": "Bearer " + u.apiKey}, siteReconHTTPTimeout, 3, out)
}

func (u *siteReconSub2APIUpstream) Daily(ctx context.Context, days []string) (map[string]siteReconUpDay, *float64, error) {
	today := u.now().In(u.loc)
	todayDate := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, time.UTC)
	earliest := days[0]
	for _, d := range days {
		if d < earliest {
			earliest = d
		}
	}
	first, err := time.Parse("2006-01-02", earliest)
	if err != nil {
		return nil, nil, err
	}
	n := int(todayDate.Sub(first).Hours()/24) + 1
	n = max(1, min(n, siteReconSub2APIMaxDays))
	var resp struct {
		Balance    *float64 `json:"balance"`
		DailyUsage []struct {
			Date       string  `json:"date"`
			Requests   int64   `json:"requests"`
			Cost       float64 `json:"cost"`
			ActualCost float64 `json:"actual_cost"`
		} `json:"daily_usage"`
	}
	q := url.Values{}
	q.Set("days", strconv.Itoa(n))
	q.Set("timezone", u.tz)
	if err := u.usage(ctx, q, &resp); err != nil {
		return nil, nil, err
	}
	got := make(map[string]siteReconUpDay, len(resp.DailyUsage))
	for _, d := range resp.DailyUsage {
		got[d.Date] = siteReconUpDay{Requests: d.Requests, ListCost: siteReconF64(d.Cost), Cost: d.ActualCost}
	}
	// 上游只返回有用量的日子：查询范围内没出现的日子 = 这把 key 当天用量为 0。
	oldest := todayDate.AddDate(0, 0, -(n - 1)).Format("2006-01-02")
	out := make(map[string]siteReconUpDay, len(days))
	for _, day := range days {
		if v, ok := got[day]; ok {
			out[day] = v
		} else if day >= oldest {
			out[day] = siteReconUpDay{ListCost: siteReconF64(0)}
		}
	}
	return out, resp.Balance, nil
}

func (u *siteReconSub2APIUpstream) DayModels(ctx context.Context, day string) (map[string]siteReconUpModel, error) {
	var resp struct {
		ModelStats []struct {
			Model      string  `json:"model"`
			Requests   int64   `json:"requests"`
			Cost       float64 `json:"cost"`
			ActualCost float64 `json:"actual_cost"`
		} `json:"model_stats"`
	}
	q := url.Values{}
	q.Set("start_date", day)
	q.Set("end_date", day)
	q.Set("days", "1")
	q.Set("timezone", u.tz)
	if err := u.usage(ctx, q, &resp); err != nil {
		return nil, err
	}
	out := make(map[string]siteReconUpModel, len(resp.ModelStats))
	for _, m := range resp.ModelStats {
		out[m.Model] = siteReconUpModel{Requests: m.Requests, ListCost: m.Cost, Cost: m.ActualCost}
	}
	return out, nil
}

// siteReconNewAPIUpstream 上游是 NewAPI：用上游后台的访问令牌（只需“用量-查看”权限）按 key 名筛选消费日志。
type siteReconNewAPIUpstream struct {
	h           *siteReconHTTP
	origin      string
	accessToken string
	proxyURL    string
	loc         *time.Location
	qpu         float64
	tokenName   string
}

func newSiteReconNewAPIUpstream(ctx context.Context, h *siteReconHTTP, origin, apiKey, accessToken, proxyURL string, loc *time.Location, status map[string]any) (*siteReconNewAPIUpstream, error) {
	qpu := 500000.0
	if v, ok := status["quota_per_unit"].(float64); ok && v > 0 {
		qpu = v
	}
	var info struct {
		Data struct {
			Name string `json:"name"`
		} `json:"data"`
	}
	// key 在上游叫什么名字，只凭 key 调 /api/usage/token/ 就能拿到。
	if err := h.getJSON(ctx, origin+"/api/usage/token/", proxyURL, map[string]string{"Authorization": "Bearer " + apiKey}, siteReconHTTPTimeout, 3, &info); err != nil {
		return nil, err
	}
	if info.Data.Name == "" {
		return nil, fmt.Errorf("上游 /api/usage/token/ 没有返回 key 名")
	}
	return &siteReconNewAPIUpstream{h: h, origin: origin, accessToken: accessToken, proxyURL: proxyURL, loc: loc, qpu: qpu, tokenName: info.Data.Name}, nil
}

func (u *siteReconNewAPIUpstream) Describe() string {
	return "上游 newapi，key 名 = " + u.tokenName
}

func (u *siteReconNewAPIUpstream) get(ctx context.Context, path string, q url.Values, out any) error {
	var body struct {
		Success bool            `json:"success"`
		Message string          `json:"message"`
		Data    json.RawMessage `json:"data"`
	}
	target := u.origin + path
	if len(q) > 0 {
		target += "?" + q.Encode()
	}
	if err := u.h.getJSON(ctx, target, u.proxyURL, map[string]string{"Authorization": "Bearer " + u.accessToken}, siteReconHTTPTimeout, 3, &body); err != nil {
		return err
	}
	if !body.Success {
		return fmt.Errorf("%s: %s", path, body.Message)
	}
	return json.Unmarshal(body.Data, out)
}

func (u *siteReconNewAPIUpstream) filter(day string) (url.Values, error) {
	start, end, err := siteReconDayWindow(day, u.loc)
	if err != nil {
		return nil, err
	}
	q := url.Values{}
	q.Set("type", "2")
	q.Set("token_name", u.tokenName)
	q.Set("start_timestamp", strconv.FormatInt(start.Unix(), 10))
	q.Set("end_timestamp", strconv.FormatInt(end.Unix()-1, 10)) // new-api 用 <=
	return q, nil
}

func (u *siteReconNewAPIUpstream) Daily(ctx context.Context, days []string) (map[string]siteReconUpDay, *float64, error) {
	out := make(map[string]siteReconUpDay, len(days))
	var mu sync.Mutex
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(siteReconUpstreamWorkers)
	for _, day := range days {
		g.Go(func() error {
			f, err := u.filter(day)
			if err != nil {
				return err
			}
			var stat struct {
				Quota float64 `json:"quota"`
			}
			if err := u.get(gctx, "/api/log/self/stat", f, &stat); err != nil {
				return err
			}
			p := cloneValues(f)
			p.Set("p", "1")
			p.Set("page_size", "1")
			var page struct {
				Total int64 `json:"total"`
			}
			if err := u.get(gctx, "/api/log/self", p, &page); err != nil {
				return err
			}
			mu.Lock()
			out[day] = siteReconUpDay{Requests: page.Total, Cost: stat.Quota / u.qpu}
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, nil, err
	}
	return out, u.balance(ctx), nil
}

// balance 余额在 /api/user/self 的 quota 里，需要令牌带“个人资料-查看”；没有就返回 nil。
// 注意这是上游整个用户的余额，同一上游下的多个账号会显示同一个数。
func (u *siteReconNewAPIUpstream) balance(ctx context.Context) *float64 {
	var self struct {
		Quota float64 `json:"quota"`
	}
	if err := u.get(ctx, "/api/user/self", nil, &self); err != nil {
		return nil
	}
	return siteReconF64(self.Quota / u.qpu)
}

type siteReconNewAPILog struct {
	ModelName string  `json:"model_name"`
	Quota     float64 `json:"quota"`
	Other     string  `json:"other"`
}

func (u *siteReconNewAPIUpstream) logs(ctx context.Context, day string) ([]siteReconNewAPILog, error) {
	f, err := u.filter(day)
	if err != nil {
		return nil, err
	}
	pageQuery := func(p int) url.Values {
		q := cloneValues(f)
		q.Set("p", strconv.Itoa(p))
		q.Set("page_size", strconv.Itoa(siteReconNewAPIPage))
		return q
	}
	type page struct {
		Total int64                `json:"total"`
		Items []siteReconNewAPILog `json:"items"`
	}
	var first page
	if err := u.get(ctx, "/api/log/self", pageQuery(1), &first); err != nil {
		return nil, err
	}
	pages := int((first.Total + siteReconNewAPIPage - 1) / siteReconNewAPIPage)
	rest := make([][]siteReconNewAPILog, max(pages-1, 0))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(siteReconUpstreamWorkers)
	for p := 2; p <= pages; p++ {
		g.Go(func() error {
			var pg page
			if err := u.get(gctx, "/api/log/self", pageQuery(p), &pg); err != nil {
				return err
			}
			rest[p-2] = pg.Items
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	rows := append([]siteReconNewAPILog(nil), first.Items...)
	for _, items := range rest {
		rows = append(rows, items...)
	}
	if int64(len(rows)) != first.Total {
		return nil, fmt.Errorf("上游日志条数不一致：应 %d，实得 %d", first.Total, len(rows))
	}
	return rows, nil
}

func (u *siteReconNewAPIUpstream) DayModels(ctx context.Context, day string) (map[string]siteReconUpModel, error) {
	rows, err := u.logs(ctx, day)
	if err != nil {
		return nil, err
	}
	out := make(map[string]siteReconUpModel)
	for _, r := range rows {
		m := out[r.ModelName]
		cost := r.Quota / u.qpu
		m.Requests++
		m.Cost += cost
		// 上游原价 = 实扣 / 上游分组倍率（倍率写在 other 里）
		ratio := 0.0
		var other struct {
			GroupRatio any `json:"group_ratio"`
		}
		if json.Unmarshal([]byte(r.Other), &other) == nil {
			switch v := other.GroupRatio.(type) {
			case float64:
				ratio = v
			case string:
				ratio, _ = strconv.ParseFloat(v, 64)
			}
		}
		if ratio > 0 {
			m.ListCost += cost / ratio
		} else {
			m.ListCost += cost
		}
		out[r.ModelName] = m
	}
	return out, nil
}

func cloneValues(v url.Values) url.Values {
	out := make(url.Values, len(v))
	for k, vals := range v {
		out[k] = append([]string(nil), vals...)
	}
	return out
}

// siteReconClassifyUpstreamError 上游错误归类：key 失效单独标出，其余为取数出错。
func siteReconClassifyUpstreamError(text string) (string, string) {
	dead := false
	for _, c := range []string{"INVALID_API_KEY", "GROUP_DELETED", "GROUP_NOT_ALLOWED", "API_KEY_DISABLED", "API_KEY_EXPIRED"} {
		if strings.Contains(text, c) {
			dead = true
			break
		}
	}
	// new-api：账号自己的 key 查 /api/usage/token/ 返回 401，说明 key 已失效 / 禁用
	if strings.Contains(text, "/api/usage/token/") && strings.Contains(text, "HTTP 401") {
		dead = true
	}
	if dead {
		return SiteReconStatusDeadKey, "上游取数失败: " + text
	}
	return SiteReconStatusUpstreamError, "上游取数失败: " + text
}

// siteReconRedact 从错误文字里抹掉密钥（上游偶尔会把请求头回显在报错里）。
func siteReconRedact(text string, secrets ...string) string {
	for _, s := range secrets {
		if len(s) >= 6 {
			text = strings.ReplaceAll(text, s, "***")
		}
	}
	return text
}
