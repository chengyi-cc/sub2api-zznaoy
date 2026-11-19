package service

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/Wei-Shaw/sub2api/internal/pkg/httpclient"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/google/uuid"
)

// 设置键（settings 表）。
const (
	siteReconConfigKey  = "site_recon_config"
	siteReconBarkURLKey = "site_recon_bark_url" // 加密存储
	siteReconStateKey   = "site_recon_schedule_state"

	siteReconLeaderLockKey  = "site_recon:scheduler"
	siteReconLeaderLockTTL  = 2 * time.Minute
	siteReconTickInterval   = 30 * time.Second
	siteReconHeartbeatEvery = 15 * time.Second
	siteReconStaleAfter     = 2 * time.Minute
	siteReconMaxTokenLen    = 4096
)

var (
	ErrSiteReconRunning      = infraerrors.Conflict("SITE_RECON_RUNNING", "a reconciliation run is already in progress")
	ErrSiteReconInvalidInput = infraerrors.BadRequest("SITE_RECON_INVALID_INPUT", "invalid site reconciliation input")
	ErrSiteReconNotFound     = infraerrors.NotFound("SITE_RECON_NOT_FOUND", "site reconciliation record not found")
	ErrSiteReconNoBark       = infraerrors.BadRequest("SITE_RECON_NO_BARK", "bark push url is not configured")
)

// SiteReconConfig 对账配置（不含密钥）。
type SiteReconConfig struct {
	Timezone        string  `json:"timezone"`
	ScheduleEnabled bool    `json:"schedule_enabled"`
	ScheduleTime    string  `json:"schedule_time"`
	Days            int     `json:"days"`
	Scan            bool    `json:"scan"`
	Exclude         []int64 `json:"exclude"`
	PushEnabled     bool    `json:"push_enabled"`
	BalanceDays     float64 `json:"balance_days"`
}

func defaultSiteReconConfig() SiteReconConfig {
	return SiteReconConfig{Timezone: "Asia/Shanghai", ScheduleTime: "08:00", Days: 7, Scan: true, Exclude: []int64{}, PushEnabled: true, BalanceDays: 2}
}

func normalizeSiteReconConfig(c SiteReconConfig) (SiteReconConfig, error) {
	c.Timezone = strings.TrimSpace(c.Timezone)
	if c.Timezone == "" {
		c.Timezone = "Asia/Shanghai"
	}
	if _, err := siteReconLoadLocation(c.Timezone); err != nil {
		return c, infraerrors.BadRequest("SITE_RECON_INVALID_TIMEZONE", "unknown timezone (e.g. Asia/Shanghai)")
	}
	c.ScheduleTime = strings.TrimSpace(c.ScheduleTime)
	if _, _, ok := siteReconParseHHMM(c.ScheduleTime); !ok {
		if c.ScheduleTime != "" {
			return c, infraerrors.BadRequest("SITE_RECON_INVALID_TIME", "schedule time must be HH:MM")
		}
		c.ScheduleTime = "08:00"
	}
	if h, m, ok := siteReconParseHHMM(c.ScheduleTime); ok {
		c.ScheduleTime = fmt.Sprintf("%02d:%02d", h, m)
	}
	c.Days = max(1, min(90, c.Days))
	seen := map[int64]bool{}
	ex := make([]int64, 0, len(c.Exclude))
	for _, id := range c.Exclude {
		if id > 0 && !seen[id] {
			seen[id] = true
			ex = append(ex, id)
		}
	}
	sort.Slice(ex, func(i, j int) bool { return ex[i] < ex[j] })
	c.Exclude = ex
	if c.BalanceDays < 0 {
		c.BalanceDays = 0
	}
	return c, nil
}

// SiteReconService 站点对账：配置、上游令牌、运行、定时、报表。
type SiteReconService struct {
	store      *siteReconStore
	accounts   AccountRepository
	settings   SettingRepository
	encryptor  SecretEncryptor
	lockCache  LeaderLockCache
	http       *siteReconHTTP
	now        func() time.Time
	instanceID string

	// newSite 构造本站数据源；单测替换成假的。
	newSite func(tz string, loc *time.Location) siteReconSite
	// openUpstream 单测替换上游。
	openUpstream func(ctx context.Context, unit *siteReconUnit) (siteReconUpstream, string, string)
	// sendBark 单测替换推送。
	sendBark func(ctx context.Context, barkURL string, fields map[string]string) error

	mu      sync.Mutex
	running bool
	liveLog []string

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	start  sync.Once
	stop   sync.Once
}

func NewSiteReconService(db *sql.DB, accounts AccountRepository, settings SettingRepository, encryptor SecretEncryptor, lockCache LeaderLockCache) *SiteReconService {
	return newSiteReconService(db, true, accounts, settings, encryptor, lockCache)
}

func newSiteReconService(db *sql.DB, postgres bool, accounts AccountRepository, settings SettingRepository, encryptor SecretEncryptor, lockCache LeaderLockCache) *SiteReconService {
	ctx, cancel := context.WithCancel(context.Background())
	s := &SiteReconService{
		store:      newSiteReconStore(db, postgres),
		accounts:   accounts,
		settings:   settings,
		encryptor:  encryptor,
		lockCache:  lockCache,
		http:       newSiteReconHTTP(),
		now:        time.Now,
		instanceID: uuid.NewString(),
		ctx:        ctx,
		cancel:     cancel,
	}
	s.newSite = func(tz string, loc *time.Location) siteReconSite {
		return newSiteReconPGSite(db, accounts, tz, loc)
	}
	s.sendBark = siteReconSendBark
	return s
}

// ProvideSiteReconService 创建并启动定时器。
func ProvideSiteReconService(db *sql.DB, accounts AccountRepository, settings SettingRepository, encryptor SecretEncryptor, lockCache LeaderLockCache) *SiteReconService {
	s := NewSiteReconService(db, accounts, settings, encryptor, lockCache)
	s.Start()
	return s
}

func (s *SiteReconService) Start() {
	if s == nil {
		return
	}
	s.start.Do(func() {
		s.wg.Add(1)
		go s.schedulerLoop()
	})
}

func (s *SiteReconService) Stop() {
	if s == nil {
		return
	}
	s.stop.Do(func() {
		s.cancel()
		s.wg.Wait()
	})
}

// ---------------------------------------------------------------- 配置

func (s *SiteReconService) loadConfig(ctx context.Context) SiteReconConfig {
	cfg := defaultSiteReconConfig()
	if raw, err := s.settings.GetValue(ctx, siteReconConfigKey); err == nil && strings.TrimSpace(raw) != "" {
		_ = json.Unmarshal([]byte(raw), &cfg)
	}
	if n, err := normalizeSiteReconConfig(cfg); err == nil {
		return n
	}
	d := defaultSiteReconConfig()
	return d
}

func (s *SiteReconService) loadState(ctx context.Context) SiteReconScheduleState {
	var st SiteReconScheduleState
	if raw, err := s.settings.GetValue(ctx, siteReconStateKey); err == nil && raw != "" {
		_ = json.Unmarshal([]byte(raw), &st)
	}
	return st
}

func (s *SiteReconService) saveState(ctx context.Context, st SiteReconScheduleState) error {
	raw, _ := json.Marshal(st)
	return s.settings.Set(ctx, siteReconStateKey, string(raw))
}

func (s *SiteReconService) barkURL(ctx context.Context) string {
	raw, err := s.settings.GetValue(ctx, siteReconBarkURLKey)
	if err != nil || raw == "" || s.encryptor == nil {
		return ""
	}
	plain, err := s.encryptor.Decrypt(raw)
	if err != nil {
		return ""
	}
	return plain
}

// siteReconBarkHint Bark 地址只展示主机名，设备 key 不回显。
func siteReconBarkHint(raw string) string {
	if raw == "" {
		return ""
	}
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Scheme + "://" + u.Host + "/****"
	}
	return "****"
}

// SiteReconConfigView 设置页数据。
type SiteReconConfigView struct {
	Config         SiteReconConfig `json:"config"`
	BarkConfigured bool            `json:"bark_configured"`
	BarkHint       string          `json:"bark_hint"`
	NextRun        string          `json:"next_run"`
}

func (s *SiteReconService) GetConfig(ctx context.Context) (*SiteReconConfigView, error) {
	cfg := s.loadConfig(ctx)
	bark := s.barkURL(ctx)
	return &SiteReconConfigView{Config: cfg, BarkConfigured: bark != "", BarkHint: siteReconBarkHint(bark), NextRun: s.nextRun(ctx, cfg)}, nil
}

func (s *SiteReconService) nextRun(ctx context.Context, cfg SiteReconConfig) string {
	loc, err := siteReconLoadLocation(cfg.Timezone)
	if err != nil {
		return ""
	}
	return siteReconNextRun(s.now().In(loc), cfg.ScheduleEnabled, cfg.ScheduleTime, s.loadState(ctx))
}

// SiteReconConfigUpdate 设置页保存。BarkURL 为 nil 表示不改；ClearBark 清空推送地址。
type SiteReconConfigUpdate struct {
	SiteReconConfig
	BarkURL   *string `json:"bark_secret_url"`
	ClearBark bool    `json:"clear_bark"`
}

func (s *SiteReconService) UpdateConfig(ctx context.Context, in SiteReconConfigUpdate) (*SiteReconConfigView, error) {
	cfg, err := normalizeSiteReconConfig(in.SiteReconConfig)
	if err != nil {
		return nil, err
	}
	raw, _ := json.Marshal(cfg)
	if err := s.settings.Set(ctx, siteReconConfigKey, string(raw)); err != nil {
		return nil, err
	}
	switch {
	case in.ClearBark:
		if err := s.settings.Set(ctx, siteReconBarkURLKey, ""); err != nil {
			return nil, err
		}
	case in.BarkURL != nil && strings.TrimSpace(*in.BarkURL) != "":
		bark := strings.TrimSpace(*in.BarkURL)
		u, perr := url.Parse(bark)
		if perr != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
			return nil, infraerrors.BadRequest("SITE_RECON_INVALID_BARK_URL", "bark url must be an http(s) url")
		}
		if s.encryptor == nil {
			return nil, infraerrors.ServiceUnavailable("SITE_RECON_ENCRYPTOR_UNAVAILABLE", "secret encryptor unavailable")
		}
		enc, err := s.encryptor.Encrypt(bark)
		if err != nil {
			return nil, err
		}
		if err := s.settings.Set(ctx, siteReconBarkURLKey, enc); err != nil {
			return nil, err
		}
	}
	// 设定的时刻今天已经过了：从明天开始，不要一保存就立刻跑一次
	if loc, err := siteReconLoadLocation(cfg.Timezone); err == nil && cfg.ScheduleEnabled {
		now := s.now().In(loc)
		st := s.loadState(ctx)
		h, m, _ := siteReconParseHHMM(cfg.ScheduleTime)
		today := now.Format("2006-01-02")
		if st.LastDay != today && (now.Hour() > h || (now.Hour() == h && now.Minute() >= m)) {
			_ = s.saveState(ctx, SiteReconScheduleState{LastDay: today})
		}
	}
	return s.GetConfig(ctx)
}

func (s *SiteReconService) TestPush(ctx context.Context) error {
	bark := s.barkURL(ctx)
	if bark == "" {
		return ErrSiteReconNoBark
	}
	return s.sendBark(ctx, bark, map[string]string{"title": "站点对账", "subtitle": "测试推送", "body": "能看到这条，说明推送配置没问题。"})
}

// ---------------------------------------------------------------- 上游访问令牌

var siteReconHostRe = regexp.MustCompile(`^[a-z0-9.-]+(:[0-9]{1,5})?$`)

// siteReconCleanHost 允许粘贴整条网址：只留 host[:port]。
func siteReconCleanHost(h string) string {
	h = strings.TrimSpace(h)
	if i := strings.Index(h, "://"); i >= 0 {
		h = h[i+3:]
	}
	if i := strings.IndexAny(h, "/?#"); i >= 0 {
		h = h[:i]
	}
	return strings.ToLower(strings.TrimSpace(h))
}

func siteReconTokenHint(token string) string {
	if len(token) < 10 {
		return "****"
	}
	return "****" + token[len(token)-4:]
}

// SiteReconTokenView 令牌列表项：从不包含明文。
type SiteReconTokenView struct {
	Host      string `json:"host"`
	Hint      string `json:"hint"`
	UpdatedAt string `json:"updated_at"`
	Readable  bool   `json:"readable"`
}

type SiteReconTokensView struct {
	Tokens     []SiteReconTokenView `json:"tokens"`
	NeedTokens []string             `json:"need_tokens"`
}

func (s *SiteReconService) ListTokens(ctx context.Context) (*SiteReconTokensView, error) {
	rows, err := s.store.listTokens(ctx)
	if err != nil {
		return nil, err
	}
	out := &SiteReconTokensView{Tokens: []SiteReconTokenView{}, NeedTokens: []string{}}
	have := map[string]bool{}
	for _, r := range rows {
		readable := false
		if s.encryptor != nil {
			_, derr := s.encryptor.Decrypt(r.Ciphertext)
			readable = derr == nil
		}
		out.Tokens = append(out.Tokens, SiteReconTokenView{Host: r.Host, Hint: r.Hint, UpdatedAt: r.UpdatedAt, Readable: readable})
		have[r.Host] = true
	}
	// 最近一次成功对账里缺访问令牌的上游
	summary, err := s.store.lastOKSummary(ctx)
	if err != nil {
		return nil, err
	}
	if problems, ok := summary["problems"].(map[string]any); ok {
		if lines, ok := problems[SiteReconStatusNeedToken].([]any); ok {
			seen := map[string]bool{}
			for _, l := range lines {
				line, _ := l.(string)
				host := strings.TrimSpace(strings.SplitN(line, "  ←  ", 2)[0])
				if host != "" && !have[host] && !seen[host] {
					seen[host] = true
					out.NeedTokens = append(out.NeedTokens, host)
				}
			}
		}
	}
	sort.Strings(out.NeedTokens)
	return out, nil
}

func (s *SiteReconService) SetToken(ctx context.Context, rawHost, token string, by int64) error {
	host := siteReconCleanHost(rawHost)
	token = strings.TrimSpace(token)
	if host == "" || !siteReconHostRe.MatchString(host) || token == "" || len(token) > siteReconMaxTokenLen {
		return ErrSiteReconInvalidInput
	}
	if s.encryptor == nil {
		return infraerrors.ServiceUnavailable("SITE_RECON_ENCRYPTOR_UNAVAILABLE", "secret encryptor unavailable")
	}
	enc, err := s.encryptor.Encrypt(token)
	if err != nil {
		return err
	}
	return s.store.upsertToken(ctx, host, enc, siteReconTokenHint(token), by, s.now())
}

func (s *SiteReconService) DeleteToken(ctx context.Context, rawHost string) error {
	host := siteReconCleanHost(rawHost)
	if host == "" {
		return ErrSiteReconInvalidInput
	}
	ok, err := s.store.deleteToken(ctx, host)
	if err != nil {
		return err
	}
	if !ok {
		return ErrSiteReconNotFound
	}
	return nil
}

// decryptTokens 运行时才解密；解不开（加密密钥变了）的跳过并记日志。
func (s *SiteReconService) decryptTokens(ctx context.Context, log func(string)) map[string]string {
	out := map[string]string{}
	rows, err := s.store.listTokens(ctx)
	if err != nil {
		log("读取上游访问令牌失败: " + err.Error())
		return out
	}
	for _, r := range rows {
		if s.encryptor == nil {
			continue
		}
		plain, err := s.encryptor.Decrypt(r.Ciphertext)
		if err != nil {
			log(fmt.Sprintf("上游 %s 的访问令牌无法解密（加密密钥可能变了），请到设置页重新填写", r.Host))
			continue
		}
		out[r.Host] = plain
	}
	return out
}

// ---------------------------------------------------------------- 运行

// SiteReconStartInput 手动跑一次。
type SiteReconStartInput struct {
	Date   string  `json:"date"`
	Days   int     `json:"days"`
	Only   []int64 `json:"only"`
	Rescan bool    `json:"rescan"`
	Push   bool    `json:"push"`
}

func (s *SiteReconService) StartManualRun(ctx context.Context, in SiteReconStartInput) (int64, error) {
	cfg := s.loadConfig(ctx)
	loc, err := siteReconLoadLocation(cfg.Timezone)
	if err != nil {
		return 0, ErrSiteReconInvalidInput
	}
	target, err := time.ParseInLocation("2006-01-02", strings.TrimSpace(in.Date), loc)
	if err != nil {
		return 0, infraerrors.BadRequest("SITE_RECON_INVALID_DATE", "date must be YYYY-MM-DD")
	}
	if target.Format("2006-01-02") > s.now().In(loc).Format("2006-01-02") {
		return 0, infraerrors.BadRequest("SITE_RECON_FUTURE_DATE", "date cannot be in the future")
	}
	days := in.Days
	if days <= 0 {
		days = cfg.Days
	}
	days = max(1, min(90, days))
	opts := siteReconRunOptions{Target: target.Format("2006-01-02"), Days: days, Rescan: in.Rescan}
	if len(in.Only) > 0 {
		opts.Only = map[int64]bool{}
		for _, id := range in.Only {
			if id > 0 {
				opts.Only[id] = true
			}
		}
	}
	return s.startRun(ctx, "manual", opts, in.Push)
}

func (s *SiteReconService) startRun(ctx context.Context, trigger string, opts siteReconRunOptions, push bool) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		return 0, ErrSiteReconRunning
	}
	now := s.now()
	if err := s.store.abortStaleRuns(ctx, now.Add(-siteReconStaleAfter)); err != nil {
		return 0, err
	}
	if active, err := s.store.activeRun(ctx, now.Add(-siteReconStaleAfter)); err != nil {
		return 0, err
	} else if active != nil {
		return 0, ErrSiteReconRunning // 另一个实例正在跑
	}
	id, err := s.store.insertRun(ctx, trigger, opts.Target, opts.Days, now)
	if err != nil {
		return 0, err
	}
	s.running = true
	s.liveLog = nil
	s.wg.Add(1)
	go s.execute(id, trigger, opts, push)
	return id, nil
}

func (s *SiteReconService) emit(msg string) {
	s.mu.Lock()
	s.liveLog = append(s.liveLog, msg)
	s.mu.Unlock()
}

func (s *SiteReconService) logText() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.liveLog, "\n")
}

// execute 完整跑一次：对账 → 推送 → 写运行记录。不 panic，失败写进记录。
func (s *SiteReconService) execute(id int64, trigger string, opts siteReconRunOptions, push bool) {
	defer s.wg.Done()
	ctx := s.ctx
	started := s.now()
	cfg := s.loadConfig(ctx)
	loc, _ := siteReconLoadLocation(cfg.Timezone)
	if loc == nil {
		loc = time.FixedZone("Asia/Shanghai", 8*3600)
	}

	hbDone := make(chan struct{})
	go func() {
		t := time.NewTicker(siteReconHeartbeatEvery)
		defer t.Stop()
		for {
			select {
			case <-hbDone:
				return
			case <-t.C:
				_ = s.store.heartbeatRun(context.Background(), id, s.logText(), s.now())
			}
		}
	}()

	status := SiteReconRunOK
	var summary *siteReconSummary
	func() {
		defer func() {
			if r := recover(); r != nil {
				status = SiteReconRunFailed
				summary = &siteReconSummary{Target: opts.Target, Error: fmt.Sprint(r)}
				s.emit(fmt.Sprintf("对账失败: %v", r))
				logger.LegacyPrintf("service.site_recon", "[SiteRecon] run %d panic: %v", id, r)
			}
		}()
		engine := &siteReconEngine{
			store: s.store, site: s.newSite(cfg.Timezone, loc), http: s.http, cfg: cfg, loc: loc, now: s.now,
			log: s.emit, openUpstream: s.openUpstream,
		}
		engine.tokens = s.decryptTokens(ctx, s.emit)
		res, err := engine.run(ctx, opts)
		if err != nil {
			status = SiteReconRunFailed
			if ctx.Err() != nil {
				status = SiteReconRunAborted
			}
			summary = &siteReconSummary{Target: opts.Target, Days: opts.Days, Error: err.Error()}
			s.emit("对账失败: " + err.Error())
			return
		}
		summary = res
		keys := make([]string, 0, len(res.Problems))
		for k := range res.Problems {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			text := siteReconStatusText[k]
			if text == "" {
				text = k
			}
			s.emit(fmt.Sprintf("【%s】%d 个账号", text, len(res.Problems[k])))
			for _, line := range res.Problems[k] {
				s.emit("    " + line)
			}
		}
	}()

	finishCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if push && cfg.PushEnabled && status != SiteReconRunAborted {
		if bark := s.barkURL(finishCtx); bark != "" {
			var fields map[string]string
			var err error
			if status == SiteReconRunOK {
				fields, err = s.buildPush(finishCtx, summary.Target, cfg)
			} else {
				fields = siteReconFailurePush(summary.Error)
			}
			if err == nil {
				err = s.sendBark(finishCtx, bark, fields)
			}
			if err != nil {
				s.emit("推送失败: " + err.Error())
			} else {
				s.emit("已推送到手机")
			}
		}
	}
	summary.Elapsed = int64(s.now().Sub(started).Seconds())
	s.emit(fmt.Sprintf("完成，耗时 %d 秒", summary.Elapsed))
	close(hbDone)
	raw, _ := json.Marshal(summary)
	days := summary.Days
	if days == 0 {
		days = opts.Days
	}
	target := summary.Target
	if target == "" {
		target = opts.Target
	}
	if err := s.store.finishRun(finishCtx, id, target, days, status, string(raw), s.logText(), s.now()); err != nil {
		logger.LegacyPrintf("service.site_recon", "[SiteRecon] finish run %d: %v", id, err)
	}
	if trigger == "schedule" || trigger == "retry" {
		st := siteReconAfterScheduledRun(s.now().In(loc), s.loadState(finishCtx), status == SiteReconRunOK)
		_ = s.saveState(finishCtx, st)
	}
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
}

// ---------------------------------------------------------------- 定时

func (s *SiteReconService) schedulerLoop() {
	defer s.wg.Done()
	// 上次进程退出时还“进行中”的记录：心跳过期后标成中断
	_ = s.store.abortStaleRuns(s.ctx, s.now().Add(-siteReconStaleAfter))
	t := time.NewTicker(siteReconTickInterval)
	defer t.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-t.C:
			s.tick()
		}
	}
}

func (s *SiteReconService) tick() {
	defer func() {
		if r := recover(); r != nil {
			logger.LegacyPrintf("service.site_recon", "[SiteRecon] scheduler panic: %v", r)
		}
	}()
	ctx := s.ctx
	cfg := s.loadConfig(ctx)
	if !cfg.ScheduleEnabled {
		return
	}
	loc, err := siteReconLoadLocation(cfg.Timezone)
	if err != nil {
		return
	}
	s.mu.Lock()
	busy := s.running
	s.mu.Unlock()
	if busy || siteReconDue(s.now().In(loc), true, cfg.ScheduleTime, s.loadState(ctx)) == "" {
		return
	}
	// 多实例：只有拿到锁的实例去启动；状态写入后其他实例不会再判定为到点。
	if s.lockCache != nil {
		ok, err := s.lockCache.TryAcquireLeaderLock(ctx, siteReconLeaderLockKey, s.instanceID, siteReconLeaderLockTTL)
		if err != nil || !ok {
			return
		}
		defer func() { _ = s.lockCache.ReleaseLeaderLock(context.Background(), siteReconLeaderLockKey, s.instanceID) }()
	}
	now := s.now().In(loc)
	state := s.loadState(ctx)
	what := siteReconDue(now, true, cfg.ScheduleTime, state)
	if what == "" {
		return
	}
	if active, err := s.store.activeRun(ctx, s.now().Add(-siteReconStaleAfter)); err != nil || active != nil {
		return
	}
	if what == "schedule" {
		state = SiteReconScheduleState{LastDay: now.Format("2006-01-02")}
	} else {
		state.RetryAt = ""
	}
	if err := s.saveState(ctx, state); err != nil {
		return
	}
	logger.LegacyPrintf("service.site_recon", "[SiteRecon] %s 开始定时对账（%s）", now.Format("2006-01-02 15:04"), what)
	if _, err := s.startRun(ctx, what, siteReconRunOptions{}, true); err != nil && !errors.Is(err, ErrSiteReconRunning) {
		logger.LegacyPrintf("service.site_recon", "[SiteRecon] start scheduled run: %v", err)
	}
}

// ---------------------------------------------------------------- 运行页

type SiteReconRunsView struct {
	Running    bool            `json:"running"`
	ActiveID   int64           `json:"active_id"`
	ActiveLog  string          `json:"active_log"`
	Runs       []SiteReconRun  `json:"runs"`
	NextRun    string          `json:"next_run"`
	DefaultDay string          `json:"default_day"`
	Config     SiteReconConfig `json:"config"`
}

func (s *SiteReconService) Runs(ctx context.Context) (*SiteReconRunsView, error) {
	cfg := s.loadConfig(ctx)
	loc, _ := siteReconLoadLocation(cfg.Timezone)
	if loc == nil {
		loc = time.UTC
	}
	staleBefore := s.now().Add(-siteReconStaleAfter)
	runs, err := s.store.listRuns(ctx, 30)
	if err != nil {
		return nil, err
	}
	for i := range runs {
		if runs[i].Status == SiteReconRunRunning && runs[i].HeartbeatAt < staleBefore.Format(time.RFC3339) {
			runs[i].Status = SiteReconRunAborted
		}
	}
	view := &SiteReconRunsView{Runs: runs, NextRun: siteReconNextRun(s.now().In(loc), cfg.ScheduleEnabled, cfg.ScheduleTime, s.loadState(ctx)),
		DefaultDay: s.now().In(loc).AddDate(0, 0, -1).Format("2006-01-02"), Config: cfg}
	if view.Runs == nil {
		view.Runs = []SiteReconRun{}
	}
	active, err := s.store.activeRun(ctx, staleBefore)
	if err != nil {
		return nil, err
	}
	if active != nil {
		view.Running, view.ActiveID, view.ActiveLog = true, active.ID, active.Log
		s.mu.Lock()
		local := s.running
		s.mu.Unlock()
		if local {
			view.ActiveLog = s.logText() // 本实例在跑：直接给内存里的最新日志
		}
	}
	return view, nil
}

func (s *SiteReconService) GetRun(ctx context.Context, id int64) (*SiteReconRun, error) {
	r, err := s.store.getRun(ctx, id)
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, ErrSiteReconNotFound
	}
	return r, nil
}

// ---------------------------------------------------------------- 报表

type SiteReconOverview struct {
	Day    string                 `json:"day"`
	Days   []string               `json:"days"`
	Totals SiteReconTotals        `json:"totals"`
	Prev   SiteReconTotals        `json:"prev"`
	Alerts []SiteReconAlert       `json:"alerts"`
	Trend  []SiteReconTrendPoint  `json:"trend"`
	Groups []*SiteReconGroupPnL   `json:"groups"`
	Counts SiteReconAnomalyCounts `json:"anomaly_counts"`
}

type SiteReconDetail struct {
	Day         string                 `json:"day"`
	Days        []string               `json:"days"`
	Accounts    []SiteReconAccountRow  `json:"accounts"`
	Wallets     []*SiteReconWallet     `json:"wallets"`
	BalanceDays float64                `json:"balance_days"`
	Models      []SiteReconModelReport `json:"models"`
	Pending     []SiteReconPending     `json:"pending"`
	History     []SiteReconHistoryRow  `json:"history"`
}

type SiteReconAnomalies struct {
	Day     string                   `json:"day"`
	Days    []string                 `json:"days"`
	Scanned bool                     `json:"scanned"`
	Stats   []SiteReconAnomalyStat   `json:"stats"`
	Samples []SiteReconAnomalySample `json:"samples"`
	History []SiteReconAnomalyCounts `json:"history"`
}

func (s *SiteReconService) resolveDay(ctx context.Context, day string) (string, []string, error) {
	days, err := s.store.daysAvailable(ctx)
	if err != nil {
		return "", nil, err
	}
	if days == nil {
		days = []string{}
	}
	for _, d := range days {
		if d == day {
			return day, days, nil
		}
	}
	if len(days) > 0 {
		return days[0], days, nil
	}
	return "", days, nil
}

func (s *SiteReconService) Overview(ctx context.Context, day string) (*SiteReconOverview, error) {
	day, days, err := s.resolveDay(ctx, day)
	if err != nil {
		return nil, err
	}
	out := &SiteReconOverview{Day: day, Days: days, Alerts: []SiteReconAlert{}, Trend: []SiteReconTrendPoint{}, Groups: []*SiteReconGroupPnL{}}
	if day == "" {
		return out, nil
	}
	cfg := s.loadConfig(ctx)
	if out.Totals, err = s.store.dayTotals(ctx, day); err != nil {
		return nil, err
	}
	if out.Prev, err = s.store.dayTotals(ctx, siteReconShiftDay(day, -1)); err != nil {
		return nil, err
	}
	if a, err := s.store.alerts(ctx, day, cfg.BalanceDays); err != nil {
		return nil, err
	} else if a != nil {
		out.Alerts = a
	}
	if tr, err := s.store.trend(ctx, day, 30); err != nil {
		return nil, err
	} else if tr != nil {
		out.Trend = tr
	}
	if g, err := s.store.groupPnL(ctx, day); err != nil {
		return nil, err
	} else if g != nil {
		out.Groups = g
	}
	if out.Counts, err = s.store.anomalyCounts(ctx, day); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SiteReconService) Detail(ctx context.Context, day string) (*SiteReconDetail, error) {
	day, days, err := s.resolveDay(ctx, day)
	if err != nil {
		return nil, err
	}
	cfg := s.loadConfig(ctx)
	out := &SiteReconDetail{Day: day, Days: days, BalanceDays: cfg.BalanceDays, Accounts: []SiteReconAccountRow{},
		Wallets: []*SiteReconWallet{}, Models: []SiteReconModelReport{}, Pending: []SiteReconPending{}, History: []SiteReconHistoryRow{}}
	if h, err := s.store.history(ctx, 60); err != nil {
		return nil, err
	} else if h != nil {
		out.History = h
	}
	if day == "" {
		return out, nil
	}
	if a, err := s.store.accounts(ctx, day); err != nil {
		return nil, err
	} else if a != nil {
		out.Accounts = a
	}
	if w, err := s.store.wallets(ctx, day, cfg.BalanceDays); err != nil {
		return nil, err
	} else if w != nil {
		out.Wallets = w
	}
	if m, err := s.store.modelReport(ctx, day); err != nil {
		return nil, err
	} else if m != nil {
		out.Models = m
	}
	if p, err := s.store.pending(ctx, day); err != nil {
		return nil, err
	} else if p != nil {
		out.Pending = p
	}
	return out, nil
}

func (s *SiteReconService) Anomalies(ctx context.Context, day string) (*SiteReconAnomalies, error) {
	day, days, err := s.resolveDay(ctx, day)
	if err != nil {
		return nil, err
	}
	out := &SiteReconAnomalies{Day: day, Days: days, Stats: []SiteReconAnomalyStat{}, Samples: []SiteReconAnomalySample{}, History: []SiteReconAnomalyCounts{}}
	if h, err := s.store.anomalyHistory(ctx, 14); err != nil {
		return nil, err
	} else if h != nil {
		out.History = h
	}
	if day == "" {
		return out, nil
	}
	t, err := s.store.dayTotals(ctx, day)
	if err != nil {
		return nil, err
	}
	out.Scanned = t.Scanned > 0
	if st, err := s.store.anomalyStats(ctx, day); err != nil {
		return nil, err
	} else if st != nil {
		out.Stats = st
	}
	if sm, err := s.store.anomalySamples(ctx, day); err != nil {
		return nil, err
	} else if sm != nil {
		out.Samples = sm
	}
	return out, nil
}

// ---------------------------------------------------------------- Bark 推送

var siteReconMDEscape = regexp.MustCompile("([\\\\`*_\\[\\]#~|])")

func siteReconMD(s string) string { return siteReconMDEscape.ReplaceAllString(s, `\$1`) }

func siteReconShort(name string, n int) string {
	name = strings.Join(strings.Fields(name), " ")
	r := []rune(name)
	if len(r) <= n {
		return name
	}
	return string(r[:n-1]) + "…"
}

// buildPush 对账完成后的手机推送：要处理的事放最前面。
func (s *SiteReconService) buildPush(ctx context.Context, target string, cfg SiteReconConfig) (map[string]string, error) {
	t, err := s.store.dayTotals(ctx, target)
	if err != nil {
		return nil, err
	}
	p, err := s.store.dayTotals(ctx, siteReconShiftDay(target, -1))
	if err != nil {
		return nil, err
	}
	gp, pgp := t.Rev-t.Cost, p.Rev-p.Cost
	margin := "-"
	if t.Rev != 0 {
		margin = fmt.Sprintf("%.1f%%", gp/t.Rev*100)
	}
	change := ""
	if p.Has && pgp != 0 {
		change = fmt.Sprintf(" · 较前一天 %+.1f%%", (gp-pgp)/abs64(pgp)*100)
	}
	groups, err := s.store.groupPnL(ctx, target)
	if err != nil {
		return nil, err
	}
	var top, losers []*SiteReconGroupPnL
	for _, g := range groups {
		if g.Known <= 0 || g.Profit == nil {
			continue
		}
		if *g.Profit >= 0 {
			top = append(top, g)
		} else {
			losers = append(losers, g)
		}
	}
	sort.Slice(top, func(i, j int) bool { return *top[i].Profit > *top[j].Profit })
	if len(top) > 5 {
		top = top[:5]
	}
	sort.Slice(losers, func(i, j int) bool { return *losers[i].Profit < *losers[j].Profit })
	alerts, err := s.store.alerts(ctx, target, cfg.BalanceDays)
	if err != nil {
		return nil, err
	}
	var urgent, minor []string
	for _, a := range alerts {
		switch a.Level {
		case "bad":
			urgent = append(urgent, a.Text)
		case "warn":
			minor = append(minor, a.Text)
		}
	}
	var blocks, body []string
	if len(urgent) > 0 {
		lines := make([]string, len(urgent))
		for i, n := range urgent {
			lines[i] = "- " + siteReconMD(n)
			body = append(body, "！"+n)
		}
		blocks = append(blocks, "### 要处理\n"+strings.Join(lines, "\n"))
	}
	blocks = append(blocks, fmt.Sprintf("收入 **¥%s**　成本 **¥%s**", siteReconFormatMoney(t.Rev), siteReconFormatMoney(t.Cost)))
	body = append(body, fmt.Sprintf("收入 ¥%s　成本 ¥%s", siteReconFormatMoney(t.Rev), siteReconFormatMoney(t.Cost)))
	if len(top) > 0 {
		lines := make([]string, len(top))
		for i, g := range top {
			m := 0.0
			if g.Margin != nil {
				m = *g.Margin * 100
			}
			lines[i] = fmt.Sprintf("- **¥%s**　%.0f%%　%s", siteReconFormatMoney(*g.Profit), m, siteReconMD(siteReconShort(g.Name, 24)))
			body = append(body, fmt.Sprintf("¥%s %.0f%% %s", siteReconFormatMoney(*g.Profit), m, siteReconShort(g.Name, 24)))
		}
		blocks = append(blocks, "### 分组毛利\n"+strings.Join(lines, "\n"))
	}
	if len(losers) > 0 {
		lines := make([]string, len(losers))
		for i, g := range losers {
			lines[i] = fmt.Sprintf("- **-¥%s**　%s", siteReconFormatMoney(-*g.Profit), siteReconMD(siteReconShort(g.Name, 24)))
			body = append(body, fmt.Sprintf("亏 ¥%s %s", siteReconFormatMoney(-*g.Profit), siteReconShort(g.Name, 24)))
		}
		blocks = append(blocks, "### 亏损\n"+strings.Join(lines, "\n"))
	}
	if len(minor) > 0 {
		lines := make([]string, len(minor))
		for i, n := range minor {
			lines[i] = "- " + siteReconMD(n)
		}
		blocks = append(blocks, "### 留意\n"+strings.Join(lines, "\n"))
		body = append(body, minor...)
	}
	title := fmt.Sprintf("%s 毛利 ¥%s", target[min(5, len(target)):], siteReconFormatMoney(gp))
	if len(urgent) > 0 {
		title += fmt.Sprintf("（%d 件事要处理）", len(urgent))
	}
	return map[string]string{
		"title":    title,
		"subtitle": "毛利率 " + margin + change,
		"markdown": strings.Join(blocks, "\n\n"),
		"body":     strings.Join(body, "\n"),
	}, nil
}

func abs64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func siteReconFailurePush(reason string) map[string]string {
	if reason == "" {
		reason = "未知错误"
	}
	r := []rune(reason)
	if len(r) > 300 {
		reason = string(r[:300])
	}
	return map[string]string{"title": "对账失败", "subtitle": "今天的报告没有生成", "body": reason + "\n到“运行”页看日志，修好后可手动重跑。"}
}

func siteReconSendBark(ctx context.Context, barkURL string, fields map[string]string) error {
	payload := map[string]string{"group": "站点对账"}
	for k, v := range fields {
		payload[k] = v
	}
	raw, _ := json.Marshal(payload)
	client, err := httpclient.GetClient(httpclient.Options{Timeout: 20 * time.Second})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(barkURL, "/"), bytes.NewReader(raw))
	if err != nil {
		return errors.New("Bark 地址无效")
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Bark 请求失败: %s", siteReconNetReason(err))
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	var out struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &out) != nil || out.Code != 200 {
		return fmt.Errorf("Bark 返回 HTTP %d: %s", resp.StatusCode, strings.TrimSpace(out.Message))
	}
	return nil
}
