package service

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

const siteReconAccountWorkers = 10 // 同时对账的单元数

// siteReconUnit 一个对账单元 = 一把上游 key。共用同一把 key 的多个本站账号合并，数据挂在 IDs[0] 下。
type siteReconUnit struct {
	IDs      []int64
	Name     string
	Key      string
	Host     string
	Origin   string
	APIKey   string
	ProxyURL string
}

// siteReconRunOptions 一次对账的参数。
type siteReconRunOptions struct {
	Target string // 目标日；空 = 昨天
	Days   int    // 同时刷新的天数；0 = 配置值
	Scan   *bool  // nil = 配置值
	Rescan bool   // 窗口内每天都重新逐条扫描
	Only   map[int64]bool
}

// siteReconSummary 一次对账的摘要。
type siteReconSummary struct {
	Target   string              `json:"target"`
	FirstDay string              `json:"first_day,omitempty"`
	Days     int                 `json:"days,omitempty"`
	Listed   int                 `json:"listed,omitempty"`
	Units    int                 `json:"units,omitempty"`
	Problems map[string][]string `json:"problems,omitempty"`
	Error    string              `json:"error,omitempty"`
	Elapsed  int64               `json:"elapsed"`
}

func siteReconDecodeSummary(raw string) map[string]any {
	out := map[string]any{}
	if strings.TrimSpace(raw) == "" {
		return out
	}
	_ = json.Unmarshal([]byte(raw), &out)
	return out
}

// siteReconEngine 跑一次对账需要的全部依赖。
type siteReconEngine struct {
	store    *siteReconStore
	site     siteReconSite
	http     *siteReconHTTP
	tokens   map[string]string // host -> NewAPI 访问令牌（已解密，仅本次运行使用）
	cfg      SiteReconConfig
	loc      *time.Location
	now      func() time.Time
	log      func(string)
	detector *siteReconDetector

	// openUpstream 可在单测里替换。返回 (上游或 nil, 状态, 错误文字)。
	openUpstream func(ctx context.Context, unit *siteReconUnit) (siteReconUpstream, string, string)
}

type siteReconRunCtx struct {
	target  string
	dayList []string
	scan    bool
	rescan  bool
}

func (e *siteReconEngine) defaultOpenUpstream(ctx context.Context, unit *siteReconUnit) (siteReconUpstream, string, string) {
	if unit.Origin == "" || unit.APIKey == "" {
		return nil, SiteReconStatusNoKey, "账号没有 base_url/api_key，无法对账"
	}
	det := e.detector.detect(ctx, unit.Origin, unit.ProxyURL)
	switch det.Kind {
	case "official":
		return nil, SiteReconStatusOfficial, "官方上游，没有按 key 查扣费的接口"
	case "unknown":
		return nil, SiteReconStatusUnknown, "上游无法访问或无法识别程序类型"
	case "newapi":
		token := e.tokens[unit.Host]
		if token == "" {
			return nil, SiteReconStatusNeedToken, fmt.Sprintf("NewAPI 上游缺访问令牌（到设置页给 %s 补上）", unit.Host)
		}
		up, err := newSiteReconNewAPIUpstream(ctx, e.http, unit.Origin, unit.APIKey, token, unit.ProxyURL, e.loc, det.Status)
		if err != nil {
			st, text := siteReconClassifyUpstreamError(siteReconRedact(err.Error(), unit.APIKey, token))
			return nil, st, text
		}
		return up, SiteReconStatusOK, ""
	default:
		return &siteReconSub2APIUpstream{h: e.http, origin: unit.Origin, apiKey: unit.APIKey, proxyURL: unit.ProxyURL,
			tz: e.cfg.Timezone, loc: e.loc, now: e.now}, SiteReconStatusOK, ""
	}
}

// buildUnits 按 (上游 host, key) 归并：共用同一把 key 的账号必须合并对账，否则上游成本会重复计算。
func (e *siteReconEngine) buildUnits(ctx context.Context, accounts []siteReconAccount, first, target string) ([]*siteReconUnit, error) {
	type groupKey struct{ host, key string }
	grouped := map[groupKey][]siteReconAccount{}
	var order []groupKey
	for _, a := range accounts {
		_, host := siteReconOrigin(a.BaseURL)
		key := a.APIKey
		if key == "" {
			key = fmt.Sprintf("#%d", a.ID)
		}
		k := groupKey{host, key}
		if _, ok := grouped[k]; !ok {
			order = append(order, k)
		}
		grouped[k] = append(grouped[k], a)
	}
	units := make([]*siteReconUnit, 0, len(order))
	for _, k := range order {
		members := grouped[k]
		ids := make([]int64, len(members))
		for i, m := range members {
			ids[i] = m.ID
		}
		// 主账号要稳定：优先沿用窗口内已经挂着数据的那个，免得同一把 key 的历史在两个账号间跳来跳去
		have, err := e.store.accountsWithRows(ctx, first, target, ids)
		if err != nil {
			return nil, err
		}
		sort.SliceStable(members, func(i, j int) bool { return have[members[i].ID] && !have[members[j].ID] })
		names := make([]string, len(members))
		for i, m := range members {
			ids[i] = m.ID
			names[i] = m.Name
		}
		origin, host := siteReconOrigin(members[0].BaseURL)
		units = append(units, &siteReconUnit{
			IDs: ids, Name: strings.Join(names, " + "), Key: siteReconUnitKey(k.host, k.key),
			Host: host, Origin: origin, APIKey: members[0].APIKey, ProxyURL: members[0].ProxyURL,
		})
	}
	return units, nil
}

// run 对账一次。返回摘要。
func (e *siteReconEngine) run(ctx context.Context, opts siteReconRunOptions) (*siteReconSummary, error) {
	if e.detector == nil {
		e.detector = newSiteReconDetector(e.http)
	}
	if e.openUpstream == nil {
		e.openUpstream = e.defaultOpenUpstream
	}
	target := opts.Target
	if target == "" {
		target = e.now().In(e.loc).AddDate(0, 0, -1).Format("2006-01-02")
	}
	days := opts.Days
	if days <= 0 {
		days = e.cfg.Days
	}
	scan := e.cfg.Scan
	if opts.Scan != nil {
		scan = *opts.Scan
	}
	dayList, err := siteReconDayList(target, days)
	if err != nil {
		return nil, fmt.Errorf("日期不对: %w", err)
	}
	first := dayList[len(dayList)-1]

	listed, err := e.site.ListAPIKeyAccounts(ctx)
	if err != nil {
		return nil, fmt.Errorf("读取本站账号失败: %w", err)
	}
	exclude := map[int64]bool{}
	for _, id := range e.cfg.Exclude {
		exclude[id] = true
	}
	var candidates []siteReconAccount
	for _, a := range listed {
		if exclude[a.ID] || (len(opts.Only) > 0 && !opts.Only[a.ID]) {
			continue
		}
		candidates = append(candidates, a)
	}
	// 只对账窗口内本站有消费的账号：没消费就没收入可对，也不去打扰上游
	used, err := e.site.WindowRequests(ctx, first, target)
	if err != nil {
		return nil, fmt.Errorf("统计本站用量失败: %w", err)
	}
	var accounts []siteReconAccount
	paused := 0
	for _, a := range candidates {
		if used[a.ID] > 0 {
			accounts = append(accounts, a)
			if a.Status != StatusActive {
				paused++
			}
		}
	}
	msg := fmt.Sprintf("本站 apikey 账号 %d 个，%s～%s 有消费的 %d 个", len(candidates), first, target, len(accounts))
	if paused > 0 {
		msg += fmt.Sprintf("（其中 %d 个现已停用/报错，照常对账）", paused)
	}
	e.log(msg + fmt.Sprintf("，其余 %d 个跳过", len(candidates)-len(accounts)))

	units, err := e.buildUnits(ctx, accounts, first, target)
	if err != nil {
		return nil, err
	}

	// 清掉会造成重复计成本的旧行：
	//   - 已并入共用 key 那一行的次要账号；
	//   - 同一把 key 以前挂在别的账号下的行（主账号变了）；
	//   - 全量运行时，本站还在但窗口内零消费（或被排除）的账号。只跑部分账号时不动其他账号。
	// 已从本站删除的账号不在 listed 里，它们的历史原样保留。
	stale := map[int64]bool{}
	for _, u := range units {
		for _, id := range u.IDs[1:] {
			stale[id] = true
		}
	}
	if len(opts.Only) == 0 {
		running := map[int64]bool{}
		for _, a := range accounts {
			running[a.ID] = true
		}
		present, err := e.store.accountsWithRows(ctx, first, target, nil)
		if err != nil {
			return nil, err
		}
		for _, a := range listed {
			if present[a.ID] && !running[a.ID] {
				stale[a.ID] = true
			}
		}
	}
	staleIDs := make([]int64, 0, len(stale))
	for id := range stale {
		staleIDs = append(staleIDs, id)
	}
	if err := e.store.dropRows(ctx, first, target, staleIDs); err != nil {
		return nil, err
	}
	for _, u := range units {
		others, err := e.store.otherAccountsForUnit(ctx, first, target, u.Key, u.IDs[0])
		if err != nil {
			return nil, err
		}
		if err := e.store.dropRows(ctx, first, target, others); err != nil {
			return nil, err
		}
	}

	rc := &siteReconRunCtx{target: target, dayList: dayList, scan: scan, rescan: opts.Rescan}
	problems := map[string][]string{}
	var mu sync.Mutex
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(siteReconAccountWorkers)
	for _, u := range units {
		g.Go(func() error {
			status, uerr := e.reconcileUnit(gctx, rc, u)
			if uerr != nil {
				if gctx.Err() != nil {
					return gctx.Err()
				}
				e.log(fmt.Sprintf("[%d %s] 失败: %v", u.IDs[0], u.Name, uerr))
				errText := "对账过程出错: " + uerr.Error()
				_ = e.store.mergeDaily(gctx, target, u.IDs[0], e.now().In(e.loc), func(d *SiteReconDaily) {
					d.AccountName, d.Upstream, d.UnitKey, d.Error = u.Name, u.Host, u.Key, siteReconStr(errText)
				})
				status = SiteReconStatusError
			}
			if status != SiteReconStatusOK {
				mu.Lock()
				problems[status] = append(problems[status], fmt.Sprintf("%s  ←  #%d %s", u.Host, u.IDs[0], u.Name))
				mu.Unlock()
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	for k := range problems {
		sort.Strings(problems[k])
	}
	return &siteReconSummary{Target: target, FirstDay: first, Days: days, Listed: len(candidates), Units: len(units), Problems: problems}, nil
}

// reconcileUnit 对账一个单元：窗口内每天的两边汇总 + 需要的日子逐条扫描。返回状态。
func (e *siteReconEngine) reconcileUnit(ctx context.Context, rc *siteReconRunCtx, u *siteReconUnit) (string, error) {
	accountID := u.IDs[0]
	log := func(m string) { e.log(fmt.Sprintf("[%d %s] %s", accountID, u.Name, m)) }
	secret := e.tokens[u.Host]

	upstream, status, upErr := e.openUpstream(ctx, u)
	var upDaily map[string]siteReconUpDay
	var upBalance *float64
	if upstream != nil {
		log(upstream.Describe())
		d, bal, err := upstream.Daily(ctx, rc.dayList)
		if err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			status, upErr = siteReconClassifyUpstreamError(siteReconRedact(err.Error(), u.APIKey, secret))
			upstream = nil
		} else {
			upDaily, upBalance = d, bal
		}
	}
	if upErr != "" {
		log(upErr)
	}

	ourDays, err := e.site.DaysStats(ctx, u.IDs, rc.dayList)
	if err != nil {
		return "", fmt.Errorf("统计本站用量失败: %w", err)
	}
	judged := make(map[string]siteReconJudged, len(rc.dayList))
	var kept []string
	for _, day := range rc.dayList {
		old, err := e.store.getDaily(ctx, day, accountID)
		if err != nil {
			return "", err
		}
		var up *siteReconUpDay
		if v, ok := upDaily[day]; ok {
			up = &v
		}
		j := siteReconKeepKnown(old, siteReconJudgeDay(ourDays[day], up, upErr))
		judged[day] = j
		if j.Kept {
			kept = append(kept, day)
		}
		isTarget := day == rc.target
		if err := e.store.mergeDaily(ctx, day, accountID, e.now().In(e.loc), func(d *SiteReconDaily) {
			d.AccountName, d.Upstream, d.UnitKey = u.Name, u.Host, u.Key
			d.OurRequests, d.OurListCost, d.Revenue = j.OurRequests, j.OurListCost, j.Revenue
			d.UpRequests, d.Cost, d.Error = j.UpRequests, j.Cost, j.Error
			// NewAPI 的每日汇总不带原价，原价是逐条扫描时补的；别用空值把以前扫出来的冲掉
			if !(j.Cost != nil && j.UpListCost == nil) {
				d.UpListCost = j.UpListCost
			}
			if isTarget {
				d.UpBalance = upBalance
			}
		}); err != nil {
			return "", err
		}
		if j.Error != nil && upErr == "" {
			log(day + " " + *j.Error)
		}
	}
	if len(kept) > 0 {
		sort.Strings(kept)
		log("上游现在查不到或变少了，保留之前记下的上游数据：" + strings.Join(kept, "、"))
	}
	t := judged[rc.target]
	upReq := "-"
	if t.UpRequests != nil {
		upReq = fmt.Sprintf("%d", *t.UpRequests)
	}
	cost := 0.0
	if t.Cost != nil {
		cost = *t.Cost
	}
	log(fmt.Sprintf("%s 本站 %d 次 收 ¥%.2f | 上游 %s 次 扣 ¥%.2f", rc.target, t.OurRequests, t.Revenue, upReq, cost))

	if !rc.scan {
		return status, nil
	}
	for _, day := range rc.dayList { // 从目标日往前排
		j := judged[day]
		if j.OurRequests == 0 && (j.UpRequests == nil || *j.UpRequests == 0) {
			continue // 两边都没流量，没什么可扫
		}
		need, err := e.needsScan(ctx, day, accountID, j)
		if err != nil {
			return "", err
		}
		scannedBefore := !need
		if j.Kept && scannedBefore {
			continue // 上游数据已经不全，重扫只会把按模型的对比弄坏
		}
		if !(day == rc.target || rc.rescan || !scannedBefore) {
			continue
		}
		if err := e.scanDay(ctx, u, day, j, upstream, log); err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			log(fmt.Sprintf("%s 逐条扫描失败（下次运行会重试）: %s", day, siteReconRedact(err.Error(), u.APIKey, secret)))
			if status == SiteReconStatusOK {
				status = SiteReconStatusUpstreamError
			}
		}
	}
	return status, nil
}

// needsScan 没扫过；或者扫的时候拿不到上游成本、现在拿到了（比如后来补了上游令牌）。
func (e *siteReconEngine) needsScan(ctx context.Context, day string, accountID int64, j siteReconJudged) (bool, error) {
	d, err := e.store.getDaily(ctx, day, accountID)
	if err != nil {
		return false, err
	}
	if d == nil || d.ScannedRows == nil {
		return true, nil
	}
	if j.Cost == nil {
		return false, nil
	}
	return e.store.hasUnallocatedGroups(ctx, day, accountID)
}

// scanDay 逐条扫描一天：异常、按模型对比、按分组分摊成本。先全部算完，再一次性写库。
func (e *siteReconEngine) scanDay(ctx context.Context, u *siteReconUnit, day string, j siteReconJudged, upstream siteReconUpstream, log func(string)) error {
	accountID := u.IDs[0]
	haveCost := j.Cost != nil
	// 成本是沿用旧记录的那种日子：上游的按模型数据已经不全，不拿来对比，成本整体按本站原价占比分摊
	byModel := upstream != nil && haveCost && !j.Kept
	res, err := e.site.ScanDay(ctx, u.IDs, day)
	if err != nil {
		return err
	}
	upModels := map[string]siteReconUpModel{}
	if byModel {
		if upModels, err = upstream.DayModels(ctx, day); err != nil {
			return err
		}
	}
	ourModels := res.Models()
	var total *float64
	if haveCost {
		total = j.Cost
	}
	alloc := siteReconAllocateGroupCost(res.GroupModels, ourModels, upModels, total)

	groups := map[siteReconGroupKey]*siteReconAgg{}
	for k, x := range res.GroupModels {
		gk := siteReconGroupKey{Group: k.Group, Rate: k.Rate}
		g := groups[gk]
		if g == nil {
			g = &siteReconAgg{}
			groups[gk] = g
		}
		g.Requests += x.Requests
		g.ListCost += x.ListCost
		g.Revenue += x.Revenue
	}

	detail := &siteReconDayDetail{Samples: res.Samples, Stats: res.Stats}
	modelNames := map[string]bool{}
	for m := range ourModels {
		modelNames[m] = true
	}
	for m := range upModels {
		modelNames[m] = true
	}
	names := make([]string, 0, len(modelNames))
	for m := range modelNames {
		names = append(names, m)
	}
	sort.Strings(names)
	for _, m := range names {
		row := siteReconModelRow{Model: m}
		if o := ourModels[m]; o != nil {
			row.OurRequests, row.Revenue, row.OurListCost = o.Requests, o.Revenue, o.ListCost
		}
		if byModel {
			up := upModels[m]
			row.UpRequests = siteReconI64(up.Requests)
			row.Cost = siteReconF64(up.Cost)
		}
		detail.Models = append(detail.Models, row)
	}
	for gk, g := range groups {
		row := siteReconGroupRow{GroupName: gk.Group, RateMultiplier: gk.Rate, Requests: g.Requests, ListCost: g.ListCost, Revenue: g.Revenue}
		if v, ok := alloc[gk]; ok {
			row.Cost = siteReconF64(v)
		}
		detail.Groups = append(detail.Groups, row)
	}
	if err := e.store.replaceDayDetail(ctx, day, accountID, detail); err != nil {
		return err
	}

	var anomalies, mismatches int64
	for k, st := range res.Stats {
		if k.Kind == SiteReconKindMismatch {
			mismatches += st.Requests
		} else {
			anomalies += st.Requests
		}
	}
	var upList *float64
	if byModel && j.UpListCost == nil {
		// NewAPI 的每日汇总没有原价，用当天逐条累计的原价补上（供“上游倍率”使用）
		sum := 0.0
		for _, m := range upModels {
			sum += m.ListCost
		}
		upList = &sum
	}
	if err := e.store.mergeDaily(ctx, day, accountID, e.now().In(e.loc), func(d *SiteReconDaily) {
		d.ScannedRows, d.Anomalies, d.Mismatches = siteReconI64(res.N), siteReconI64(anomalies), siteReconI64(mismatches)
		if upList != nil {
			d.UpListCost = upList
		}
	}); err != nil {
		return err
	}
	msg := fmt.Sprintf("%s 扫描 %d 条，异常 %d 条", day, res.N, anomalies)
	if mismatches > 0 {
		msg += fmt.Sprintf("，模型变体 %d 条", mismatches)
	}
	log(msg)
	return nil
}
