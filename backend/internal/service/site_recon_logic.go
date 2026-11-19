package service

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// 站点对账（fork 自有模块，迁移自独立部署的「站点对账」Python 服务）。
//
// 口径（与原服务一致）：
//   - 收入 = 本站该账号用户真实被扣的金额（usage_logs.actual_cost，billing_type=0）
//   - 成本 = 上游对这把 key 的真实扣费
//   - 请求差 = 上游请求数 - 本站请求数；>0 表示上游扣了钱、本站没记账
//   - 拿不到上游成本的收入单独列出，不计入毛利
//
// 本文件只放纯逻辑（无 IO），便于单测：每日判定、保留已知成本、分组成本分摊、定时判定。

// 异常种类（入库用稳定代码，展示文案在前端 / 推送里映射）。
const (
	SiteReconKindZeroPaid = "zero_paid" // 有原价但扣费为 0（漏扣）
	SiteReconKindUnpriced = "unpriced"  // 有 token 但原价为 0（模型未定价）
	SiteReconKindFree     = "free"      // 倍率为 0（免费）
	SiteReconKindMismatch = "mismatch"  // 上游返回模型与请求不一致（低优先级）
)

// 每个账号每天每类异常最多留多少条样本（汇总数字不受影响）。
const siteReconSampleCap = 200

// 对账单元状态（problems 的键）。
const (
	SiteReconStatusOK            = "ok"
	SiteReconStatusNeedToken     = "need_token"
	SiteReconStatusDeadKey       = "dead_key"
	SiteReconStatusOfficial      = "official"
	SiteReconStatusUnknown       = "unknown"
	SiteReconStatusUpstreamError = "upstream_error"
	SiteReconStatusNoKey         = "no_key"
	SiteReconStatusError         = "error"
)

var siteReconStatusText = map[string]string{
	SiteReconStatusNeedToken:     "NewAPI 上游缺访问令牌（到设置页补上）",
	SiteReconStatusDeadKey:       "上游 key 已失效（分组删除/key 无效）",
	SiteReconStatusOfficial:      "官方上游，无法按 key 查扣费",
	SiteReconStatusUnknown:       "上游无法访问或无法识别",
	SiteReconStatusUpstreamError: "上游取数出错",
	SiteReconStatusNoKey:         "账号没有 base_url/api_key",
	SiteReconStatusError:         "对账过程出错",
}

// SiteReconDaily 是 site_recon_daily 的一行：某天的一个对账单元（一把上游 key；
// 共用同一把 key 的多个本站账号合并成一行，挂在主账号 account_id 下）。
type SiteReconDaily struct {
	Day         string   `json:"day"`
	AccountID   int64    `json:"account_id"`
	AccountName string   `json:"account_name"`
	Upstream    string   `json:"upstream"`
	UnitKey     string   `json:"-"`
	OurRequests int64    `json:"our_requests"`
	UpRequests  *int64   `json:"up_requests"`
	OurListCost float64  `json:"our_list_cost"`
	UpListCost  *float64 `json:"up_list_cost"`
	Revenue     float64  `json:"revenue"`
	Cost        *float64 `json:"cost"`
	UpBalance   *float64 `json:"up_balance"`
	ScannedRows *int64   `json:"scanned_rows"`
	Anomalies   *int64   `json:"anomalies"`
	Mismatches  *int64   `json:"mismatches"`
	Error       *string  `json:"error"`
	UpdatedAt   string   `json:"updated_at"`
}

// siteReconDayStats 本站某天汇总。
type siteReconDayStats struct {
	Requests int64
	ListCost float64
	Revenue  float64
}

// siteReconUpDay 上游某天汇总；ListCost 为 nil 表示上游不提供原价（NewAPI 每日汇总）。
type siteReconUpDay struct {
	Requests int64
	ListCost *float64
	Cost     float64
}

// siteReconUpModel 上游按模型汇总。
type siteReconUpModel struct {
	Requests int64
	ListCost float64
	Cost     float64
}

// siteReconAgg 本站按模型 / 分组的汇总。
type siteReconAgg struct {
	Requests int64
	ListCost float64
	Revenue  float64
}

type siteReconGroupModelKey struct {
	Group string
	Rate  float64
	Model string
}

type siteReconGroupKey struct {
	Group string
	Rate  float64
}

type siteReconStatKey struct {
	Kind   string
	Model  string
	Detail string
}

type siteReconStat struct {
	Requests   int64
	ListCost   float64
	ActualCost float64
}

type siteReconSample struct {
	Kind       string
	LogID      int64
	CreatedAt  string
	UserID     int64
	Model      string
	TotalCost  float64
	ActualCost float64
	Tokens     int64
}

// siteReconScan 一个对账单元某天的逐条扫描结果（在数据库里聚合完成）。
type siteReconScan struct {
	N           int64
	Samples     []siteReconSample
	Stats       map[siteReconStatKey]*siteReconStat
	GroupModels map[siteReconGroupModelKey]*siteReconAgg
}

// Models 按上游模型汇总（由分组×模型的汇总折叠得到）。
func (s *siteReconScan) Models() map[string]*siteReconAgg {
	out := make(map[string]*siteReconAgg)
	for k, v := range s.GroupModels {
		m := out[k.Model]
		if m == nil {
			m = &siteReconAgg{}
			out[k.Model] = m
		}
		m.Requests += v.Requests
		m.ListCost += v.ListCost
		m.Revenue += v.Revenue
	}
	return out
}

// siteReconJudged 某天两边数据合成后要入库的字段。
type siteReconJudged struct {
	OurRequests int64
	OurListCost float64
	Revenue     float64
	UpRequests  *int64
	UpListCost  *float64
	Cost        *float64
	Error       *string
	Kept        bool
}

func siteReconF64(v float64) *float64 { return &v }
func siteReconI64(v int64) *int64     { return &v }
func siteReconStr(v string) *string   { return &v }

// siteReconJudgeDay 把某天两边的数合成要入库的字段，并判定这一天算不算“对上了”。
//
// 成本为 nil = 不知道（不参与毛利）；为 0 = 确认上游没扣钱。两者不能混：
//   - 上游取数失败 → 成本未知，原因记在 error；
//   - 上游说这把 key 当天没用量，本站却收了钱 → 成本未知（多半是账号换过 key）；
//   - 上游说没用量，本站也没收入 → 成本 0，算已对账。
func siteReconJudgeDay(ours siteReconDayStats, up *siteReconUpDay, upErr string) siteReconJudged {
	out := siteReconJudged{OurRequests: ours.Requests, OurListCost: ours.ListCost, Revenue: ours.Revenue}
	if upErr != "" {
		out.Error = siteReconStr(upErr)
		return out
	}
	if up == nil {
		out.Error = siteReconStr("超出上游可查询的天数范围")
		return out
	}
	out.UpRequests = siteReconI64(up.Requests)
	if up.ListCost != nil {
		out.UpListCost = siteReconF64(*up.ListCost)
	}
	out.Cost = siteReconF64(up.Cost)
	if up.Requests == 0 && ours.Revenue > 0.005 {
		out.Cost = nil
		out.UpListCost = nil
		out.Error = siteReconStr(fmt.Sprintf("本站有 %s 次请求、收入 ¥%s，但上游这把 key 当天没有任何记录（账号多半换过 key，旧 key 的扣费查不到）",
			siteReconFormatInt(ours.Requests), siteReconFormatMoney(ours.Revenue)))
	}
	return out
}

// siteReconKeepKnown 已经记下的上游数据，不让更差的新结果覆盖。
//
// 某一天过去之后，上游对这把 key 的请求数只会不变或（补记后）变多。如果这次拿到的更少，或者干脆
// 拿不到（上游清理了旧数据、key 失效或被换掉、上游临时故障），之前记下的才是对的，原样保留。
func siteReconKeepKnown(old *SiteReconDaily, next siteReconJudged) siteReconJudged {
	if old == nil || old.Cost == nil {
		return next
	}
	var oldN, newN int64
	if old.UpRequests != nil {
		oldN = *old.UpRequests
	}
	if next.UpRequests != nil {
		newN = *next.UpRequests
	}
	lost := next.Cost == nil && (oldN > 0 || next.Revenue <= 0.005)
	if !(lost || newN < oldN) {
		return next
	}
	next.UpRequests = old.UpRequests
	next.UpListCost = old.UpListCost
	next.Cost = old.Cost
	next.Error = nil
	next.Kept = true
	return next
}

// siteReconAllocateGroupCost 把账号的上游成本分摊到 (分组, 扣费倍率)。
//
//  1. 先按模型：某模型的上游成本，按各分组在该模型上的本站原价占比分下去；
//  2. 剩余（上游有但本站名字对不上的模型、日界差异等）按整体本站原价占比补齐，保证合计 = 账号总成本。
func siteReconAllocateGroupCost(
	groupModels map[siteReconGroupModelKey]*siteReconAgg,
	ourModels map[string]*siteReconAgg,
	upModels map[string]siteReconUpModel,
	total *float64,
) map[siteReconGroupKey]float64 {
	out := make(map[siteReconGroupKey]float64)
	if total == nil {
		return out
	}
	keys := sortedGroupModelKeys(groupModels)
	allocated := 0.0
	for _, k := range keys {
		x := groupModels[k]
		gk := siteReconGroupKey{Group: k.Group, Rate: k.Rate}
		out[gk] += 0
		up, hasUp := upModels[k.Model]
		var base float64
		if m := ourModels[k.Model]; m != nil {
			base = m.ListCost
		}
		if hasUp && base != 0 {
			c := up.Cost * x.ListCost / base
			out[gk] += c
			allocated += c
		}
	}
	allList := 0.0
	for _, x := range groupModels {
		allList += x.ListCost
	}
	if allList != 0 {
		rest := *total - allocated
		for _, k := range keys {
			x := groupModels[k]
			out[siteReconGroupKey{Group: k.Group, Rate: k.Rate}] += rest * x.ListCost / allList
		}
	}
	return out
}

func sortedGroupModelKeys(m map[siteReconGroupModelKey]*siteReconAgg) []siteReconGroupModelKey {
	keys := make([]siteReconGroupModelKey, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].Group != keys[j].Group {
			return keys[i].Group < keys[j].Group
		}
		if keys[i].Rate != keys[j].Rate {
			return keys[i].Rate < keys[j].Rate
		}
		return keys[i].Model < keys[j].Model
	})
	return keys
}

// siteReconUnitKey 对账单元标识：上游 host + key 的摘要（不落明文 key）。
func siteReconUnitKey(host, apiKey string) string {
	sum := sha256.Sum256([]byte(host + "|" + apiKey))
	return hex.EncodeToString(sum[:])[:16]
}

// siteReconDayList 从目标日往前排的 n 天。
func siteReconDayList(target string, n int) ([]string, error) {
	end, err := time.Parse("2006-01-02", target)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, end.AddDate(0, 0, -i).Format("2006-01-02"))
	}
	return out, nil
}

// siteReconShiftDay 日期加减天数（非法日期原样返回）。
func siteReconShiftDay(day string, n int) string {
	t, err := time.Parse("2006-01-02", day)
	if err != nil {
		return day
	}
	return t.AddDate(0, 0, n).Format("2006-01-02")
}

// siteReconDayWindow 某天在 loc 下的 [起, 止) 时间。
func siteReconDayWindow(day string, loc *time.Location) (time.Time, time.Time, error) {
	t, err := time.ParseInLocation("2006-01-02", day, loc)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	return t, t.AddDate(0, 0, 1), nil
}

// siteReconLoadLocation 时区名解析失败时，常见中国时区退回固定 UTC+8。
func siteReconLoadLocation(name string) (*time.Location, error) {
	loc, err := time.LoadLocation(name)
	if err == nil {
		return loc, nil
	}
	switch name {
	case "Asia/Shanghai", "Asia/Chongqing", "Asia/Hong_Kong", "Asia/Taipei", "PRC":
		return time.FixedZone(name, 8*3600), nil
	}
	return nil, err
}

// ---------------------------------------------------------------- 定时

// SiteReconScheduleState 定时对账状态（存 settings）。
type SiteReconScheduleState struct {
	LastDay string `json:"last_day,omitempty"`
	Retries int    `json:"retries,omitempty"`
	RetryAt string `json:"retry_at,omitempty"`
}

const (
	siteReconMaxRetries   = 3
	siteReconRetryMinutes = 30
)

func siteReconParseHHMM(v string) (int, int, bool) {
	parts := strings.Split(strings.TrimSpace(v), ":")
	if len(parts) != 2 || len(parts[1]) != 2 || len(parts[0]) == 0 || len(parts[0]) > 2 {
		return 0, 0, false
	}
	h, err1 := strconv.Atoi(parts[0])
	m, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil || h < 0 || h > 23 || m < 0 || m > 59 {
		return 0, 0, false
	}
	return h, m, true
}

// siteReconDue 现在该不该跑：返回 "schedule" / "retry" / ""。
// state.LastDay = 最近一次定时对账在哪个本地日期启动（每天只启动一次）；预定时刻没开机，当天晚些照样补跑。
func siteReconDue(now time.Time, enabled bool, hhmm string, state SiteReconScheduleState) string {
	if !enabled {
		return ""
	}
	h, m, ok := siteReconParseHHMM(hhmm)
	if !ok {
		return ""
	}
	today := now.Format("2006-01-02")
	if state.LastDay != today {
		if now.Hour() > h || (now.Hour() == h && now.Minute() >= m) {
			return "schedule"
		}
		return ""
	}
	if state.RetryAt != "" {
		if at, err := time.Parse(time.RFC3339, state.RetryAt); err == nil && !now.Before(at) {
			return "retry"
		}
	}
	return ""
}

// siteReconNextRun 下次运行的展示文字；定时关闭返回 ""。
func siteReconNextRun(now time.Time, enabled bool, hhmm string, state SiteReconScheduleState) string {
	if !enabled {
		return ""
	}
	if state.LastDay == now.Format("2006-01-02") {
		if state.RetryAt != "" {
			if at, err := time.Parse(time.RFC3339, state.RetryAt); err == nil {
				return at.In(now.Location()).Format("15:04") + "（上次失败，自动重试）"
			}
		}
		return now.AddDate(0, 0, 1).Format("2006-01-02") + " " + hhmm
	}
	if siteReconDue(now, enabled, hhmm, state) != "" {
		return "马上"
	}
	return now.Format("2006-01-02") + " " + hhmm
}

// siteReconAfterScheduledRun 定时对账跑完后更新状态：失败则安排重试，最多 siteReconMaxRetries 次。
func siteReconAfterScheduledRun(now time.Time, state SiteReconScheduleState, ok bool) SiteReconScheduleState {
	if ok {
		state.Retries = 0
		state.RetryAt = ""
		return state
	}
	state.Retries++
	if state.Retries <= siteReconMaxRetries {
		state.RetryAt = now.Add(siteReconRetryMinutes * time.Minute).Format(time.RFC3339)
	} else {
		state.RetryAt = ""
	}
	return state
}

// ---------------------------------------------------------------- 格式化（日志 / 推送）

func siteReconFormatInt(v int64) string {
	s := strconv.FormatInt(v, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

func siteReconFormatMoney(v float64) string {
	v = math.Round(v*100) / 100
	neg := v < 0
	if neg {
		v = -v
	}
	whole := int64(v)
	frac := int64(math.Round((v - float64(whole)) * 100))
	if frac == 100 {
		whole++
		frac = 0
	}
	s := siteReconFormatInt(whole) + fmt.Sprintf(".%02d", frac)
	if neg {
		return "-" + s
	}
	return s
}
