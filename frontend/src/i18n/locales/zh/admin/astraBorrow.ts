export default {
  astraBorrow: {
    title: 'Astra 网关借票', description: '借用来源账号的路由，目标账号验证通过后使用。',
    refresh: '刷新状态', guide: '先选一个 Astra 表现正常的来源账号和一个目标账号，使用固定出口，保存启用后自动获取路由并进行两轮验证。来源与目标不能重叠。',
    boundary: '当前支持 HTTP 请求，客户端地址仍为 /v1/responses。目标账号请关闭 Excel 协议；选为目标期间不支持原生双向长连接。探针会消耗上游额度，探针结果不代表模型能力永久恢复。',
    enabled: '启用网关借票', on: '已启用', off: '已关闭', preparing: '正在准备', idle: '按需更新路由',
    search: '搜索账号名称或编号', sources: '来源账号', targets: '目标账号', source: '来源', target: '目标',
    sourceHint: '用于取得路由，请先确认账号的 Astra 表现正常。不会复制来源凭据或门票。',
    targetHint: '使用目标自己的凭据验证和请求。最多选择 20 个；先用一个账号测试。',
    emptyAccounts: '没有匹配的登录授权账号', selected: '已选择',
    followProxy: '跟随来源代理（建议开启）', proxyHint: '开启后，验证和业务请求使用来源取得路由时的代理；来源直连时目标也直连。一次验证及后续使用应保持出口稳定。持续失败可换一个固定出口，重新获取来源路由再验证；当前没有自动换节点。代理地址相同不保证公网 IP 相同。',
    probeStatuses: '两轮响应状态码：首次 {mint} / 续接 {continuation}（200 表示请求被接受，不代表验证通过）',
    probeTickets: '门票长度：首次 {mint} / 续接 {continuation}；0 表示没有收到门票。仅显示长度，不显示票据内容。',
    ticketLengthWarning: '参考项目仅说明在 780 字符门票上验证过这一判据。当前长度不同，不能仅凭换票就认定降智或 IP 有问题。',
    ttl: '路由有效期（秒）', ttlHint: '30–240 秒，默认 230。使用不会延长有效期；上游提前过期则取更短时间。',
    save: '保存配置', saving: '正在保存…', discard: '撤销修改', unsaved: '有未保存修改，验证使用已保存配置',
    runtime: '运行状态 · 当前实例', historyError: '历史记录读取或保存失败，请检查数据库。',
    remaining: '剩余 {seconds} 秒', verify: '重新验证', verifying: '验证中…', emptyRuntime: '尚未选择账号',
    history: '验证历史 · 保留 30 天', time: '时间', result: '结果', emptyHistory: '暂无验证记录', older: '加载更早记录',
    invalidTTL: '有效期必须是 30–240 的整数。', tooMany: '来源和目标各最多选择 20 个账号。', overlap: '来源账号与目标账号不能重叠。', chooseBoth: '启用前请选择来源账号和目标账号。',
    requestError: '请求失败，请重试。', accountsError: '账号列表加载失败，请重新打开页面。', saved: '配置已保存。启用后自动准备，请以运行状态为准。', verified: '本次两轮验证通过。', upstreamError: '上游返回 {status}，本次验证未通过',
    states: { ready: '可用', failed: '失败', expired: '已过期', disabled: '已关闭', not_tested: '未验证', verifying: '验证中' },
    reasons: {
      astra_proxy_unavailable: '账号绑定的代理不可用', astra_business_transport_failed: '业务请求连接失败，未自动重放',
      not_tested: '尚未验证', verifying: '正在验证目标账号', astra_probe_passed: '两轮完整成功，续接未换票', astra_source_ready: '已取得来源路由',
      astra_borrow_busy: '其他验证正在进行，请稍后重试', astra_account_unavailable: '账号不可用、暂停或冷却中', astra_disable_excel_first: '请先关闭所选账号的 Excel 协议', astra_model_unavailable: '账号未允许 Astra 或映射到了其他模型', astra_auth_failed: '无法获取账号登录凭据',
      astra_settings_unavailable: '无法读取配置', astra_settings_invalid: '已保存配置无效', astra_save_failed: '配置保存失败', astra_configuration_changed: '配置、来源身份或路由已变化，请重新验证', astra_account_changed: '账号配置已变化，请重新验证',
      astra_probe_cooldown: '目标验证失败，正在冷却', astra_source_cooldown: '来源获取失败，正在冷却', astra_no_source_route: '没有可用来源路由', astra_source_cookie_missing: '来源未返回有效路由标记',
      astra_probe_timeout: '探针超时或已取消', astra_account_busy: '账号并发已满', astra_rate_limited: '账号请求频率受限', astra_probe_request_failed: '探针请求构造失败', astra_network_error: '上游连接失败',
      astra_stream_incomplete: '响应未完整结束或没有正文', astra_stream_invalid: '响应格式无效', astra_stream_failed: '上游流内失败', astra_model_mismatch: '响应模型与 Astra 不符', astra_ticket_invalid: '上游门票格式无效',
      astra_route_changed: '探针期间上游更换了路由', astra_ticket_missing: '首轮没有返回门票', astra_route_expired: '路由已过期', astra_ticket_changed: '续接返回了新门票，未通过验证', astra_business_route_invalidated: '业务请求失败或路由改变，已撤销可用状态',
      astra_request_not_inspectable: '无法确认请求模型', astra_http_responses_only: '借票仅支持 HTTP Responses 请求'
    }
  }
}
