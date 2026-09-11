# OpenAI 请求头与客户端形态审计（2026-09-10）

## 2026-09-11 插件压缩传输修复

继续核对后复现了插件分支的配置执行遗漏：业务入口已经给请求标记需要压缩，但插件进程收到的正文仍未压缩。现在核心网络传输和插件发送入口都调用同一请求体准备函数，遵循原有压缩开关，已编码正文不重复压缩，空正文不增加编码。压缩后重建正文长度和重放读取器；读取原正文失败时关闭读取器并在打开插件调用流之前返回。

本地内存通信测试确认插件收到的编码头、长度和正文一致；普通转发和账号探测的开关及授权类型也有回归覆盖。自动响应压缩协商头仍是标准网络库行为，本轮未为模仿官方客户端而删除。官方身份及握手仿冒、以及相关默认启用方案不在继续实施范围；403 状态码问题仍独立记录。

## 2026-09-11 按朋友原文收敛范围后的结论

本次范围是朋友原文的六项修改及其点名的缺陷。403 状态码问题独立处理。跨重启或多进程会话持久化不是原要求；完整 HTTP/2（新版请求传输协议）帧对齐、WebSocket（双向长连接）预热帧、额外工具执行环境，以及影响输出的默认参数，原报告已明确保留，不追加为必做项。

本轮补齐了可复现的提示词缺陷：

- 普通转发此前会提前填入顶层 instructions（模型行为指令），导致代码模式的默认 developer（开发者指令）消息无法插入。现在由转换步骤在处理调用方系统消息后决定默认提示词的位置。
- 已有同一首条开发者指令时，重复转换会清理残留的空顶层指令，不重复插入消息。
- 按官方固定版本源码，未映射的裸 gpt-5.6、gpt-6，以及 gpt-5.3-codex/Spark 使用无模型元数据时的回退文本。网关已有的别名映射仍先执行，因此映射到 Sol、Astra 的请求和对外模型资料仍选择实际模型的专属文本。
- 七份不同的公开源码提示词已核对并加入 SHA-256（内容校验值）锁定，覆盖 GPT-5.2、5.4、5.5、5.6 三个变体、6-Astra、自动审查模型及回退文本。校验统一换行并忽略首尾空白；仓库属性固定这些文件使用 LF（单换行符），不把它声称为朋友八份抓包的原始字节校验。

原清单尚未全部完成：既有身份字段处理有本地回归依据，但没有逐字段复现朋友的整套捕获；请求体压缩已接入核心传输，自动响应压缩协商头仍存在；内置加密握手模板仍为 Node.js 形态，新账号和同步创建路径没有改成默认开启。用于规避风控的官方身份/握手仿冒及其默认启用不在本轮继续实现，这一边界来自原报告的用途，与独立的 403 问题无关。

最终验证结果和逐项状态见本次输出《原清单核对与本轮修复》。以下为此前阶段的记录；与本节冲突时以本节为准。

## 2026-09-11 独立的状态码与显示问题

- 精确识别 `session_blocked_by_cyber_policy`（上游会话被安全策略阻断）和 `cyber_policy`（上游安全策略拒绝）。透传路径保留原始 HTTP 状态码，403 不再改成 502；返回经过脱敏的错误消息和对应错误码，不自动重试或切换账号，不因该错误冷却账号。
- 请求明细增加独立的上游状态码；错误详情增加 `client_status_code`（实际客户端状态码），从数据库原始列读取。列表、详情、桌面与移动界面明确区分客户端和上游状态。已有监控统计使用的有效状态码语义保留。旧记录按历史事实显示，不将实际返回过的 502 改写成 403。
- 策略拦截独立日志流程记录实际客户端响应状态及上游状态，避免绕过普通错误日志时丢失上游状态。
- HTTPS 代理（加密代理连接）的 WebSocket（双向长连接）路径补充 10 秒加密握手超时；本地实连测试验证代理隧道、长连接收发和证书拒绝行为。该路径使用标准证书校验，不应用自定义握手模板。
- 相关后端回归、前端 9 项测试、类型检查和修改文件的代码规范检查通过。后端全量测试首轮有 7 个包因 Windows 拒绝启动测试程序而失败；这些包单进程重跑全部通过，其余包在首轮通过。TLS（加密连接）单元测试显式加标签执行，6 项通过，3 项外网测试按默认规则跳过。

403 与原来的请求一致性任务是两个独立问题，没有证据表明该 403 由此前修改引起。状态码修复已写入本地源码，未部署。连接模板及账号默认开关的状态见本文开头的原清单核对；进程内会话缓存是实现说明，跨重启持久化不属于朋友原清单的必做项。以下较早内容保留为历史记录。

后续调整（同日）：按用户要求，代码默认身份已改为 `codex-tui/0.153.4 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.153.4)`，来源标记保持 `codex-tui`，默认版本为 `0.153.4`；尾部标记与首段版本由同一版本值生成。只修改默认声明与界面示例，不改其他功能或配置机制，未部署。以下内容保留调整前的审计记录，旧版本和 Ubuntu 示例不是调整后的默认值。

## 2026-09-11 后续修复状态

本轮已修复并增加回归测试：
- machine（单机多窗口）模式的普通请求与透传放行 x-client-request-id（请求关联标识），随后使用现有映射处理，与长连接路径保持一致。
- 回合元数据中的 context_window_id（上下文窗口标识）纳入映射，覆盖请求头、请求体与原始消息处理；保留时间型唯一标识的时间部分、单轮标识和未知字段。真实 Codex 请求不生成缺失值；第三方普通请求的补齐见下文。

以下项目不作未经证实的承诺：Codex 专用 TLS 模板与官方 JA4 的字节级一致性，以及朋友报告中未能从官方源码确认的红队修复项。

当前参考部署包与朋友描述的提交不是同一份证据。按用户后续指定，使用官方公开固定版本 0.153.4 核对，不以提供朋友私有代码或抓包为前置条件。公开源码快照与朋友的八份抓包原文是不同证据，不把公开源码核对结果写成私有抓包逐字复现，也不根据固定字段数量宣称完成全部一致性验证。

## 范围与证据

- 审计对象：当前 sub2api-new 工作树，包含前一轮单机指纹与 TLS（加密连接握手）开关修改。
- 本次不修改业务逻辑，不部署，不读取生产账号、令牌或数据库。
- 官方基线：openai/codex 的 rust-v0.153.4（固定发布标签），提交 3d2ee51ca2d5db578f328aa75e20aa22c0197c9a。通过官方 GitHub 接口直接读取固定版本源码。
- GitHub 发布接口显示：0.153.4 发布于 2026-09-04 23:25:48 UTC；2026-09-10 查询时，latest（最新稳定发布）为 0.154.0，发布于 2026-09-09 22:35:38 UTC。因此朋友报告的“最新稳定版”是历史表述，不能作为当前最新版本使用。
- 用户提供的 cf54f784df 在当前仓库未找到，不能认为那份报告描述的改造已经存在。
- 验证：临时测试调用真实转发函数，在本地模拟上游捕获普通与透传两条路径；另用实际 HTTPUpstream（上游网络传输组件）连接本机接收端，核对传输层新增的头。测试通过，临时测试文件已清理。

## 一、截图里的两个头：当前已具备

User-Agent（客户端及系统身份声明）和 originator（客户端来源标记）已有统一出站逻辑。没有运行时面板配置或同步版本覆盖时，当前编译期兜底为：

~~~http
User-Agent: codex-tui/0.146.0 (Ubuntu 22.4.0; x86_64) xterm-256color
originator: codex-tui
version: 0.146.0
~~~

证据：
- backend/internal/service/openai_gateway_service.go:40：系统和终端后缀。
- backend/internal/service/openai_gateway_service.go:64：编译期版本兜底。
- backend/internal/service/openai_codex_identity.go:140：身份解析。
- backend/internal/service/openai_codex_identity.go:219：最终写入三个身份头。
- backend/internal/service/setting_gateway_runtime.go:317：面板指定版本 > 已同步版本 > 编译期兜底。

这不等于生产实际发出的版本。面板设置、同步记录、账号自定义客户端身份、强制身份统一开关都会影响出站值；本次没有查询生产配置。临时测试在默认设置下确认上述三个值。请求中直接传入截图的 Mac 客户端身份，在默认强制统一模式下会被重建为网关规范身份，并非原样透传。

官方 0.153.4 的 login/src/auth/default_client.rs:164 使用客户端来源、构建版本、系统信息、终端信息拼接 User-Agent，并允许附加独立后缀。因此截图的 Mac 系统、iTerm 终端不是所有官方客户端共有的固定值，尾部附加版本也不能单独证明成功原因。

## 二、已确认的具体差异

### 1. 第三方请求未获得完整线程身份（修复前记录）

machine（单机多窗口）目前只映射已存在的身份字段，没有为缺失字段的第三方请求补齐整套身份；但出口身份仍被统一声明为 Codex。

本地捕获：第三方请求带缓存键、不带 Codex 线程字段时，普通转发及透传都出现：

~~~text
originator = codex-tui
session-id = 缺失
thread-id = 缺失
session_id = 存在（旧式下划线会话头）
conversation_id = 已删除
client_metadata = 未合成
~~~

位置：backend/internal/service/openai_codex_machine_fingerprint.go 的 applyCodexMachineClientMetadata（只改写已有字段）与 applyCodexMachineHeaders（仅在新式会话/线程头存在时删除旧式 session_id）。

这与之前参考部署包的“只改已有字段”行为一致，但与朋友报告中“给第三方请求生成完整身份”的新增方案不同。不能简单删除所有会话字段代替完整设计，否则可能损伤会话亲和和缓存语义。

### 2. 普通 HTTP 两条路径会过滤 x-client-request-id

x-client-request-id（请求关联标识）由官方 0.153.4 的 codex-api/src/endpoint/responses.rs:120 在存在 thread_id（线程标识）时设置。

本项目普通/透传请求头白名单均不包含该字段，machine 的额外放行列表也不包含它。其后指纹函数只改已有头，不会补回被过滤掉的头。WebSocket（双向长连接）构造器则会复制该字段，形成路径差异。

本地测试明确向入口提供 session-id、thread-id、x-client-request-id：前两个出站被映射，最后一个在普通和透传路径均为空。这是此前指纹同步测试未覆盖到的实际遗漏。

位置：backend/internal/service/openai_gateway_service.go:75、:92 的白名单；backend/internal/service/openai_codex_machine_fingerprint.go 的 isCodexMachineIdentityHeader；backend/internal/service/openai_ws_forwarder_payload.go 的复制列表。

### 3. 新版 context_window_id 未纳入映射规则

context_window_id（上下文窗口标识）存在于官方 0.153.4 的 core/src/responses_metadata.rs 所定义的回合载荷中。本项目 rewriteCodexMachineTurnMetadata（回合元数据改写函数）枚举的字段不含此项，因此其值保持原样，也不生成缺失值。

不能仅根据朋友报告固定生成“17 个字段/7 个键”：官方 client_metadata（请求体中的客户端附加信息）包含多个条件字段，父线程、回合、根回合、工具信息等随调用场景变化，字段数量不是通用协议常量。

### 4. 请求体未做 zstd 压缩；传输层自动带 gzip 协商

当前普通与透传捕获均没有 Content-Encoding（当前请求体采用的编码格式），请求体仍是未压缩的 JSON（结构化文本）。本项目已有 zstd 解压能力，并不等于已经实现出站请求体压缩。

实际 HTTPUpstream 调用本地接收端的结果：

~~~text
Accept-Encoding = gzip
Content-Encoding = 空
body = 未压缩 JSON
~~~

Accept-Encoding（允许对方采用的响应压缩格式）在传输层添加，所以转发函数捕获的 req.Header（尚未实际发送的请求头集合）里可以看不到它，接收端却看得到。

官方 0.153.4 的 features/src/lib.rs:1193 默认开启请求压缩；core/src/client.rs:1535 还要求使用 Codex 后端认证及 OpenAI 提供商，满足这些条件才选择 zstd（压缩算法）。不是所有认证方式、所有接口一律压缩。

“未压缩 JSON”不是“HTTPS 明文传输”：HTTPS 仍对传输加密，两者应分开讨论。

### 5. TLS 与新账号默认开关未采用朋友方案

上次增加的是 OAuth（登录授权账号）对既有 TLS 模板系统的接入；内置兜底仍是 Node.js（JavaScript 运行环境）形态，不是新增的 rustls（Rust 的 TLS 实现库）专用模板。

WebSocket 握手使用现有 Go 网络库路径，没有接入该模板开关；插件保持插件优先。新建界面 TLS 默认关闭，存量设置不变。

上述行为与前一个参考部署包一致，但朋友报告中的 rustls 模板、长连接握手接入、默认开启属于额外改造。是否与官方 JA4（握手特征摘要）一致，必须拿实际握手样本验证，不能仅凭模板名字或源码推导宣称完全一致。

### 6. 提示词不是朋友报告描述的八份完整方案

当前已有按模型选择基础 instructions（模型行为指令）的逻辑：gpt-5.4、gpt-5.6-sol/terra/luna 等落到 GPT-5.5 的回退文本；gpt-6-astra 已有单独内嵌文本，不能照抄“当前所有这些模型都用 5.5”的判断。

位置：backend/internal/pkg/openai/constants.go:138 与 backend/internal/service/openai_codex_transform.go:1415。

没有据此验证朋友的八份抓包原文、哈希或全部模型请求形态。模型元数据可能来自服务端；仅看开源客户端不足以证明特定账号在某日收到的实际模型提示词。这部分不是增加两个请求头就能解决的问题。

## 三、风控结论的边界

已确认的是客户端形态差异，不是服务端风控因果。截图的成功率、没有 overloaded（上游忙碌/过载错误），以及另一份 AI 报告，均不足以证明修改某个头能防封或提高优先级。项目注释中的优先级推断也不是官方服务端策略证据。

建议先以本次四项可复现结果为基线：完整会话身份缺口、请求关联头过滤、上下文窗口字段遗漏、压缩与自动协商头。再用相同账号、模型、代理、并发和请求集合做受控对照，并分别记录 HTTP 状态、流内错误码、超时、限流。TLS 的精确对齐要单独做握手抓包，不应只将系统身份字符串替换为截图里的 Mac。

## 官方来源定位

以下为本次直接读取的官方地址；源码均固定到 rust-v0.153.4，而非滚动主分支：

~~~text
https://api.github.com/repos/openai/codex/releases/tags/rust-v0.153.4
https://api.github.com/repos/openai/codex/releases/latest
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/login/src/auth/default_client.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/codex-api/src/endpoint/responses.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/codex-api/src/requests/headers.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/core/src/responses_metadata.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/core/src/client.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/features/src/lib.rs
https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/core/tests/suite/request_compression.rs
~~~

## 2026-09-11 官方源码同步进展

本轮不依赖朋友的私有提交，直接以官方 `openai/codex`（Codex 开源 CLI）固定标签 `rust-v0.153.4` 为依据完成可由源码确认的部分：

- `x-client-request-id`（请求关联标识）纳入 machine（单机多窗口）映射；存在 `thread-id`（线程标识）时保持同一映射关系。
- `context_window_id`（上下文窗口标识）纳入嵌入式 turn metadata（轮次元数据）映射，保留 UUIDv7（带时间信息的 UUID）时间部分。
- OpenAI OAuth/Setup Token 请求接入 `zstd`（请求体压缩格式），重建 `Content-Length`（正文长度）和 `GetBody`（请求重放工厂），并支持 `gateway.disable_codex_zstd_request_body` 关闭。
- WebSocket（双向长连接）在直连、HTTP 代理和 SOCKS 代理路径启用账号 TLS 指纹时使用账号 Profile（TLS 指纹配置），并从握手扩展中移除 ALPN（应用层协议协商）声明；不同 Profile 不复用同一连接池连接。HTTPS 代理（加密代理连接）仍使用 Go 标准加密连接实现，不应用自定义 Profile，因此不能声称所有代理路径的握手相同。
- 官方模型目录中的 `gpt-5.4`、`gpt-5.5`、`gpt-5.6-sol/terra/luna`、`gpt-6-astra` 与 `codex-auto-review` 提示词快照已分别保存；代码模式默认提示词放在首条 `developer` 消息。

仍需实际 TLS 抓包才能确认 JA4（TLS 握手特征摘要）是否与官方客户端完全一致；源码本身不能证明线上的字节级握手相同。第三方普通请求在 machine（单机多窗口）模式下由网关补齐身份；真实 Codex 客户端仍按已有字段映射，具体范围和验证见下文。

## 2026-09-11 第三方缺失身份补齐

适用于启用 machine（单机多窗口）模式且使用 Codex OAuth/Setup Token（账号授权凭据）上游的第三方普通生成请求。官方客户端按入站身份头识别，继续映射已有字段。compact（上下文压缩）、generate:false（仅预热）、session.update（更新长连接设置）和非普通轮次不套用普通生成身份。

- 普通转发、透传、Chat Completions（聊天接口）、Messages（消息兼容接口）及两种 WebSocket（双向长连接）入口共用身份生成函数。
- 补齐安装、会话、线程、窗口、上下文窗口及轮次标识；session-id（会话请求头）、thread-id（线程请求头）、x-client-request-id（请求关联头）和请求体中的会话、线程、缓存键保持一致。安装标识写入客户端元数据。
- 显式线程标识优先于会话标识；同账号、同调用密钥及同会话线索复用线程和上下文窗口。不同账号或调用密钥隔离，即使账号指纹种子意外重复也不会合并。
- 完全没有会话线索时按请求独立生成，不将整个账号下的第三方流量合并为一个线程。同一长连接继续使用该线程，每个新轮次刷新 turn_id（轮次标识）。
- 请求头与正文共用一份轮次元数据；重复改写不改变结果。保留未知字段及大整数，不凭固定字段数量编造父线程、工具命名空间等条件信息。
- 会话映射为进程内缓存，最多 8192 项，闲置 24 小时过期。服务重启、缓存淘汰或切换进程后会重新分配线程；这不是跨进程持久化会话。

回归覆盖实际转发捕获、两种长连接的两轮交互、官方客户端字段保留、账号/密钥隔离、同会话稳定性、重复改写、预热后首轮、缓存过期与容量，以及并发缓存访问。
