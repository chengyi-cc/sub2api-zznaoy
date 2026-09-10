# OpenAI 请求头与客户端形态审计（2026-09-10）

后续调整（同日）：按用户要求，代码默认身份已改为 `codex-tui/0.153.4 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.153.4)`，来源标记保持 `codex-tui`，默认版本为 `0.153.4`；尾部标记与首段版本由同一版本值生成。只修改默认声明与界面示例，不改其他功能或配置机制，未部署。以下内容保留调整前的审计记录，旧版本和 Ubuntu 示例不是调整后的默认值。

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

### 1. 第三方请求未获得完整线程身份

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
