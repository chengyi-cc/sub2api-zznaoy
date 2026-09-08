# v0.2.3 上游合并复核记录

复核日期：2026-09-08。本记录针对本地代码和测试，不代表线上验收或镜像发布。

## 复核基线与结论

- 合并前本地版本：`7296616f9`；本次合入的上游版本：`772a0382f`（v0.2.3）。这里的版本是本次复核对象，不声称它仍是官方最新版本。
- 合并提交：`84a0139c6`；随后补齐翻译的提交：`ba329f04c`。本次修复基于后者，未改写已推送历史。
- 对照合并前后的自有差异，132 个文件中 126 个文件的新增/删除行保持一致，其余差异涉及格式、上游收敛和翻译；未发现自有功能被整段覆盖的证据。这不能替代线上业务验收。
- 自有发票、异步图片任务、图片计费、强制优先级、自定义菜单及俄语翻译仍在。已核对的关键后端回归测试通过。
- 未证实客户所述 GPT-5.6 偶发“没有工具”由这次合并引起，也不能据此排除上游或客户端问题。本次测试修复不是该客户问题的修复。

## 本次修复

文件：`frontend/src/views/admin/__tests__/GroupsView.codexManifest.spec.ts`。

- 补充 `useAuthStore`（读取登录及运行模式状态的入口）的测试替身，提供页面使用的 `isSimpleMode`（简易模式开关）。
- 同步 `getModelAllowlistCandidates`（获取可选白名单模型的接口）及 `model_allowlist`（分组模型白名单数据）名称，移除测试中的旧字段名。
- 保留原有“连续两次子组件更新后，启用状态和账号选择均不丢失”的断言，没有跳过测试或削弱校验。
- 原失败是测试环境缺少状态依赖；生产入口已注册 Pinia（前端共享状态管理库），不能把该测试失败直接解释为生产页面故障。
- 没有修改生产业务代码、线上白名单或数据库。

## 实际验证结果

命令按下表工作目录运行；均在本地 Windows / PowerShell 环境执行。后端使用 Go 1.27.0。

| 工作目录 | 命令 | 实际结果 |
| --- | --- | --- |
| frontend | `pnpm exec vitest run src/views/admin/__tests__/GroupsView.codexManifest.spec.ts src/views/admin/__tests__/GroupsView.duplicate.spec.ts` | 修复后 2 个文件、8 项测试通过 |
| frontend | `pnpm exec vitest run --reporter=json --outputFile=../.cache/review-v023-fixed-front-all.json` | 修复后 262 个文件、1898 项测试全部通过，0 失败、0 待执行 |
| frontend | `pnpm run build` | 修复后通过，包含翻译完整性、生产源码类型检查及打包 |
| backend | `go build ./...` | 通过，本次修复后再次执行通过 |
| backend | `go vet ./...` | 静态检查通过，本次修复后再次执行通过 |
| backend | `go test -tags=unit ./internal/service -run 'TestForwardResponses_\|Test.*Group\|Test.*FastPolicy' -count=1` | 前一轮复核通过；本次未修改后端代码，未重跑该命令 |
| backend | `go test -tags=unit ./internal/server/middleware -run 'Test.*GroupModelAllowlist' -count=1` | 本次白名单请求准入测试通过 |
| backend | `go test -tags=unit ./...` | 前一轮复核有 8 项失败，见下文；本次未重跑后端全量 |
| 仓库根目录 | `git diff --check` | 本次修改的空白格式检查通过 |

Vitest 是前端测试执行器；pnpm 是项目使用的包管理与脚本工具。`-tags=unit` 用于包含由构建标签控制的 Go 单元测试；不带标签出现 `no tests to run`（没有执行到测试）不能算验证通过。前端构建配置排除了测试文件，因此类型检查通过也不能替代运行测试。

构建仍有浏览器兼容数据过旧、部分打包文件超过建议大小的提示，本次不扩展范围处理。

本地原始日志位于被版本控制忽略的 `.cache/`：

- 修复后前端：`review-v023-fixed-front-all.json`、`review-v023-fixed-front-all.log`、`review-v023-fixed-build.log`。
- 白名单入口测试：`review-v023-allowlist-middleware.log`。
- 修复前复核：`review-v023-front-all.json`、`review-v023-full-unit.log`、`review-v023-backend.log`。

### 后端尚未通过的 8 项测试

以下为前一轮全量执行的真实失败，不归并成“全部通过”，也没有在本次修改中屏蔽：

| 测试名 | 日志中的失败原因 |
| --- | --- |
| `TestPgDumperHoldsMigrationLockThroughReaderClose` | 找不到 sh（测试调用的 Unix 命令解释器），并出现模拟解锁预期不匹配 |
| `TestPgDumperReleasesMigrationLockWhenProcessFails` | 同上 |
| `TestPgDumperReportsUnlockFailureAndDiscardsConnection` | 同上 |
| `TestPluginPackageInstallerInstallUnsignedDevelopmentPackage` | 插件包文件重命名时，Windows 报文件仍被占用 |
| `TestPluginPackageInstallerAllowsRepeatedIdenticalUpload` | 同上 |
| `TestPluginPackageInstallerVerifiesTrustedSignature` | 同上，不是签名校验失败 |
| `TestPluginPackageInstallerKeepsHostVersionMismatchDisabled` | 同上 |
| `TestContentModerationRuntimeSnapshotRefreshFailureKeepsStaleConfig` | 1 秒内未满足等待条件 |

插件包实现及测试、内容审核实现及相关测试在本次 v0.2.1 到 v0.2.3 合并中未变化；内容审核的计时失败也已列入 `scripts/verify-sync.sh`。这些证据不等于已经证明所有失败都是无害环境问题。插件文件占用至少反映本地平台兼容性问题；正式发布前应在目标 Linux 环境复跑相关测试。本次没有完成 Linux 对照运行。

## 历史合并说明的纠正

`84a0139c6` 提交说明中的“261 文件 / 1923 用例全通过”“后端仅 2 项失败”与本次实际结果不一致，不应继续作为验收依据。

- 修复前实测：前端 262 个文件、1898 项测试，1897 通过、1 失败。
- 本次修复后：前端 262 个文件、1898 项全部通过。
- 后端全量失败是上表 8 项，而不是提交说明列举的两个测试；其中插件失败的直接原因是文件占用，不是签名环境。
- 本记录不追认原说明中“已用纯上游隔离工作目录证明为环境问题”的结论，未独立完成的对照不记为已完成。

## 上线前必须核对：模型白名单语义变化

`backend/migrations/235_group_model_allowlist.sql`（数据库结构升级脚本）把 `groups.models_list_config`（旧模型列表展示配置）更名为 `model_allowlist`，配置数据原样保留，但现在同时限制请求是否允许进入。`236_group_model_allowlist_repair.sql` 负责修复列缺失或新旧列并存等结构状态。

请求准入按客户端发送的模型名判断，在账号映射和合成分组路由改写前执行；支持代码定义的名称归一化规则及末尾 `*` 通配符，并不是任意别名都自动放行。实现见 `backend/internal/service/group_model_allowlist.go` 和 `backend/internal/server/middleware/group_model_allowlist.go`（进入业务处理前的请求检查）。

风险：旧配置即使以前只用于隐藏模型，现在也可能拒绝客户端一直在使用的模型别名。普通入口被拒绝时返回 HTTP 404（请求的模型不允许使用），可在运维信息中查找 `model_not_allowed`（模型准入拒绝标记）或 `local_model_configuration`（本地模型配置原因）。这不等同于“请求成功但工具消失”。

本地常见配置位置的文件清单仅发现示例配置、部署模板及项目配置，未发现可直接核对的生产分组数据快照；没有连接线上数据库。因此实际受影响分组数量仍待确认。

上线前检查步骤：

1. 备份数据库并确认恢复方法；迁移可能重命名列，不要假设仅切回旧镜像就能安全回滚，也不要直接修改迁移记账记录。
2. 在管理页面逐一核对已开启模型白名单的分组，重点检查客户端实际发送的模型名和自定义别名。只加入业务需要的模型，不要为绕过排查而统一关闭白名单或填入 `*`。
3. 若采用数据库查询核对，由有权限的运维人员在对应环境执行下方只读查询，先确认实际列名，再读取配置；无需导出账号密钥或完整请求内容。
4. 在预发布环境，用授权账号验证允许模型可调用、禁止模型被拒绝，并回归模型别名、长连接后续轮次、自有图片计费及发票/异步图片页面。

只读结构检查（SQL 是数据库查询语言）：

```sql
SELECT table_schema, table_name, column_name
FROM information_schema.columns
WHERE table_name = 'groups'
  AND column_name IN ('models_list_config', 'model_allowlist');
```

确认新列存在后，在正确数据库和表所在模式下执行：

```sql
SELECT id, name, model_allowlist
FROM groups
WHERE model_allowlist ->> 'enabled' = 'true'
ORDER BY id;
```

如果数据库只有旧列，则使用 `models_list_config` 替换第二段查询中的两处新列名；不要为了执行检查而改动表结构。

## 本次未执行

复核完成时尚未提交或推送本次修改；后续提交和推送以仓库历史及远端状态为准。此次复核未创建发布标签，未构建或发布容器镜像，未部署服务器，未执行真实数据库迁移，未修改线上白名单。

客户“工具消失”的定位仍需经脱敏的客户端实际请求、代理转发内容及响应对照，确认工具声明和工具调用结果在哪一环节发生变化。模型对自身工具的文字描述不是实际请求证据，本记录不把其作为故障根因。
