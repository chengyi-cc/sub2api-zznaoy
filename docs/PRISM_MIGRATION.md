# 第二阶段：Prism 独立通道

## 已核对的上游

2026-10-04 查询：官方 Wei-Shaw/sub2api 最新发布仍为 v0.2.13，本项目已包含该版本；main 相比发布标签只多版本号同步提交。

参考分支 ranxi2001/sub2api 最新发布仍为 v2.9.8，但 production 已增加 12 次提交。Prism 修复 PR #292 已合并。本次固定参考提交 `0ae36e501952000c5c910a2e616c6e0861f66a49`，包含内存回收、有界启动等待、项目启动限流、明确沙箱重连及脱敏错误分类。

来源：https://github.com/ranxi2001/sub2api/pull/292

完整上游浏览器实现和测试位于 `prism-adapter/`。网关按本项目的账号、会话、调度与计费接口适配，不整体合并参考分支，不改动普通通道的会话解析。第一阶段借票另有独立配置和文档。

## 功能和开关

1. **服务器总开关**：`GATEWAY_PRISM_BROWSER_ENABLED`，程序默认 false。还需配置本机适配器地址 `GATEWAY_PRISM_BROWSER_BASE_URL=http://127.0.0.1:8319/v1`，以及桥接密钥 `GATEWAY_PRISM_BROWSER_API_KEY`。
2. **逐账号开关**：账号新建或编辑中的“Prism 浏览器通道”，默认关闭。只适用于直接 OpenAI OAuth（登录授权）账号，不适用于上游接口密钥、手动令牌、影子账号或代理身份。
3. **模型范围**：勾选 `gpt-6.1-sol`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-6-luna`。按账号模型映射后的名字匹配；显式空列表表示不走 Prism，未勾选的模型保留原路由。Astra 不在此列表中，继续使用第一阶段借票或普通通道。

账号选择 Prism 后，服务器未就绪或适配器拒绝时会明确失败，不偷偷改走普通通道或其他模型。模型真实权限仍以该账号 Prism 页面为准。

启停通道或改变模型范围后，新建客户端会话再验证，避免沿用旧连接及其他通道产生的加密历史。

客户端使用原有地址、密钥和 HTTP `/v1/responses`（普通请求接口）；支持文本及 6.1 Sol 的客户端工具调用。工具由客户端在自己的权限下执行，网关与适配器不执行用户工具代码。

上游目前不提供可靠用量，返回 `usage: null`，本项目不会捏造用量或按零用量记账，因此这是未计费的试验通道。上游账号额度仍可能消耗。结果按上游终态返回，不模拟逐字输出。

当前边界包括：不支持图片、独立 compact（上下文压缩）、原生 WebSocket（双向长连接）、仅凭上一响应编号的续接，以及 Chat Completions / Messages 两种兼容入口。长上下文和工具载荷仍受适配器实际限制；详细边界见 `prism-adapter/README.md`。

## 部署组成

新网关与匹配的浏览器适配器必须一起部署。单独替换主程序不会安装浏览器。适配器要求 Linux、Python 3.12、固定版本的 Playwright（浏览器自动控制库）及匹配 Chromium（浏览器内核），以非 root 用户运行并保留浏览器沙箱。

所有桥接通信限制在本机回环地址，不使用账号代理、环境代理或重定向传递登录凭据。浏览器当前不使用各账号的代理；先确认服务器本身能够访问 Prism。

先保持串行浏览器模式（`PRISM_ADAPTER_MODE=browser`），用一个账号完成短文本和工具回传验收，再单独评估并发模式。本次保留最新并发修复，但不把配置上限当作已实测容量。

### 已有 Docker Compose 部署

部署包提供 `gateway/`（预编译网关和原项目运行资源）、`prism-adapter/`、`deploy/prism/compose.prism.yml`。Compose 是容器编排配置。适配器和主程序共享网络命名空间，因此主程序可以访问 `127.0.0.1:8319`；不会把适配器端口公开到外网。

overlay（追加配置）默认主服务名称为 `sub2api`。必须沿用现有项目名、原 Compose 配置、原 `.env`、数据库及数据目录；第二套实例不能套用第一套实例的项目名或数据目录。部署前保存原镜像及配置、备份数据库。

生成专用桥接密钥文件（脚本不会打印密钥，也不会覆盖已有文件）：

```sh
python3 /absolute/release/deploy/prism/prepare_env.py --output /absolute/private/prism.env
export PRISM_BUNDLE_DIR=/absolute/release
```

使用原有项目名及配置，加上新环境文件和 overlay：

```sh
docker compose -p ORIGINAL_PROJECT --env-file /original/.env --env-file /absolute/private/prism.env \
  -f /original/compose.yml -f "$PRISM_BUNDLE_DIR/deploy/prism/compose.prism.yml" config --quiet
docker compose -p ORIGINAL_PROJECT --env-file /original/.env --env-file /absolute/private/prism.env \
  -f /original/compose.yml -f "$PRISM_BUNDLE_DIR/deploy/prism/compose.prism.yml" build sub2api prism-adapter
docker compose -p ORIGINAL_PROJECT --env-file /original/.env --env-file /absolute/private/prism.env \
  -f /original/compose.yml -f "$PRISM_BUNDLE_DIR/deploy/prism/compose.prism.yml" up -d --no-build --no-deps --force-recreate sub2api prism-adapter
```

主程序镜像只包装预编译二进制；适配器镜像安装固定版本运行依赖，不在生产机编译 Go 或前端。浏览器保留沙箱并使用 Playwright 官方 v1.63.0 的 seccomp（系统调用过滤配置），没有使用 privileged 或关闭沙箱。

`prism_state` 数据卷保存未决请求与工具状态。保留它，不执行 `down -v`，不要通过清空状态解决未知回合错误。已有适配器的会话/状态需要核对后迁移，不能假设新空卷就是原状态。主容器重新创建时，应一起重建适配器容器，以绑定新的网络命名空间。

### 已有 systemd 原生部署

systemd 是 Linux 服务管理程序。参考 `prism-adapter/sub2api-prism-adapter.service` 和 `sub2api-prism.conf`，按实际服务用户、主程序单元名、安装目录及状态目录调整。

使用 Python 3.12 创建适配器虚拟环境，按 `requirements.txt` 安装二进制依赖，并安装匹配 Chromium。将同一随机桥接密钥分别设置为 `PRISM_ADAPTER_API_KEY` 与 `GATEWAY_PRISM_BROWSER_API_KEY`，只写入权限受限的环境文件。还需配置 `PRISM_ADAPTER_CHROME`、`CHROME_DEVEL_SANDBOX`、`PRISM_ADAPTER_STATE_DIR`。

模板默认 `/opt/sub2api/prism-adapter`、用户 `sub2api`、状态 `/var/lib/sub2api-prism`、环境文件 `/etc/sub2api-prism.env`。若有多个实例，分别使用独立端口、密钥、服务名和状态目录。保留旧二进制及适配器，先启动新适配器并确认健康，再更新、重启主程序。

## 验收与回退

- 适配器 `/health` 仅证明进程可访问，不证明登录和模型可用。
- 先在 Linux 上执行适配器完整单测及本地模拟浏览器测试；脚本 `deploy/prism/verify_linux.sh` 不调用真实账号。容器中可运行 `python3 -m unittest discover -s /opt/sub2api/prism-adapter -p 'test_*.py' -v`。
- 用一个实际授权账号，在账号测试页选择已勾选模型，检查成功终态与正文；随后测试客户端工具调用、结果回传及后续回答。真实测试会消耗额度。
- 测试同一会话复用，以及不同密钥、账号、线程之间的隔离。第一阶段 Astra 借票也需复测。
- 账号 Prism 关闭后恢复原路由；需要回退整个版本时，先关闭相关账号开关，再恢复旧主程序。保留未知请求及工具状态，不擅自重放。

第二阶段没有新增数据库迁移。若第一阶段程序尚未部署，首次启动合并后的程序仍会执行第一阶段的 `245_astra_borrow_history.sql`。

## 本次本地验证的限制

本机是 Windows。网关与页面可以在本机验证；上游适配器完整测试使用 Linux 的目录同步及进程资源限制。在 Windows 执行完整 76 项测试时有 34 项因这些平台能力缺失报错，不能算作全部通过，也没有通过移除保护或跳过用例来掩盖它们。

已通过：网关/处理器的 Prism 与第一阶段借票专项测试及竞态检查，相关原生长连接、Excel 和用量记录回归；131 项前端回归、类型检查、代码规范检查和生产构建；Linux amd64 与 Windows amd64 主程序编译，Windows 版本启动检查；适配器 Python 语法、上游文件逐字一致性、密钥文件创建/防覆盖检查、部署脚本语法和追加配置的 YAML 解析检查。YAML 解析不等于实际 Docker 部署验收。

本机 Ubuntu 子系统未能启动，报告 `HCS_E_SERVICE_NOT_AVAILABLE`；没有可用 Docker 环境。因此容器运行、Linux 浏览器沙箱、适配器完整测试和真实账号请求，仍需在目标 Linux 服务器或独立验收机上完成。

部署目标尚未确定：旧本地工具记录中的服务器在 SSH 22 端口连接超时，另有 `/opt/sub2api` 与 `/opt/sub2api-2` 两套部署说明。本次不会猜测并覆盖其中任一实例。
