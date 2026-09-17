# 独立的一次性 IPv6 出口池

这是单独部署的 Python（程序运行环境）服务，不导入、不修改旧 IPv6 代理管理器。源码放在本仓库 tools（辅助工具目录）下以便一起版本管理，但它不在 sub2api 进程中运行，也不使用业务数据库。

## 行为

- 后台预生成 24 个可用 IPv6（第六版互联网地址），每批最多并行生成 6 个。
- 从配置的专用子网随机选地址，保持操作系统的重复地址检测，验证到 chatgpt.com 的 IPv6 连接后才入池。
- 新地址使用 preferred_lft=0（不成为普通业务连接的首选源地址）；代理显式绑定分配地址。
- 每次租用独占地址和随机凭证；租期默认 120 秒。
- 归还或到期后关闭该租用连接并删除自己的地址，永久记录为退役，不重新分配。不会把已用地址放回可用池。
- SQLite（本机文件数据库）持久化地址归属和退役记录；重启后废弃旧租约。
- 仅允许指定的目标域名和端口，不是公共开放代理。
- 使用 TLS（网络传输加密）保护管理凭证和代理连接；生成独立证书，不改动旧网站证书。
- 有限的就绪库存、活跃租约和连接数；采集方还需限制任务次数。

生成失败的候选也会被清理并退役。启动或运行时出现外部地址冲突，不接管也不删除该外部地址。

## 部署

适用于有可用、可路由 IPv6 地址段的 Linux（服务器操作系统），需要 Python 3.10+、openssl（证书工具）、iproute2（网络地址管理工具）和 systemd（服务管理器）。安装和运行需要网络地址管理权限。

先核对真实地址段、网卡、端口及旧服务地址。不要把其他服务使用的专用子网配置给本服务。安装器不会修改默认路由、系统网络参数、防火墙或旧代理文件。

在源码目录执行：

    python3 -m unittest -v test_pool
    python3 install.py --public-ip YOUR_SERVER_IPV4 --prefix YOUR_DEDICATED_IPV6_SUBNET --interface eth0

安装器发现已有新服务配置时拒绝直接覆盖。

独立资源：

| 路径或名称 | 用途 |
| --- | --- |
| /opt/sub2api-turn-state-egress | 新程序目录 |
| /etc/sub2api-turn-state-egress/config.json | 新服务配置和管理密钥 |
| /etc/sub2api-turn-state-egress/ca.crt | 提供给调用方的公开信任证书 |
| /etc/sub2api-turn-state-egress/sub2api.env | 提供给业务服务的私密连接参数 |
| /var/lib/sub2api-turn-state-egress/pool.sqlite3 | 自己的地址归属和退役记录 |
| sub2api-turn-state-egress.service | 新服务名称 |
| 18443 | 默认加密管理与代理共用端口 |

ca.key（证书签发私钥）和 server.key（服务证书私钥）不得复制到业务服务或提交到仓库。证书存在有效期，运维需在到期前更换。

尽量在外围防火墙上仅允许业务服务器访问新端口。不要停止旧服务或清空网卡上的所有 IPv6 地址。

## 接口

所有管理接口需要 Authorization: Bearer ...（管理密钥认证头）。

| 请求 | 用途 |
| --- | --- |
| GET /v1/status | 查询库存、生成中状态和错误摘要 |
| POST /v1/leases | 原子领取一个已准备好的独占出口 |
| DELETE /v1/leases/{id} | 退役租用，重复释放不会重复分配 |

领取结果包含 id（租用标识）、ipv6（实际源地址）、expires_at（到期时间）和 proxy_url（包含本次随机凭证的加密代理地址）。禁止将完整结果写入普通日志。

同一个加密端口支持 CONNECT（创建目标网络隧道的代理方法），使用租用对应的 Basic（用户名密码）代理认证，不使用管理密钥作为代理密码。

## 验证与停止

    python3 smoke_test.py
    systemctl status sub2api-turn-state-egress
    journalctl -u sub2api-turn-state-egress

smoke_test.py（实际链路验证程序）分配三个地址，通过代理向地址查询服务验证真正的 IPv6 出口，结束时释放租用；不使用上游账号凭证，不执行模型请求。目标网站偶发错误会使该轮验证失败，不等于所有出口不可用。

停止只影响新池：

    systemctl stop sub2api-turn-state-egress

程序在正常退出时只清理自己数据库记录的地址。不要删除数据库再清理网络，否则会丢失地址归属与已用地址记录。
