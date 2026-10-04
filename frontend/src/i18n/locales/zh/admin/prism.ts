export default {
  prism: {
    title: 'Prism 浏览器通道',
    description: '通过服务器上的独立浏览器服务调用 Prism，逐账号、逐模型开启。默认关闭。',
    models: '使用 Prism 的模型',
    modelsHint: '按账号映射后的模型名匹配；未勾选模型走原路径，全部取消表示不使用 Prism。Astra 借票独立运行。',
    requirements: '请先确认共享服务处于运行中。此通道暂无可靠用量统计，当前不计费；服务失败不会自动换通道。',
    runtime: {
      title: '共享浏览器服务', start: '启动', stop: '停止', restart: '重启', check: '检查连接', logs: '运行日志', cancel: '取消',
      scope: '账号开关随账号保存，只影响当前账号。此处服务操作立即生效，影响本实例全部 Prism 账号。',
      confirm: '停止或重启将中断本实例所有 Prism 账号正在进行的请求。会话状态会保留，请勿自动重发结果不明的请求。确定继续？',
      states: { loading: '读取中', not_installed: '未安装集成服务', unmanaged: '外部管理', misconfigured: '配置异常', unreachable: '管理服务不可达', stopped: '已停止', starting: '启动中', running: '运行中', error: '异常' },
      hints: {
        install: '首次需升级为包含浏览器的集成部署版本；完成后即可在此启停，无需手动设置内部密钥。',
        unmanaged: '当前使用单独部署的浏览器服务。切换到集成版本后才可在此控制；不会接管或停止现有外部服务。',
        gatewayDisabled: '浏览器服务可管理，但主程序尚未开启 Prism。集成部署会自动配置此开关。'
      },
      loadFailed: '无法读取服务状态，请检查网络后重试。', actionFailed: '操作未确认成功，请检查连接和运行日志；不要连续重复操作。',
      checkOK: '浏览器转接进程可连接；账号权限和模型是否可用，仍需在账号测试中验证。', checkFailed: '浏览器转接进程尚未就绪，请查看运行日志。',
      logsHint: '本实例最近 200 条服务事件，自动刷新；管理进程重启后清空。不包含凭据、对话正文或原始浏览器日志。', noLogs: '暂无事件',
      events: { browser_missing: '缺少匹配的浏览器，请重新构建集成镜像', sandbox_missing: '缺少浏览器沙箱，请重新构建集成镜像', non_root_required: '浏览器必须由普通用户运行', adapter_configuration_invalid: '浏览器服务配置不完整', dependencies_missing: '缺少工具验证依赖，请重新构建集成镜像', manager_ready: '管理服务就绪', port_in_use: '浏览器端口已被其他服务占用', service_starting: '正在启动浏览器转接服务', service_start_failed: '启动失败', service_stopped: '服务已停止', service_exited: '服务意外退出，按冷却策略重试', service_ready: '服务已就绪', health_failed: '服务健康检查失败', check_ok: '连接检查通过', check_failed: '连接检查未通过', control_failed: '服务控制失败', prism_adapter_error: '浏览器转接请求发生错误', prism_worker_close_error: '浏览器关闭过程发生错误', unknown_event: '其他服务事件' }
    },
    toolsHint: '客户端使用 HTTP /v1/responses（普通请求接口）。仅 6.1 Sol 支持客户端工具调用；目前不支持图片、独立上下文压缩或原生双向长连接。'
  }
}
