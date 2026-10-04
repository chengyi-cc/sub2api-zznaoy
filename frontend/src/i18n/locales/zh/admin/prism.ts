export default {
  prism: {
    title: 'Prism 浏览器通道',
    description: '通过服务器上的独立浏览器服务调用 Prism，逐账号、逐模型开启。默认关闭。',
    models: '使用 Prism 的模型',
    modelsHint: '按账号映射后的模型名匹配；未勾选模型走原路径，全部取消表示不使用 Prism。Astra 借票独立运行。',
    requirements: '需先部署 Linux 浏览器适配服务并开启服务器总开关。此通道暂无上游用量统计，当前不计费；适配器失败不会自动换通道。',
    toolsHint: '客户端使用 HTTP /v1/responses（普通请求接口）。仅 6.1 Sol 支持客户端工具调用；目前不支持图片、独立上下文压缩或原生双向长连接。'
  }
}
