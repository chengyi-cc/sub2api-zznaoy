export default {
  prism: {
    title: 'Prism browser channel',
    description: 'Use a separate server browser service for Prism. Enable it per account and model. Off by default.',
    models: 'Models using Prism',
    modelsHint: 'Matches the account-mapped model name. Unselected models keep their original route; an empty selection disables Prism routing. Astra borrowing operates independently.',
    requirements: 'Deploy the Linux browser adapter and enable the server switch first. Upstream usage is unavailable, so requests are currently unbilled. Adapter failures never switch channels automatically.',
    toolsHint: 'Use HTTP /v1/responses. Only 6.1 Sol supports client tools. Images, standalone context compaction and native WebSocket are not supported.'
  }
}
