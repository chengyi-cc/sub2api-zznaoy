import { transparentGoldArt } from './transparent-artwork.mjs'

const symbol = '<svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M7 24V8h5l9 16h4V8" stroke="currentColor" stroke-width="2.2"/></svg>'
const brand = `<a class="hf-brand" href="#hf-top"><span class="hf-brand-symbol">${symbol}</span><span>zz.naoy</span></a>`
const header = (extra = '') => `<header class="hf-header hf-wrap">${brand}<nav aria-label="主导航"><a href="#hf-capabilities">平台优势</a><a href="#hf-models">模型生态</a><a href="#hf-guide">接入指南</a></nav><a class="hf-login" href="/dashboard" target="_top">进入控制台 <span aria-hidden="true">↗</span></a>${extra}</header>`
const actions = (primary = '开始构建', secondary = '了解接入方式') => `<div class="hf-actions"><a class="hf-button hf-primary" href="/dashboard" target="_top">${primary}<span aria-hidden="true">↗</span></a><a class="hf-button hf-secondary" href="#hf-guide">${secondary}<span aria-hidden="true">→</span></a></div>`
const stage = (label = 'CONNECTED INTELLIGENCE') => `<div class="hf-stage"><div class="hf-stage-top"><span>${label}</span></div><input class="hf-motion-pause" id="hf-motion-pause" type="checkbox" aria-label="暂停主视觉动效"><div class="hf-art" data-hc-artwork="random">${transparentGoldArt}</div><p class="hf-interaction-hint">点击转动 · 按住拖拽旋转<span>方向键也可调整</span></p><div class="hf-stage-bottom"><span><i aria-hidden="true"></i><span data-hc-artwork-name>金色轨道</span></span><span class="hf-art-controls"><button type="button" class="hf-art-reset" data-hc-art-reset aria-label="恢复主视觉初始角度">复位 ↺</button><label class="hf-motion-control" for="hf-motion-pause"><span class="hf-motion-running">暂停动效 Ⅱ</span><span class="hf-motion-stopped">继续动效 ▷</span></label></span></div></div>`
const models = () => `<section class="hf-models hf-wrap" id="hf-models"><div><span class="hf-eyebrow">A CONNECTED ECOSYSTEM</span><p>连接不同模型，延展你的可能。</p></div><div class="hf-model-list" aria-label="模型生态示意"><span><b aria-hidden="true">✳</b> Claude</span><span><b aria-hidden="true">◎</b> GPT</span><span><b aria-hidden="true">✦</b> Gemini</span></div><small>模型生态示意；实际开放模型与价格以控制台为准。</small></section>`
const capabilities = (title = '少一些繁琐，多一些创造。') => `<section class="hf-section hf-wrap" id="hf-capabilities"><div class="hf-section-heading"><div><span class="hf-eyebrow">THE ESSENTIALS / 平台优势</span><h2>${title}</h2></div><p>把连接和管理做清楚，<br>让你专注下一步。</p></div><div class="hf-features"><article><span class="hf-feature-index">01 / CONNECT</span><div class="hf-feature-icon" aria-hidden="true">↗</div><h3>一个入口，连接所需。</h3><p>通过兼容接口连接你的应用与模型，不再让重复配置打断工作。</p></article><article><span class="hf-feature-index">02 / ORGANIZE</span><div class="hf-feature-icon" aria-hidden="true">⌘</div><h3>管理有序，心中有数。</h3><p>在同一个控制台管理访问密钥，为不同应用梳理清楚使用边界。</p></article><article><span class="hf-feature-index">03 / UNDERSTAND</span><div class="hf-feature-icon" aria-hidden="true">◴</div><h3>每次使用，都有记录。</h3><p>集中查看调用记录与用量信息，让每一次使用都清晰可查。</p></article></div></section>`
const guide = () => `<section class="hf-section hf-wrap hf-guide" id="hf-guide"><div class="hf-section-heading"><div><span class="hf-eyebrow">THREE SIMPLE STEPS / 接入指南</span><h2>从这里，到你的应用。</h2></div><a href="/keys" target="_top" class="hf-text-link">前往密钥管理 ↗</a></div><div class="hf-steps"><article><b>01</b><div><h3>进入控制台</h3><p>登录账户，查看可用模型和账户设置。</p></div></article><article><b>02</b><div><h3>创建访问密钥</h3><p>创建 API Key（接口访问凭证），用于连接你的应用。</p></div></article><article><b>03</b><div><h3>完成应用配置</h3><p>使用控制台提供的接口地址、模型名称与密钥。</p></div></article></div></section>`
const faq = () => `<section class="hf-section hf-wrap hf-faq"><div><span class="hf-eyebrow">BEFORE YOU START / 常见问题</span><h2>你关心的，<br>先说清楚。</h2></div><div class="hf-faq-list"><details><summary>现有应用能直接接入吗？<span aria-hidden="true">＋</span></summary><p>如果应用支持所选模型的兼容接口，通常可以通过更改接口地址、模型名称和访问密钥完成接入。具体配置请以控制台说明为准。</p></details><details><summary>支持哪些模型，如何计费？<span aria-hidden="true">＋</span></summary><p>以控制台实际显示的模型、所属分组和计费规则为准。页面上的品牌名称用于介绍模型生态，不代表所有模型均已开放。</p></details><details><summary>在哪里查看调用与用量？<span aria-hidden="true">＋</span></summary><p>登录后可在控制台查看调用记录、用量与账户余额，也可以通过页面底部的密钥用量查询入口了解相关记录。</p></details></div></section>`
const finalCta = () => `<section class="hf-final hf-wrap"><div><span class="hf-eyebrow">YOUR NEXT IDEA STARTS HERE</span><h2>下一次创造，从这里开始。</h2></div><a class="hf-button hf-primary" href="/dashboard" target="_top">进入控制台 <span aria-hidden="true">↗</span></a></section>`
const footer = () => `<footer class="hf-footer hf-wrap">${brand}<p>为创造而连接。</p><a href="/key-usage" target="_top">密钥用量查询 ↗</a></footer>`
const page = (theme, hero, content, extra = '') => `<div class="hf-site hf-${theme}" id="hf-top"><a class="hf-skip" href="#hf-main">跳到主要内容</a>${header(extra)}<main id="hf-main">${hero}${content}${guide()}${faq()}${finalCta()}</main>${footer()}</div>`

export const pageStyles = [
  {
    id: '11-cloud', name: '云白极简', light: true,
    caption: 'CLOUD / A CLEARER WAY', tag: '纯净白底 · 清晰产品感',
    description: '白色大留白、清晰中文标题与克制的信息层级。右侧为透明悬浮的冷银蓝线条，没有黑底展台，直接融入页面。',
    html: page('cloud', `<section class="hf-hero hf-wrap"><div class="hf-copy"><div class="hf-pill"><span aria-hidden="true">✦</span> 更简单的模型连接方式</div><h1>让智能接入，<br><span>简单一点。</span></h1><p class="hf-lead">把不同模型连接到同一个清晰的入口。<br>少一些配置，多一些把想法实现的时间。</p>${actions('开始使用', '看看如何接入')}<div class="hf-hero-note"><span aria-hidden="true">✓</span> 统一入口 <span aria-hidden="true">✓</span> 密钥管理 <span aria-hidden="true">✓</span> 用量记录</div></div>${stage('ONE CONNECTION. MORE POSSIBILITIES.')}</section>`, `${models()}${capabilities('强大的能力，不必复杂。')}`),
  },
  {
    id: '12-command', name: '深蓝指挥',
    caption: 'COMMAND / BUILT TO CONNECT', tag: '深海蓝黑 · 开发者平台',
    description: '深海蓝黑、等宽小标题与结构化信息面板。把首页做成专业基础设施入口，表达直接，强调接入步骤而不是奢华装饰。',
    html: page('command', `<div class="hf-command-rail"><div class="hf-wrap"><span>FOR DEVELOPERS. FOR YOUR NEXT IDEA.</span><span>连接 · 管理 · 构建</span></div></div><section class="hf-hero hf-wrap"><div class="hf-copy"><p class="hf-eyebrow"><span class="hf-signal"></span> THE INTELLIGENCE INTERFACE</p><h1>你的应用，<br>下一层<span>智能。</span></h1><p class="hf-lead">把模型连接交给统一入口。<br>你负责构建，我们让接入与管理更加清晰。</p>${actions('打开控制台', '查看接入步骤')}<div class="hf-command-chips"><span>兼容接口</span><span>访问凭证</span><span>调用记录</span></div></div>${stage('INTELLIGENCE TOPOLOGY / 连接结构')}</section><section class="hf-config-strip hf-wrap" aria-label="接入配置示意"><div><span class="hf-eyebrow">YOUR CONFIGURATION</span><b>三个配置，开始连接。</b></div><div><span>接口地址</span><code>BASE_URL</code></div><div><span>访问密钥</span><code>API_KEY</code></div><div><span>模型名称</span><code>MODEL</code></div><small>按控制台实际配置填写</small></section>`, `${capabilities('把基础工作，交给基础设施。')}${models()}`),
  },
  {
    id: '13-editorial', name: '砂岩编辑', light: true,
    caption: 'EDITORIAL / A CONSIDERED APPROACH', tag: '砂岩暖白 · 中文编辑美学',
    description: '暖白纸感、中文衬线字体、章节式排版与赭色细节。像一本清晰的产品刊物；与 05 不同，这里由中文内容与留白组织页面。',
    html: page('editorial', `<div class="hf-editorial-meta hf-wrap"><span>连接的另一种可能</span><span>DESIGNED AROUND YOUR IDEAS</span></div><section class="hf-hero hf-wrap"><div class="hf-copy"><p class="hf-eyebrow">01 / 从一个好想法开始</p><h1>让技术退后，<br>让<span>想法向前。</span></h1><p class="hf-lead">工具的意义，不是增加复杂。<br>而是让每一次思考，都有继续生长的空间。</p>${actions('开始你的创造', '阅读接入指南')}<div class="hf-editorial-signature"><span></span> 一处连接，从容创造。</div></div><div class="hf-editorial-figure">${stage('FIGURE 01 / THE ART OF CONNECTION')}<p><b>图 / 智能的形态</b><span>六种主视觉，打开时随机呈现。</span></p></div></section>`, `${models()}<div class="hf-editorial-statement hf-wrap"><span>不是更多按钮。</span><h2>而是更少阻力。</h2><p>把重要的能力放在一起，<br>把宝贵的注意力留给创造。</p></div>${capabilities('好的体验，自有章法。')}`),
  },
  {
    id: '14-cobalt', name: '钴蓝先锋',
    caption: 'COBALT / MAKE SOMETHING GREAT', tag: '大胆钴蓝 · 强品牌冲击',
    description: '大面积钴蓝、奶油色按钮、超大中文标题与平直分区。更大胆、更有品牌辨识度；图形是内容的一部分，而不是整页风格的唯一来源。',
    html: page('cobalt', `<section class="hf-hero hf-wrap"><div class="hf-copy"><p class="hf-eyebrow">LESS SETUP. MORE WHAT-IF.</p><h1>好想法，<br><span>即刻连接。</span></h1><p class="hf-lead">让模型能力，进入你的工作流。<br>从一个灵感，到一个真正能运行的作品。</p>${actions('现在开始', '探索平台能力')}<div class="hf-cobalt-counter"><b>01 → ∞</b><span>从一个入口，<br>走向更多可能。</span></div></div><div class="hf-cobalt-figure">${stage('THE SHAPE OF YOUR NEXT IDEA')}</div></section><div class="hf-cobalt-ribbon"><span>连接模型</span><b aria-hidden="true">✳</b><span>保持专注</span><b aria-hidden="true">✳</b><span>继续创造</span><b aria-hidden="true">✳</b><span>MAKE IT REAL.</span></div>`, `${capabilities('想法够大，接入够简单。')}${models()}`),
  },
  {
    id: '15-silver', name: '雾银未来', light: true,
    caption: 'SILVER / INTELLIGENCE IN FOCUS', tag: '冷银雾白 · 精致工具感',
    description: '冷银灰、低饱和蓝紫和柔和的分块布局。像新一代生产力产品：精致但不厚重，重点在清楚的信息层级与自然的操作入口。',
    html: page('silver', `<section class="hf-hero hf-wrap"><div class="hf-silver-copy"><div class="hf-copy"><p class="hf-pill"><span aria-hidden="true">✧</span> 为下一种工作方式而来</p><h1>连接所想。<br><span>专注所长。</span></h1><p class="hf-lead">将模型、应用与灵感连在一起。<br>用更清晰的入口，开启更流畅的工作。</p>${actions('进入工作空间', '了解接入方式')}</div><div class="hf-silver-mini"><div><span aria-hidden="true">⌘</span><b>一处管理</b><small>访问密钥与应用配置</small></div><div><span aria-hidden="true">◴</span><b>清晰可见</b><small>调用记录与用量信息</small></div></div></div><div class="hf-silver-figure">${stage('INTELLIGENCE, IN FOCUS')}<div class="hf-silver-under"><span>你熟悉的工具，更多的可能。</span><b aria-hidden="true">↗</b></div></div></section>`, `${models()}${capabilities('连接更自然，工作更专注。')}`),
  },
]
