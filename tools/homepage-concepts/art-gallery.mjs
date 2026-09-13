import { artDirections } from './art-directions.mjs'
import { originalGoldArt } from './concepts.mjs'

export function createArtGallery(template) {
  const controls = artDirections.map((design, index) => `<button type="button" data-concept="${design.id}" aria-pressed="${index === 0}"><span class="concept-swatch art-swatch"><img src="${design.id}-mark.svg" alt=""></span><span><b>${design.id.slice(0, 2)} ${design.name}</b><small>${design.tag}</small></span><span class="concept-arrow">↗</span></button>`).join('\n')
  const original = `<button type="button" data-concept="05-atelier" aria-pressed="false"><span class="concept-swatch swatch-atelier"></span><span><b>05 原版 · 保留作对照</b><small>原始造型与比例，不作修改</small></span><span class="concept-arrow">↗</span></button>`
  return template
    .replace('zz.naoy · 五种首页，五种气质', '05 主视觉 · 五种新造型')
    .replace('src="preview-data.js"', 'src="art-directions-data.js"')
    .replace('</head>', '<link rel="stylesheet" href="art-gallery.css">\n</head>')
    .replace('<body>', '<body class="art-collection">')
    .replace('首页设计室<small>zz.naoy / DESIGN EXPLORATIONS', '主视觉探索室<small>ATELIER / FIVE NEW FORMS')
    .replace('五种首页，<br>五种气质。', '同一份质感，<br>五种新造型。')
    .replace('不是换一个颜色，<br>而是换一种让人记住你的方式。', '这里仅对比右侧图形，<br>整页风格请看“新五套页面风格”。')
    .replace(/<nav class="concept-nav"[\s\S]*?<\/nav>/, `<nav class="concept-nav" aria-label="选择主视觉造型">${controls}\n${original}</nav>`)
    .replace('五款同屏', '全部同屏')
    .replace('五款对比', '全部对比')
    .replace('href="05-art-directions.html"', 'href="05-art-overview.html"')
    .replace('主视觉新五款 ↗', '图形总览 ↗')
    .replace('<noscript>', '<noscript><p><a href="05-art-overview.html">打开无需脚本的五种图形总览</a></p>')
    .replace('设计预览，不影响现有站点', '原版保留 · 仅本地预览')
}

const overviewCards = [
  { id: '05-atelier', name: '原版 · 保留', tag: '当前原始造型，不作修改', art: originalGoldArt, vector: '05-mark.svg' },
  ...artDirections.map(design => ({ ...design, vector: `${design.id}-mark.svg` })),
]

export const artOverview = `<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="dark"><meta name="robots" content="noindex,nofollow"><title>05 右侧主视觉 · 原版与五种新造型</title><style>
*{box-sizing:border-box}body{margin:0;background:#151613;color:#e5dfd2;font-family:"Segoe UI","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}a{color:inherit;text-decoration:none}a:focus-visible{outline:1px solid #dac394;outline-offset:5px}.art-overview{width:calc(100% - 80px);max-width:1580px;margin:auto}.overview-header{display:flex;align-items:center;justify-content:space-between;gap:25px;padding:27px 0;border-bottom:1px solid #34352c}.overview-brand{font:25px Georgia,serif}.overview-header nav{display:flex;gap:25px;font-size:11px;color:#c1b396}.overview-intro{padding:46px 0 30px}.overview-eyebrow{font:9px Consolas,monospace;letter-spacing:2px;color:#a99b80}.overview-intro h1{font-size:32px;font-family:"Songti SC","SimSun",serif;font-weight:400;letter-spacing:2px;margin:14px 0}.overview-intro p{font-size:12px;color:#aaa496;line-height:1.9;margin:0}.overview-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0}.overview-card{padding:25px 23px;border-top:1px solid #34352c;border-right:1px solid #34352c;min-width:0}.overview-card:nth-child(3n){border-right:0}.overview-card h2{font-size:13px;font-weight:500;letter-spacing:.6px;margin:0 0 4px}.overview-card h2 span{font:10px Consolas,monospace;color:#938b77;letter-spacing:1px;margin-right:10px}.overview-card>p{font-size:10px;color:#aaa18b;margin:8px 0 0}.overview-card .hc-gold-art{width:100%;padding:0;aspect-ratio:600/590}.overview-card svg{width:100%;height:auto;display:block}.hc-gold-art-label,.hc-gold-coordinate{display:none}.overview-card footer{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:10px;color:#cdb88c;margin-top:4px}.overview-card footer a{padding:9px 0}.overview-card footer a:hover{color:#f0debb}.overview-footer{display:flex;align-items:center;justify-content:space-between;gap:30px;padding:27px 0 35px;border-top:1px solid #34352c;font-size:10px;color:#9d9583}.overview-footer a{color:#cdb88c;white-space:nowrap}
@media(max-width:1050px){.overview-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.overview-card:nth-child(3n){border-right:1px solid #34352c}.overview-card:nth-child(2n){border-right:0}.art-overview{width:calc(100% - 48px)}}
@media(max-width:620px){.art-overview{width:calc(100% - 36px)}.overview-grid{grid-template-columns:1fr}.overview-card,.overview-card:nth-child(3n){border-right:0;padding:22px 0}.overview-header{gap:16px;padding-block:23px}.overview-header nav{font-size:9px;gap:14px}.overview-brand{font-size:22px}.overview-intro{padding-block:30px}.overview-intro h1{font-size:25px;letter-spacing:1px}.overview-intro p{font-size:11px}.overview-footer{align-items:flex-start;flex-direction:column;gap:18px;line-height:1.8;font-size:9px}}
</style></head><body><div class="art-overview"><header class="overview-header"><a href="index.html#05-atelier" class="overview-brand">zz.naoy /</a><nav aria-label="页面导航"><a href="index.html#05-atelier">原五套首页</a><a href="05-art-directions.html">切换预览与导出 ↗</a></nav></header><main><section class="overview-intro"><span class="overview-eyebrow">ATELIER / FIVE NEW FORMS</span><h1>保留质感，探索另一种形态。</h1><p>原版保留在左上。以下五种新造型均已放入同一套 05 首页，可查看完整页面后再比较。<br>不是再换五种颜色，而是五种不同的空间结构。</p></section><section class="overview-grid" aria-label="原版与五种主视觉造型">${overviewCards.map(design => `<article class="overview-card" data-direction="${design.id}"><h2><span>${design.id.slice(0, 2)} /</span>${design.name}</h2><p>${design.tag}</p>${design.art}<footer><a href="05-art-directions.html#${design.id}">套入首页看效果 ↗</a><a href="${design.vector}" download>下载矢量图 ↓</a></footer></article>`).join('')}</section></main><footer class="overview-footer"><span>所有图形均为 SVG（可无损缩放的矢量图形），无外部图片依赖。只新增候选方案，未替换原版或线上首页。</span><a href="05-art-directions.html">选择一款继续细化 →</a></footer></div></body></html>`
