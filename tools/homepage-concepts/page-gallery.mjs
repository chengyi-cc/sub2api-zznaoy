import { pageStyles } from './page-styles.mjs'

export function createPageGallery(template, revision = '') {
  const controls = pageStyles.map((design, index) => `<button type="button" data-concept="${design.id}" aria-pressed="${index === 0}"><span class="concept-swatch swatch-${design.id}"></span><span><b>${design.id.slice(0, 2)} ${design.name}</b><small>${design.tag}</small></span><span class="concept-arrow">↗</span></button>`).join('\n')
  return template
    .replace('zz.naoy · 五种首页，五种气质', '全新页面风格 · 保留六款主视觉')
    .replace('src="preview-data.js"', 'src="page-styles-data.js"')
    .replace('<script defer src="preview.js">', '<script defer src="random-artwork.js"></script>\n<script defer src="preview.js">')
    .replace('</head>', '<link rel="stylesheet" href="page-gallery.css">\n</head>')
    .replace('<body>', '<body class="page-collection">')
    .replace('首页设计室<small>zz.naoy / DESIGN EXPLORATIONS', '全新页面风格<small>NEW LAYOUTS / SIX SIGNATURE FORMS')
    .replace('五种首页，<br>五种气质。', '主视觉确定。<br>页面重新设计。')
    .replace('不是换一个颜色，<br>而是换一种让人记住你的方式。', '六款雕塑全部保留，打开随机一款。<br>这次选择排版、配色与整页气质。')
    .replace(/<nav class="concept-nav"[\s\S]*?<\/nav>/, `<nav class="concept-nav" aria-label="选择全新页面风格">${controls}</nav>`)
    .replace('<a id="open-page"', '<button id="reroll-artwork" type="button">换一个主视觉 ↻</button><a id="open-page"')
    .replace('<button id="compare-toggle"', '<a href="selected-home.html">12 + 15 定稿预览 ↗</a><button id="compare-toggle"')
    .replace('href="01-orbit.html"', 'href="11-cloud.html"')
    .replace('<a href="page-styles.html">新五套页面风格 ↗</a>', '<a href="index.html">原五套页面 ↗</a>')
    .replace('主视觉新五款 ↗', '主视觉收藏 ↗')
    .replace('设计预览，不影响现有站点', '透明主视觉 · 可拖拽旋转 · 仅预览')
    .replace('页面可滚动、常见问题可展开。预览里的业务按钮只提示目标，不会登录或修改账户。', '单款预览支持点击转动、按住拖拽旋转，松手有短暂惯性；“复位”恢复初始视角。对比时固定同一款雕塑，点击“换一个主视觉”再次随机。业务按钮只提示目标。')
    .replace(/<noscript>[\s\S]*?<\/noscript>/, `<noscript><p class="noscript-note">脚本未启用，以下页面仍可查看；雕塑保留原版作为静态后备。</p><p>${pageStyles.map(design => `<a href="${design.id}.html">${design.id.slice(0, 2)} ${design.name}</a>`).join(' · ')}</p></noscript>`)
    .replace('无需重新编译；保存前请备份旧内容。', '随机主视觉需要先部署本次更新后的前端程序；旧程序仍能显示页面，但只显示后备图形。保存前请备份旧内容。')
    .replace(/(src|href)="([^\"]+\.(?:js|css))"/g, (match, attribute, filename) => revision ? `${attribute}="${filename}?v=${revision}"` : match)
}
