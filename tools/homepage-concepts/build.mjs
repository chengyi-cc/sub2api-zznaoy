import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { createHash } from 'node:crypto'
import { concepts } from './variants.mjs'
import { goldArt } from './concepts.mjs'
import { markComparison } from './mark-comparison.mjs'
import { artDirections } from './art-directions.mjs'
import { artOverview, createArtGallery } from './art-gallery.mjs'
import { homeArtworks } from './home-artwork-pack.mjs'
import { pageStyles } from './page-styles.mjs'
import { createPageGallery } from './page-gallery.mjs'
import { buildSelectedHomePreview } from './build-selected.mjs'

const source = new URL('./', import.meta.url)
const output = new URL('../../frontend/public/homepage-concepts/', import.meta.url)
const stylesheets = ['themes.css', 'light-prism.css', 'dev-gold.css', 'responsive.css']
const css = (await Promise.all(stylesheets.map(filename => readFile(new URL(filename, source), 'utf8')))).join('\n')
const pageCss = (await Promise.all(['page-styles.css', 'page-artwork.css'].map(filename => readFile(new URL(filename, source), 'utf8')))).join('\n')
const require = createRequire(new URL('../../frontend/package.json', import.meta.url))
const typescript = require('typescript')
const runtimeSources = await Promise.all(['homeArtworkSelection.ts', 'homeArtworkRenderer.ts', 'homeArtworkInteraction.ts'].map(filename => readFile(new URL(`../../frontend/src/utils/${filename}`, source), 'utf8')))
const selectionRuntime = typescript.transpileModule(runtimeSources.join('\n').replace(/^import .+ from .+\r?\n/gm, '').replace(/^export /gm, ''), { compilerOptions: { target: typescript.ScriptTarget.ES2020, module: typescript.ModuleKind.None } }).outputText
const safeJson = value => JSON.stringify(value).replaceAll('<', '\\u003c')
const browserRuntime = `(() => {\n${selectionRuntime}\nwindow.HC_HOME_ART = { artworks: ${safeJson(homeArtworks)}, pick: pickHomeArtwork, apply: applyRandomHomeArtwork, interact: initializeHomeArtworkInteractions, render: renderHomeArtwork };\n})();\n`
const standaloneRuntime = `<script>\n${browserRuntime}\nwindow.HC_HOME_ART.apply(document, window.HC_HOME_ART.artworks);\nwindow.HC_HOME_ART.interact(document, window.HC_HOME_ART.artworks);\n</script>\n`
const previewData = []
await mkdir(new URL('fragments/', output), { recursive: true })
await mkdir(new URL('../../frontend/src/assets/', source), { recursive: true })
await writeFile(new URL('../../frontend/src/assets/home-artwork.css', source), await readFile(new URL('page-artwork.css', source)))
await writeFile(new URL('../../frontend/src/assets/home-artwork-fallback.svg', source), homeArtworks[0].markup.match(/<svg[\s\S]*?<\/svg>/)[0].replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" '))
await writeFile(new URL('../../frontend/src/assets/home-artworks.json', source), `${JSON.stringify(homeArtworks, null, 2)}\n`)
await writeFile(new URL('random-artwork.js', output), browserRuntime)

for (const concept of [...concepts, ...artDirections, ...pageStyles]) {
  const randomArtwork = pageStyles.some(design => design.id === concept.id)
  const fragment = `<style>\n${randomArtwork ? pageCss : css}\n</style>\n${concept.html}\n`
  const document = `<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="color-scheme" content="${concept.light ? 'light' : 'dark'}">\n<meta name="robots" content="noindex,nofollow">\n<title>${concept.name} · zz.naoy 首页设计</title>\n<style>html,body{margin:0;padding:0}html{scroll-behavior:smooth}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}</style>\n</head>\n<body>\n${fragment}${randomArtwork ? standaloneRuntime : ''}</body>\n</html>\n`
  await writeFile(new URL(`${concept.id}.html`, output), document)
  await writeFile(new URL(`fragments/${concept.id}.html`, output), fragment)
  previewData.push({ id: concept.id, name: concept.name, fragment, document, ...(concept.description ? { presentation: [concept.caption, concept.description, concept.tag] } : {}) })
}

const originalData = previewData.filter(item => concepts.some(concept => concept.id === item.id))
const artData = [...previewData.filter(item => artDirections.some(design => design.id === item.id)), previewData.find(item => item.id === '05-atelier')]
await writeFile(new URL('preview-data.js', output), `window.HOMEPAGE_CONCEPTS = ${JSON.stringify(originalData)};\n`)
await writeFile(new URL('art-directions-data.js', output), `window.HOMEPAGE_CONCEPTS = ${JSON.stringify(artData)};\n`)
await writeFile(new URL('page-styles-data.js', output), `window.HOMEPAGE_CONCEPTS = ${safeJson(previewData.filter(item => pageStyles.some(design => design.id === item.id)))};\n`)
const previewRevision = createHash('sha256').update(browserRuntime).update(pageCss).update(JSON.stringify(pageStyles)).update(await readFile(new URL('preview.js', source))).digest('hex').slice(0, 12)
await writeFile(new URL('page-styles.html', output), createPageGallery(await readFile(new URL('index.html', source), 'utf8'), previewRevision))
await writeFile(new URL('05-art-directions.html', output), createArtGallery(await readFile(new URL('index.html', source), 'utf8')))
await writeFile(new URL('05-art-overview.html', output), artOverview)
await writeFile(new URL('05-mark-comparison.html', output), markComparison)
const goldVector = goldArt.match(/<svg[\s\S]*?<\/svg>/)[0]
  .replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ')
  .replace('aria-hidden="true"', 'role="img" aria-labelledby="hc-gold-vector-title"')
  .replace('<defs>', '<title id="hc-gold-vector-title">金色轨道 · 原版</title><defs>')
await writeFile(new URL('05-mark.svg', output), `${goldVector}\n`)

for (const design of artDirections) {
  const vector = design.art.match(/<svg[\s\S]*?<\/svg>/)[0]
    .replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ')
    .replace('aria-hidden="true"', `role="img" aria-labelledby="hc-${design.id}-title"`)
    .replace('<defs>', `<title id="hc-${design.id}-title">${design.name}</title><defs>`)
  await writeFile(new URL(`${design.id}-mark.svg`, output), `${vector}\n`)
}

for (const filename of ['index.html', 'preview.css', 'preview.js', 'art-gallery.css', 'page-gallery.css']) {
  await writeFile(new URL(filename, output), await readFile(new URL(filename, source)))
}

await buildSelectedHomePreview()
console.log(`Generated ${concepts.length} original pages, ${artDirections.length} artwork directions, ${pageStyles.length} new page styles, the selected dual-theme preview, and ${homeArtworks.length} random artworks in ${fileURLToPath(output)}`)
