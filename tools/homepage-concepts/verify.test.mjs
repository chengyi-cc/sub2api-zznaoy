import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import { Script } from 'node:vm'
import { concepts } from './variants.mjs'
import { goldThreads } from './atelier-mark.mjs'
import { goldArt, originalGoldArt } from './concepts.mjs'
import { artDirections } from './art-directions.mjs'
import { pageStyles } from './page-styles.mjs'
import { homeArtworks } from './home-artwork-pack.mjs'
import { transparentGoldArt } from './transparent-artwork.mjs'
import { frontSections } from './art-depth.mjs'
import { project, rotate, curve } from './art-geometry.mjs'

const require = createRequire(new URL('../../frontend/package.json', import.meta.url))
const { JSDOM, VirtualConsole } = require('jsdom')
const postcss = require('postcss')
const output = new URL('../../frontend/public/homepage-concepts/', import.meta.url)
const readOutput = filename => readFile(new URL(filename, output), 'utf8')
const galleryHtml = await readOutput('index.html')
const galleryCode = await readOutput('preview.js')
const galleryData = await readOutput('preview-data.js')

test('selected native preview renders both themes and translations without replacing the sculpture', async () => {
  const errors = []
  const virtualConsole = new VirtualConsole().on('jsdomError', error => errors.push(error)).on('error', error => errors.push(error))
  const dom = new JSDOM(await readOutput('selected-home.html'), { url: 'https://preview.example/', runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole })
  const { window } = dom
  window.matchMedia = query => ({ matches: query.includes('reduced-motion'), addEventListener() {}, removeEventListener() {} })
  try {
    window.eval(await readOutput('selected/selected-home.js'))
    await new Promise(resolve => setImmediate(resolve))
    const document = window.document
    const root = document.querySelector('[data-home-design]')
    assert.equal(root.dataset.homeDesign, '12-command')
    assert.equal(document.querySelector('h1').textContent, '你的应用，下一层智能。')
    assert.equal(document.querySelector('[data-testid="home-primary-action"]').getAttribute('href'), '/login')
    const slot = document.querySelector('[data-hc-artwork]')
    assert.equal(slot.dataset.hcInteractive, 'true')
    const identifier = slot.dataset.hcCurrentArt
    slot.click()
    assert.equal(Number(slot.dataset.hcYaw), 32)
    const svg = slot.querySelector('svg')
    document.querySelector('[data-testid="home-theme-toggle"]').click()
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(root.dataset.homeDesign, '15-silver')
    assert.equal(document.querySelector('h1').textContent, '连接所想。专注所长。')
    assert.equal(slot.querySelector('svg'), svg)
    assert.equal(slot.dataset.hcCurrentArt, identifier)
    assert.equal(Number(slot.dataset.hcYaw), 32)
    document.querySelector('.preview-language').click()
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(document.querySelector('h1').textContent, 'Connect ideas.Find your focus.')
    assert.equal(document.documentElement.lang, 'en')
    assert.equal(document.querySelectorAll('h1').length, 1)
    assert.ok(!document.body.textContent.includes('home.designed.'))
    assert.equal(errors.length, 0)
  } finally {
    window.close()
  }
})

test('selected themes keep legible text, local fonts and frameless shared artwork styles', async () => {
  const css = await readFile(new URL('../../frontend/src/components/home/designed-home.css', import.meta.url), 'utf8')
  const stylesheet = postcss.parse(css)
  const settings = {}
  for (const selector of ['.designed-home', '.designed-home.hf-silver']) {
    const rule = stylesheet.nodes.find(node => node.type === 'rule' && node.selector === selector)
    rule.walkDecls(declaration => { settings[declaration.prop] = declaration.value })
    const luminance = color => {
      const channels = color.slice(1).match(/../g).map(channel => parseInt(channel, 16) / 255).map(channel => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4)
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
    }
    const contrast = (first, second) => (Math.max(luminance(first), luminance(second)) + 0.05) / (Math.min(luminance(first), luminance(second)) + 0.05)
    for (const background of ['--dh-bg', '--dh-panel']) {
      assert.ok(contrast(settings['--dh-muted'], settings[background]) >= 4.5, selector)
      assert.ok(contrast(settings['--dh-ink'], settings[background]) >= 4.5, selector)
    }
    assert.ok(contrast(settings['--dh-accent'], settings['--dh-on-accent']) >= 4.5, selector)
  }
  assert.doesNotMatch(css, /@import|fonts\.google|@font-face/)
  assert.match(css, /max-width: 1248px/)
  assert.match(css, /max-width: 800px/)
  assert.match(css, /max-width: 440px/)
  const stage = stylesheet.nodes.find(node => node.type === 'rule' && node.selector === '.designed-home .dh-figure')
  assert.equal(stage.nodes.find(node => node.prop === 'background').value, 'transparent')
  assert.equal(stage.nodes.find(node => node.prop === 'border').value, '0')
  assert.equal(await readFile(new URL('../../frontend/src/assets/home-artwork.css', import.meta.url), 'utf8'), await readFile(new URL('page-artwork.css', import.meta.url), 'utf8'))
})

test('five additional art directions have distinct identities and bounded finite geometry', () => {
  assert.equal(artDirections.length, 5)
  assert.equal(new Set(artDirections.map(design => design.id)).size, 5)
  assert.equal(new Set(artDirections.map(design => design.art)).size, 5)
  for (const design of artDirections) {
    assert.ok(design.nodes.length >= 20)
    for (const node of design.nodes) {
      for (const point of node.points) {
        assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z))
        assert.ok(point.x > 35 && point.x < 565, design.id)
        assert.ok(point.y > 30 && point.y < 560, design.id)
      }
    }
    assert.ok(!design.html.includes(originalGoldArt))
    assert.ok(!design.art.includes('hc-gold-thread'))
    const document = new JSDOM(design.html).window.document
    assert.equal(document.querySelectorAll('[data-art-style]').length, 1)
    assert.equal(document.querySelector('[data-art-style]').getAttribute('data-art-style'), design.id)
    assert.equal(document.querySelector('h1').textContent, '连接智能，自有章法。')
  }
})

test('new overview presents the original and five unique drawings with usable local links', async () => {
  const overview = new JSDOM(await readOutput('05-art-overview.html')).window.document
  assert.equal(overview.querySelectorAll('.overview-card').length, 6)
  assert.equal(overview.querySelector('.overview-card').getAttribute('data-direction'), '05-atelier')
  const identifiers = Array.from(overview.querySelectorAll('[id]'), element => element.id)
  assert.equal(new Set(identifiers).size, identifiers.length)
  for (const anchor of overview.querySelectorAll('a')) {
    const href = anchor.getAttribute('href')
    assert.ok((await readOutput(href.split('#')[0])).length > 0)
  }
  for (const design of artDirections) {
    const vector = new JSDOM(await readOutput(`${design.id}-mark.svg`), { contentType: 'image/svg+xml' }).window.document
    assert.equal(vector.documentElement.namespaceURI, 'http://www.w3.org/2000/svg')
    assert.equal(vector.documentElement.getAttribute('aria-labelledby'), vector.querySelector('title').id)
    assert.equal(vector.querySelector('title').textContent, design.name)
    assert.equal(vector.querySelectorAll('script,image,foreignObject').length, 0)
    for (const element of vector.querySelectorAll('[stroke],[fill]')) {
      for (const attribute of ['stroke', 'fill']) {
        const reference = element.getAttribute(attribute)?.match(/^url\(#(.+)\)$/)
        if (reference) assert.ok(vector.getElementById(reference[1]))
      }
    }
  }
})

test('new gallery switches five additions plus the original and exports the selected artwork', async () => {
  const html = await readOutput('05-art-directions.html')
  const data = await readOutput('art-directions-data.js')
  const { dom, document, errors } = setupGallery('', html, data)
  assert.equal(document.querySelectorAll('[data-concept]').length, 6)
  assert.equal(document.querySelector('[data-concept][aria-pressed="true"]').dataset.concept, '06-armillary')
  assert.ok(document.querySelector('a[href="05-art-overview.html"]'))
  for (const design of artDirections) {
    document.querySelector(`[data-concept="${design.id}"]`).click()
    assert.ok(document.getElementById('design-frame').srcdoc.includes(design.art))
    assert.ok(document.getElementById('export-content').value.includes(design.art))
    assert.equal(document.getElementById('concept-description').textContent, design.description)
  }
  document.getElementById('compare-toggle').click()
  assert.equal(document.querySelectorAll('.comparison-card').length, 6)
  document.querySelector('[data-concept="05-atelier"]').click()
  assert.ok(document.getElementById('export-content').value.includes(originalGoldArt))
  document.querySelector('[data-viewport="mobile"]').click()
  assert.equal(document.getElementById('design-frame').style.width, '390px')
  assert.equal(document.getElementById('compare-toggle').textContent, '全部对比')
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('archived ribbon experiment remains valid for comparison only', () => {
  assert.equal(goldThreads.length, 18)
  for (const thread of goldThreads) {
    assert.equal(thread.points.length, 48)
    assert.match(thread.path, /^M/)
    assert.match(thread.path, /Z$/)
    assert.doesNotMatch(thread.path, /NaN|Infinity/)
    assert.equal(thread.path.split('C').length - 1, 48)
    assert.ok(thread.width <= 0.85)
    assert.ok(thread.opacity > 0 && thread.opacity < 1)
    for (const point of thread.points) {
      assert.ok(point.x > 45 && point.x < 555)
      assert.ok(point.y > 37 && point.y < 547)
      assert.ok(Math.hypot(point.x - 300, point.y - 292) > 120)
    }
  }
})

test('comparison records both variants and vector download uses the restored original', async () => {
  const comparison = new JSDOM(await readOutput('05-mark-comparison.html')).window.document
  assert.equal(comparison.querySelectorAll('.refinement-card').length, 2)
  assert.equal(comparison.querySelector('.refinement-card').querySelectorAll('ellipse').length, 24)
  assert.equal(comparison.querySelectorAll('.hc-gold-thread').length, 18)
  assert.equal(comparison.querySelectorAll('script,iframe').length, 0)
  const identifiers = Array.from(comparison.querySelectorAll('[id]'), element => element.id)
  assert.equal(new Set(identifiers).size, identifiers.length)
  assert.equal(comparison.querySelector('a[download]').getAttribute('href'), '05-mark.svg')
  const vector = new JSDOM(await readOutput('05-mark.svg'), { contentType: 'image/svg+xml' }).window.document
  assert.equal(vector.documentElement.getAttribute('viewBox'), '0 0 600 590')
  assert.equal(vector.querySelectorAll('.hc-gold-thread').length, 0)
  assert.equal(vector.querySelectorAll('ellipse').length, 24)
  assert.equal(vector.documentElement.getAttribute('aria-labelledby'), vector.querySelector('title').id)
  for (const element of vector.querySelectorAll('[stroke],[fill]')) {
    for (const attribute of ['stroke', 'fill']) {
      const reference = element.getAttribute(attribute)?.match(/^url\(#(.+)\)$/)
      if (reference) assert.ok(vector.getElementById(reference[1]))
    }
  }
  const previousPage = new JSDOM(await readOutput('01-orbit.html')).window.document
  assert.equal(previousPage.querySelectorAll('.hc-gold-thread').length, 0)
})

test('05 page, fragment and gallery restore the exact original mark and proportions', async () => {
  assert.equal(goldArt, originalGoldArt)
  for (const filename of ['05-atelier.html', 'fragments/05-atelier.html']) {
    const content = await readOutput(filename)
    assert.ok(content.includes(originalGoldArt))
    const document = new JSDOM(content).window.document
    assert.equal(document.querySelectorAll('.hc-gold-art ellipse').length, 24)
    assert.equal(document.querySelectorAll('.hc-gold-thread').length, 0)
    assert.equal(document.querySelector('.hc-gold-art circle').getAttribute('r'), '52')
    assert.match(content, /\.hc-atelier \.hc-gold-art\{position:relative;width:115%;margin-left:-8%;margin-top:10px\}/)
    assert.match(content, /\.hc-atelier \.hc-gold-art\{width:105%;max-width:480px;/)
  }
  const { dom, document } = setupGallery('#05-atelier')
  assert.ok(document.getElementById('design-frame').srcdoc.includes(originalGoldArt))
  assert.ok(document.getElementById('export-content').value.includes(originalGoldArt))
  assert.match(document.getElementById('concept-description').textContent, /已恢复原版/)
  dom.window.close()
})

for (const concept of [...concepts, ...artDirections]) {
  test(`${concept.id}: self-contained, script-free, scoped and navigable`, async () => {
    const fragment = await readOutput(`fragments/${concept.id}.html`)
    const document = new JSDOM(fragment).window.document
    assert.equal(document.querySelectorAll('.hc-site').length, 1)
    assert.equal(document.querySelectorAll('h1').length, 1)
    assert.equal(document.querySelectorAll('main').length, 1)
    assert.equal(document.querySelectorAll('script,link,iframe,img[src],video,audio').length, 0)
    assert.equal(document.querySelectorAll('details').length, 3)
    assert.ok(document.querySelector('.hc-model-note'))
    const ids = Array.from(document.querySelectorAll('[id]'), element => element.id)
    assert.equal(new Set(ids).size, ids.length)
    for (const anchor of document.querySelectorAll('a')) {
      const href = anchor.getAttribute('href')
      if (href.startsWith('#')) assert.ok(document.getElementById(href.slice(1)), href)
      else {
        assert.ok(['/dashboard', '/keys', '/key-usage'].includes(href), href)
        assert.equal(anchor.target, '_top')
      }
    }
    for (const element of document.querySelectorAll('*')) {
      for (const attribute of element.attributes) assert.ok(!/^on/i.test(attribute.name), attribute.name)
    }
    const stylesheet = postcss.parse(document.querySelector('style').textContent)
    stylesheet.walkRules(rule => {
      if (rule.parent.type === 'atrule' && rule.parent.name === 'keyframes') return
      for (const selector of rule.selectors) assert.match(selector, /^\.hc-/)
    })
    const page = await readOutput(`${concept.id}.html`)
    assert.ok(page.includes(fragment))
    assert.match(page, /<meta name="viewport"/)
    assert.match(fragment, /prefers-reduced-motion:reduce/)
    assert.doesNotMatch(fragment, /https?:\/\//)
  })
}

function setupGallery(hash = '', html = galleryHtml, data = galleryData, runtime = '') {
  const errors = []
  const virtualConsole = new VirtualConsole()
  virtualConsole.on('jsdomError', error => errors.push(error))
  const dom = new JSDOM(html, { url: `http://localhost/homepage-concepts/index.html${hash}`, runScripts: 'outside-only', virtualConsole })
  const { window } = dom
  let blobCount = 0
  window.URL.createObjectURL = () => `blob:preview-${++blobCount}`
  window.URL.revokeObjectURL = () => {}
  window.ResizeObserver = class { observe() {} }
  window.HTMLDialogElement.prototype.showModal = function () { this.open = true }
  window.HTMLDialogElement.prototype.close = function () { this.open = false }
  window.eval(data)
  window.eval(runtime)
  window.eval(galleryCode)
  return { dom, window, document: window.document, errors }
}

test('gallery scripts parse and the default selection renders', () => {
  new Script(galleryCode)
  new Script(galleryData)
  const { dom, document, errors } = setupGallery()
  assert.equal(document.querySelectorAll('[data-concept][aria-pressed="true"]').length, 1)
  assert.equal(document.querySelector('[data-concept][aria-pressed="true"]').dataset.concept, '01-orbit')
  assert.match(document.getElementById('design-frame').srcdoc, /hc-orbit/)
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('all five choices, comparison view and viewport controls work', () => {
  const { dom, document, errors } = setupGallery()
  for (const concept of concepts) {
    document.querySelector(`[data-concept="${concept.id}"]`).click()
    assert.match(document.getElementById('concept-title').textContent, new RegExp(concept.name))
    assert.match(document.getElementById('export-label').textContent, new RegExp(concept.name))
    assert.equal(document.getElementById('mark-comparison-link').hidden, concept.id !== '05-atelier')
  }
  document.querySelector('[data-viewport="mobile"]').click()
  assert.equal(document.getElementById('design-frame').style.width, '390px')
  assert.ok(document.getElementById('preview-shell').classList.contains('is-mobile'))
  document.getElementById('compare-toggle').click()
  assert.equal(document.querySelectorAll('.comparison-card').length, 5)
  assert.equal(document.getElementById('preview-stage').hidden, true)
  document.querySelector('.comparison-caption button').click()
  assert.equal(document.getElementById('preview-stage').hidden, false)
  assert.equal(document.getElementById('comparison').hidden, true)
  document.querySelector('[data-viewport="desktop"]').click()
  assert.equal(document.getElementById('design-frame').style.width, '1440px')
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('brand customization escapes markup and updates preview, comparison and export', () => {
  const { dom, document, window, errors } = setupGallery()
  const name = '<img src=x onerror="alert(1)">&$&'
  document.getElementById('brand-name').value = name
  document.getElementById('brand-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }))
  const exported = new JSDOM(document.getElementById('export-content').value).window.document
  assert.equal(exported.querySelectorAll('img,script').length, 0)
  assert.ok(exported.querySelector('.hc-brand').textContent.includes(name))
  assert.equal(exported.querySelector('.hc-brand').getAttribute('aria-label'), `${name} 首页`)
  assert.match(document.getElementById('design-frame').srcdoc, /&lt;img/)
  document.getElementById('compare-toggle').click()
  assert.match(document.querySelector('.comparison-visual iframe').srcdoc, /&lt;img/)
  document.getElementById('export-button').click()
  assert.equal(document.getElementById('export-dialog').open, true)
  document.getElementById('close-export').click()
  assert.equal(document.getElementById('export-dialog').open, false)
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('clipboard failure selects the fragment for manual copying', async () => {
  const { dom, document, errors } = setupGallery('#05-atelier')
  assert.match(document.getElementById('concept-title').textContent, /鎏金典藏/)
  document.getElementById('copy-fragment').click()
  await Promise.resolve()
  const textarea = document.getElementById('export-content')
  assert.equal(textarea.selectionStart, 0)
  assert.equal(textarea.selectionEnd, textarea.value.length)
  assert.match(document.getElementById('status-message').textContent, /Ctrl\+C/)
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('central emblems share the sculpture camera and sit between back and front geometry', () => {
  for (const design of artDirections) {
    const dom = new JSDOM(design.art)
    const document = dom.window.document
    const layers = [...document.querySelectorAll('[data-depth-layer]')]
    assert.deepEqual(layers.map(layer => layer.dataset.depthLayer), ['back', 'core', 'front'])
    assert.equal(layers[0].children.length, design.nodes.length)
    assert.equal(layers[1].dataset.worldCenter, '0 0 0')
    const projectedGlyph = [[-14, 15], [-14, -15], [-7, -15], [8, 15], [14, 15], [14, -15]].map(([horizontal, vertical]) => project({ x: horizontal, y: vertical, z: 2.2 }, design.angles))
    const expected = new JSDOM(`<svg>${curve(projectedGlyph, { closed: false, smooth: false, width: 1.65, opacity: 1 }).markup}</svg>`)
    assert.equal(layers[1].lastElementChild.getAttribute('d'), expected.window.document.querySelector('path').getAttribute('d'))
    const clipping = layers[2].getAttribute('clip-path').match(/^url\(#(.+)\)$/)
    assert.ok(document.getElementById(clipping[1]))
    assert.doesNotMatch(design.art, /NaN|Infinity/)
    expected.window.close()
    dom.window.close()
  }
})

test('foreground clipping follows the tilted emblem plane instead of a fixed screen layer', () => {
  const angles = [0.35, -0.42, 0.1]
  const normal = rotate({ x: 0, y: 0, z: 1 }, angles)
  const behind = project({ x: -40, y: 0, z: -20 }, angles)
  const ahead = project({ x: 40, y: 0, z: 24 }, angles)
  const node = curve([behind, ahead], { closed: false, smooth: false })
  const sections = frontSections(node, normal)
  assert.equal(sections.length, 1)
  assert.equal(sections[0].length, 2)
  const middle = project({ x: 0, y: 0, z: 2 }, angles)
  for (const axis of ['x', 'y', 'z']) assert.ok(Math.abs(sections[0][0][axis] - middle[axis]) < 0.00001)
  assert.deepEqual(sections[0][1], ahead)
  assert.equal(frontSections({ points: [behind] }, normal).length, 0)
  assert.equal(frontSections({ points: [ahead] }, normal).length, 1)
  const polygon = curve([behind, ahead, project({ x: 0, y: 40, z: 24 }, angles)], { smooth: false, fill: '#ffffff09' })
  assert.equal(frontSections(polygon, normal)[0].length, 4)
})

for (const design of pageStyles) {
  test(`${design.id}: new layout, script-free fragment, complete sections and scoped responsive styling`, async () => {
    const fragment = await readOutput(`fragments/${design.id}.html`)
    const dom = new JSDOM(fragment)
    const document = dom.window.document
    assert.equal(document.querySelectorAll('.hf-site').length, 1)
    assert.equal(document.querySelectorAll('.hc-site').length, 0)
    assert.equal(document.querySelectorAll('h1').length, 1)
    assert.equal(document.querySelectorAll('details').length, 3)
    assert.equal(document.querySelectorAll('[data-hc-artwork="random"]').length, 1)
    assert.equal(document.querySelectorAll('script,iframe,link,img').length, 0)
    assert.ok(fragment.includes(transparentGoldArt))
    assert.match(document.querySelector('.hf-models small').textContent, /实际开放模型与价格/)
    const identifiers = [...document.querySelectorAll('[id]')].map(element => element.id)
    assert.equal(new Set(identifiers).size, identifiers.length)
    for (const anchor of document.querySelectorAll('a')) {
      const href = anchor.getAttribute('href')
      if (href.startsWith('#')) assert.ok(document.getElementById(href.slice(1)), href)
      else {
        assert.ok(['/dashboard', '/keys', '/key-usage'].includes(href))
        assert.equal(anchor.getAttribute('target'), '_top')
      }
    }
    const css = document.querySelector('style').textContent
    const stylesheet = postcss.parse(css)
    stylesheet.walkRules(rule => {
      if (rule.parent.type === 'atrule' && rule.parent.name === 'keyframes') return
      assert.match(rule.selector, /^\.hf-/, rule.selector)
    })
    assert.match(css, /@media\(max-width:760px\)/)
    assert.match(css, /prefers-reduced-motion:reduce/)
    assert.match(css, /focus-visible/)
    assert.doesNotMatch(css, /@import|url\(https?:/)
    const standalone = await readOutput(`${design.id}.html`)
    assert.ok(standalone.includes(fragment))
    const exported = new JSDOM(standalone, { runScripts: 'dangerously' })
    assert.ok(homeArtworks.some(artwork => artwork.id === exported.window.document.querySelector('[data-hc-artwork]').dataset.hcCurrentArt))
    const interactiveSlot = exported.window.document.querySelector('[data-hc-artwork]')
    assert.equal(interactiveSlot.dataset.hcInteractive, 'true')
    exported.window.document.querySelector('.hf-motion-pause').checked = true
    interactiveSlot.click()
    assert.equal(Number(interactiveSlot.dataset.hcYaw), 32)
    assert.ok(interactiveSlot.querySelector('[data-depth-layer="core"]'))
    exported.window.document.querySelector('[data-hc-art-reset]').click()
    assert.equal(Number(interactiveSlot.dataset.hcYaw), 0)
    assert.equal(exported.window.document.querySelectorAll('script[src]').length, 0)
    assert.equal(exported.window.document.querySelectorAll('[data-hc-artwork] svg').length, 1)
    exported.window.close()
    dom.window.close()
  })
}

test('native and standalone artwork collections match the six accepted sculptures', async () => {
  const native = JSON.parse(await readFile(new URL('../../frontend/src/assets/home-artworks.json', import.meta.url), 'utf8'))
  assert.deepEqual(native, homeArtworks)
  assert.equal(native.length, 6)
  assert.equal(native[0].markup, transparentGoldArt)
  assert.ok(native.every(artwork => !artwork.markup.includes('hc-gold-thread')))
  const dom = new JSDOM('', { runScripts: 'outside-only' })
  dom.window.eval(await readOutput('random-artwork.js'))
  assert.equal(JSON.stringify(dom.window.HC_HOME_ART.artworks), JSON.stringify(homeArtworks))
  dom.window.close()
})

test('transparent artwork removes solid badges and stage fills without changing sculpture geometry', () => {
  const original = new JSDOM(originalGoldArt)
  for (const artwork of homeArtworks) {
    const dom = new JSDOM(artwork.markup)
    const document = dom.window.document
    assert.equal(document.querySelectorAll('[data-art-surface="transparent"]').length, 1)
    assert.equal(document.querySelectorAll('svg>rect,svg>circle[r="247"],image').length, 0)
    assert.equal(document.querySelectorAll('stop[style*="--hf-metal-"]').length, artwork.id === '05-atelier' ? 4 : 5)
    if (artwork.id === '05-atelier') {
      const contours = root => [...root.querySelectorAll('ellipse')].map(element => element.outerHTML)
      assert.deepEqual(contours(document), contours(original.window.document))
      assert.equal(document.querySelector('circle').getAttribute('fill'), 'none')
      assert.equal(document.querySelector('circle').getAttribute('r'), '52')
      assert.equal(document.querySelector('path').getAttribute('d'), original.window.document.querySelector('path').getAttribute('d'))
    } else {
      const back = document.querySelector('[data-depth-layer="back"]')
      const maskId = back.getAttribute('mask').match(/^url\(#(.+)\)$/)[1]
      const mask = document.getElementById(maskId)
      assert.equal(mask.localName, 'mask')
      assert.equal(mask.getAttribute('maskUnits'), 'userSpaceOnUse')
      assert.equal(mask.querySelectorAll('path[fill="black"]').length, 2)
      assert.equal(mask.querySelector('rect').getAttribute('fill'), 'white')
      const core = document.querySelector('[data-depth-layer="core"]')
      assert.ok([...core.querySelectorAll('path')].every(path => path.getAttribute('fill') === 'none'))
      const legacy = new JSDOM(artDirections.find(design => design.id === artwork.id).art)
      assert.equal(back.innerHTML, legacy.window.document.querySelector('[data-depth-layer="back"]').innerHTML)
      const paths = root => [...root.querySelectorAll('[data-depth-layer="core"] path')].map(path => path.getAttribute('d'))
      assert.deepEqual(paths(document), paths(legacy.window.document))
      assert.equal(document.querySelector('[data-depth-layer="front"]').outerHTML, legacy.window.document.querySelector('[data-depth-layer="front"]').outerHTML)
      legacy.window.close()
    }
    dom.window.close()
  }
  original.window.close()
})

test('all five layouts remain frameless at every breakpoint and expose a keyboard-accessible motion control', async () => {
  for (const design of pageStyles) {
    const dom = new JSDOM(await readOutput(`fragments/${design.id}.html`))
    const document = dom.window.document
    const css = postcss.parse(document.querySelector('style').textContent)
    css.walkRules(rule => {
      if (!/^\.hf-\S+ \.hf-(?:stage|(?:editorial|cobalt|silver)-figure|art)$/.test(rule.selector)) return
      rule.walkDecls(declaration => {
        if (['background', 'background-color'].includes(declaration.prop)) assert.equal(declaration.value, 'transparent', rule.selector)
        if (['border', 'border-radius'].includes(declaration.prop)) assert.equal(declaration.value, '0', rule.selector)
        if (declaration.prop === 'box-shadow') assert.equal(declaration.value, 'none', rule.selector)
      })
    })
    assert.equal(dom.window.getComputedStyle(document.querySelector('.hf-stage')).backgroundColor, 'rgba(0, 0, 0, 0)')
    assert.equal(document.querySelectorAll('.hf-cobalt-stamp').length, 0)
    const checkbox = document.getElementById('hf-motion-pause')
    const label = document.querySelector('.hf-motion-control')
    assert.equal(checkbox.type, 'checkbox')
    assert.equal(checkbox.getAttribute('aria-label'), '暂停主视觉动效')
    assert.equal(label.htmlFor, checkbox.id)
    label.click()
    assert.equal(checkbox.checked, true)
    label.click()
    assert.equal(checkbox.checked, false)
    assert.match(css.toString(), /:checked~\.hf-art svg\{animation-play-state:paused\}/)
    assert.match(css.toString(), /:focus-visible~\.hf-stage-bottom \.hf-motion-control/)
    const keyframes = []
    css.walkAtRules('keyframes', animation => {
      keyframes.push(animation.params)
      animation.walkDecls(declaration => assert.equal(declaration.prop, 'transform'))
    })
    assert.deepEqual(keyframes, [])
    assert.match(css.toString(), /prefers-reduced-motion:reduce\)\{\.hf-site \.hf-art \.hc-gold-art,\.hf-site \.hf-art svg\{animation:none;transform:none\}/)
    assert.match(css.toString(), /max-width:760px\)\{\.hf-site \.hf-motion-control\{min-height:44px/)
    dom.window.close()
  }
})

test('new style gallery keeps the artwork fixed for comparison and rerolls without changing the layout', async () => {
  const html = await readOutput('page-styles.html')
  const data = await readOutput('page-styles-data.js')
  const runtime = await readOutput('random-artwork.js')
  const { dom, document, window, errors } = setupGallery('', html, data, runtime)
  assert.match(document.querySelector('script[src^="random-artwork.js"]').nextElementSibling.getAttribute('src'), /^preview\.js\?v=[a-f0-9]{12}$/)
  assert.equal(document.querySelectorAll('[data-concept]').length, 5)
  assert.match(document.querySelector('.dialog-description').textContent, /先部署本次更新后的前端/)
  const getPreview = () => new JSDOM(document.getElementById('design-frame').srcdoc)
  let preview = getPreview()
  const original = preview.window.document.querySelector('[data-hc-current-art]').dataset.hcCurrentArt
  assert.equal(preview.window.document.querySelectorAll('script').length, 0)
  preview.window.close()
  for (const design of pageStyles) {
    document.querySelector(`[data-concept="${design.id}"]`).click()
    preview = getPreview()
    assert.equal(preview.window.document.querySelector('[data-hc-current-art]').dataset.hcCurrentArt, original)
    assert.match(document.getElementById('concept-title').textContent, new RegExp(design.name))
    assert.doesNotMatch(document.getElementById('export-content').value, /data-hc-current-art|<script/)
    preview.window.close()
  }
  document.getElementById('compare-toggle').click()
  for (const frame of document.querySelectorAll('.comparison-visual iframe')) {
    const comparison = new JSDOM(frame.srcdoc)
    assert.equal(comparison.window.document.querySelector('[data-hc-current-art]').dataset.hcCurrentArt, original)
    assert.equal(frame.getAttribute('sandbox'), '')
    comparison.window.close()
  }
  document.getElementById('reroll-artwork').click()
  preview = getPreview()
  assert.notEqual(preview.window.document.querySelector('[data-hc-current-art]').dataset.hcCurrentArt, original)
  assert.equal(preview.window.document.querySelectorAll('.hf-silver').length, 1)
  preview.window.close()
  document.getElementById('brand-name').value = '</script><img src=x onerror="alert(1)">&$&'
  document.getElementById('brand-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }))
  preview = getPreview()
  assert.equal(preview.window.document.querySelectorAll('script,img').length, 0)
  assert.ok(preview.window.document.querySelector('.hf-brand').textContent.includes('</script><img'))
  preview.window.close()
  document.querySelector('[data-viewport="mobile"]').click()
  assert.equal(document.getElementById('design-frame').getAttribute('sandbox'), 'allow-same-origin')
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('gallery attaches real rotation controls from the parent without allowing iframe scripts', async () => {
  const { dom, window, document, errors } = setupGallery('', await readOutput('page-styles.html'), await readOutput('page-styles-data.js'), await readOutput('random-artwork.js'))
  const frame = document.getElementById('design-frame')
  const displayed = new JSDOM(frame.srcdoc)
  frame.contentDocument.documentElement.innerHTML = displayed.window.document.documentElement.innerHTML
  displayed.window.close()
  frame.dispatchEvent(new window.Event('load'))
  const slot = frame.contentDocument.querySelector('[data-hc-artwork]')
  assert.equal(frame.getAttribute('sandbox'), 'allow-same-origin')
  assert.equal(frame.contentDocument.querySelectorAll('script').length, 0)
  assert.equal(slot.dataset.hcInteractive, 'true')
  frame.contentDocument.querySelector('.hf-motion-pause').checked = true
  slot.click()
  assert.equal(Number(slot.dataset.hcYaw), 32)
  frame.contentDocument.querySelector('[data-hc-art-reset]').click()
  assert.equal(Number(slot.dataset.hcYaw), 0)
  document.getElementById('reroll-artwork').click()
  assert.equal(slot.dataset.hcInteractive, undefined)
  assert.equal(errors.length, 0)
  dom.window.close()
})

test('sculptures have no pointer focus frame, floating intro or forced low frame rate', async () => {
  const stylesheet = postcss.parse(await readFile(new URL('page-artwork.css', import.meta.url), 'utf8'))
  const interactive = []
  stylesheet.walkRules(rule => {
    if (rule.selector === '.hf-site .hf-art[data-hc-interactive]' || rule.selector === '.hf-site .hf-art[data-hc-interactive]:focus-visible') {
      interactive.push(rule.selector)
      assert.equal(rule.nodes.find(declaration => declaration.prop === 'outline')?.value, 'none')
      assert.ok(!rule.nodes.some(declaration => declaration.prop === 'border-radius' || declaration.prop === 'box-shadow' || declaration.prop === 'border'))
    }
  })
  assert.equal(interactive.length, 2)
  const css = stylesheet.toString()
  assert.match(css, /:focus-visible~\.hf-interaction-hint\{color:var\(--hf-accent\);text-decoration:underline/)
  assert.doesNotMatch(css, /hf-art-float|hf-art-sway|hf-art-arrival|data-hc-auto-phase="intro"/)
  assert.match(css, /\.hf-art \.hc-gold-art,\.hf-site \.hf-art svg\{animation:none;transform:none\}/)
  const runtime = await readOutput('random-artwork.js')
  assert.match(runtime, /if \(dirty\)\s+draw\(\)/)
  assert.match(runtime, /createHomeArtworkRenderer\(svg, model\)/)
  assert.doesNotMatch(runtime, /introComplete|lastDraw|autoSpinning|hf-art-arrival/)
  assert.match(runtime, /hcAutoPhase = 'spinning'/)
  assert.doesNotMatch(runtime, /65 \* radians/)
})
