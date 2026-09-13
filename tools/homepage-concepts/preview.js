(() => {
  const concepts = window.HOMEPAGE_CONCEPTS
  const descriptions = {
    '01-orbit': ['深空科技 / ORBIT', '石墨黑与鼠尾草绿，搭配轨道主视觉。科技感足够鲜明，也适合长期作为品牌首页。', '科技品牌 · 首选推荐'],
    '02-porcelain': ['极简产品 / PORCELAIN', '大面积留白、克莱因蓝与几何连接体。明亮、克制，更接近成熟国际化产品的气质。', '简洁高级 · 清爽耐看'],
    '03-prism': ['未来光影 / PRISM', '居中大标题、通透棱镜与紫色光场。更强调第一眼的视觉冲击，适合突出创意与未来感。', '视觉冲击 · 未来感'],
    '04-terminal': ['开发者美学 / TERMINAL', '炭灰底色、荧光绿与配置终端。信息表达直接，适合面向开发者和工具用户的平台。', '技术用户 · 专业直接'],
    '05-atelier': ['精品品牌 / ATELIER', '已恢复原版金色轨道：保留饱满的交织轮廓、24 道椭圆线条和原有比例。丝带实验版未采用。', '原版已恢复 · 交织雕塑'],
  }
  for (const concept of concepts) {
    if (concept.presentation) descriptions[concept.id] = concept.presentation
  }
  const frame = document.getElementById('design-frame')
  const stage = document.getElementById('preview-stage')
  const shell = document.getElementById('preview-shell')
  const comparison = document.getElementById('comparison')
  const dialog = document.getElementById('export-dialog')
  const exportContent = document.getElementById('export-content')
  const status = document.getElementById('status-message')
  const exportStatus = document.createElement('p')
  exportStatus.className = 'export-status'
  exportStatus.setAttribute('role', 'status')
  exportStatus.setAttribute('aria-live', 'polite')
  exportStatus.hidden = true
  dialog.append(exportStatus)
  let selected = concepts.find(concept => concept.id === location.hash.slice(1)) || concepts[0]
  let viewport = 'desktop'
  let comparing = false
  let brand = 'zz.naoy'
  let statusTimer
  let pageObjectUrl
  let selectedArtwork
  let cleanupInteraction

  function previewDocument(content) {
    const runtime = window.HC_HOME_ART
    if (!runtime || !content.includes('data-hc-artwork="random"')) return branded(content)
    const parsed = new DOMParser().parseFromString(branded(content), 'text/html')
    const identifier = runtime.apply(parsed, selectedArtwork ? [selectedArtwork] : runtime.artworks, { force: true, ...(selectedArtwork ? { storage: null } : {}) })
    selectedArtwork = runtime.artworks.find(artwork => artwork.id === identifier)
    parsed.querySelectorAll('script').forEach(script => script.remove())
    return `<!doctype html>\n${parsed.documentElement.outerHTML}`
  }

  function notify(message) {
    clearTimeout(statusTimer)
    status.textContent = message
    status.hidden = dialog.open
    exportStatus.textContent = message
    exportStatus.hidden = !dialog.open
    statusTimer = setTimeout(() => { status.hidden = true; exportStatus.hidden = true }, 6000)
  }

  function branded(content) {
    const safeName = brand.replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character])
    return content.replaceAll('zz.naoy', () => safeName)
  }

  function resizePreview() {
    if (comparing) {
      comparison.querySelectorAll('.comparison-visual').forEach(visual => {
        const scale = visual.clientWidth / 1440
        visual.querySelector('iframe').style.transform = `scale(${scale})`
        visual.style.height = `${920 * scale}px`
      })
      return
    }
    const padding = Number.parseFloat(getComputedStyle(stage).paddingLeft) * 2
    const available = Math.max(240, stage.clientWidth - padding)
    const frameWidth = viewport === 'mobile' ? 390 : 1440
    const border = viewport === 'mobile' ? 10 : 2
    const scale = Math.min(1, (available - border) / frameWidth)
    const visibleHeight = Math.max(420, stage.clientHeight - 4)
    shell.style.width = `${frameWidth * scale + border}px`
    shell.style.height = `${visibleHeight}px`
    frame.style.width = `${frameWidth}px`
    frame.style.height = `${(visibleHeight - border) / scale}px`
    frame.style.transform = `scale(${scale})`
  }

  function renderComparison() {
    comparison.replaceChildren()
    concepts.forEach(concept => {
      const card = document.createElement('article')
      card.className = 'comparison-card'
      const visual = document.createElement('div')
      visual.className = 'comparison-visual'
      const thumbnail = document.createElement('iframe')
      thumbnail.title = `${concept.name}桌面首屏缩略预览`
      thumbnail.setAttribute('sandbox', '')
      thumbnail.setAttribute('tabindex', '-1')
      thumbnail.setAttribute('aria-hidden', 'true')
      thumbnail.srcdoc = previewDocument(concept.document)
      const overlay = document.createElement('button')
      overlay.type = 'button'
      overlay.setAttribute('aria-label', `查看${concept.name}完整预览`)
      overlay.addEventListener('click', () => selectConcept(concept.id))
      visual.append(thumbnail, overlay)
      const caption = document.createElement('div')
      caption.className = 'comparison-caption'
      const words = document.createElement('div')
      const title = document.createElement('h3')
      title.textContent = `${concept.id.slice(0, 2)} ${concept.name}`
      const subtitle = document.createElement('p')
      subtitle.textContent = descriptions[concept.id][2]
      words.append(title, subtitle)
      const button = document.createElement('button')
      button.type = 'button'
      button.textContent = '查看这款 ↗'
      button.addEventListener('click', () => selectConcept(concept.id))
      caption.append(words, button)
      card.append(visual, caption)
      comparison.append(card)
    })
    resizePreview()
  }

  function updateCompareMode() {
    stage.hidden = comparing
    comparison.hidden = !comparing
    document.getElementById('compare-toggle').setAttribute('aria-pressed', String(comparing))
    document.getElementById('compare-toggle').textContent = comparing ? '返回预览' : concepts.length === 5 ? '五款对比' : '全部对比'
    document.querySelectorAll('[data-viewport]').forEach(button => { button.disabled = comparing })
    if (comparing) renderComparison()
    else resizePreview()
  }

  function renderSelected() {
    cleanupInteraction?.()
    cleanupInteraction = undefined
    const description = descriptions[selected.id]
    document.querySelectorAll('[data-concept]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.concept === selected.id)))
    document.getElementById('concept-category').textContent = description[0]
    document.getElementById('concept-title').textContent = `${selected.id.slice(0, 2)} ${selected.name}`
    document.getElementById('concept-description').textContent = description[1]
    document.getElementById('mark-comparison-link').hidden = selected.id !== '05-atelier'
    frame.title = `${selected.name}完整首页预览`
    frame.srcdoc = previewDocument(selected.document)
    const previousUrl = pageObjectUrl
    pageObjectUrl = URL.createObjectURL(new Blob([branded(selected.document)], { type: 'text/html;charset=utf-8' }))
    document.getElementById('open-page').href = pageObjectUrl
    if (previousUrl) URL.revokeObjectURL(previousUrl)
    exportContent.value = branded(selected.fragment)
    document.getElementById('export-label').textContent = `${selected.id.slice(0, 2)} ${selected.name} · 后台粘贴版`
    updateCompareMode()
  }

  function selectConcept(identifier) {
    const concept = concepts.find(item => item.id === identifier)
    if (!concept) return
    selected = concept
    comparing = false
    if (location.hash.slice(1) !== identifier) location.hash = identifier
    renderSelected()
  }

  function download(content, filename) {
    const objectUrl = URL.createObjectURL(new Blob([content], { type: 'text/html;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = objectUrl
    anchor.download = filename
    document.body.append(anchor)
    anchor.click()
    anchor.remove()
    setTimeout(() => URL.revokeObjectURL(objectUrl), 10000)
    notify(`已发起下载：${filename}`)
  }

  frame.addEventListener('load', () => {
    cleanupInteraction?.()
    cleanupInteraction = frame.contentDocument && window.HC_HOME_ART?.interact(frame.contentDocument, window.HC_HOME_ART.artworks)
    frame.contentDocument?.addEventListener('click', event => {
      const anchor = event.target.closest('a')
      if (anchor?.getAttribute('href')?.startsWith('/')) {
        event.preventDefault()
        notify(`这是本地设计预览。正式上线后，此按钮前往 ${anchor.getAttribute('href')}（对应业务页面）。`)
      }
    })
  })
  document.querySelectorAll('[data-concept]').forEach(button => button.addEventListener('click', () => selectConcept(button.dataset.concept)))
  document.querySelectorAll('[data-viewport]').forEach(button => button.addEventListener('click', () => {
    viewport = button.dataset.viewport
    shell.classList.toggle('is-mobile', viewport === 'mobile')
    document.querySelectorAll('[data-viewport]').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.viewport === viewport)))
    document.getElementById('viewport-label').textContent = viewport === 'mobile' ? '手机 · 390 px' : '桌面 · 1440 px'
    resizePreview()
  }))
  document.getElementById('compare-toggle').addEventListener('click', () => { comparing = !comparing; updateCompareMode() })
  document.getElementById('reroll-artwork')?.addEventListener('click', () => {
    selectedArtwork = undefined
    renderSelected()
    notify(`当前主视觉：${selectedArtwork?.name || '金色轨道'}。页面风格保持不变。`)
  })
  document.getElementById('brand-form').addEventListener('submit', event => {
    event.preventDefault()
    brand = document.getElementById('brand-name').value.trim() || 'zz.naoy'
    renderSelected()
    notify('站点名称已应用到本地预览和导出内容，线上设置未修改。')
  })
  document.getElementById('export-button').addEventListener('click', () => { exportStatus.hidden = true; dialog.showModal() })
  document.getElementById('close-export').addEventListener('click', () => dialog.close())
  document.getElementById('copy-fragment').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(exportContent.value)
      notify('后台粘贴代码已复制。请备份旧内容后，再到后台粘贴并保存。')
    } catch {
      exportContent.focus()
      exportContent.select()
      notify('浏览器未允许自动复制；代码已全选，请按 Ctrl+C（复制快捷键）。')
    }
  })
  document.getElementById('download-fragment').addEventListener('click', () => download(branded(selected.fragment), `${selected.id}-后台粘贴.html`))
  document.getElementById('download-page').addEventListener('click', () => download(branded(selected.document), `${selected.id}-独立网页.html`))
  window.addEventListener('hashchange', () => {
    const concept = concepts.find(item => item.id === location.hash.slice(1))
    if (concept && concept.id !== selected.id) { selected = concept; comparing = false; renderSelected() }
  })
  new ResizeObserver(resizePreview).observe(stage)
  new ResizeObserver(resizePreview).observe(comparison)
  window.addEventListener('resize', resizePreview)
  window.addEventListener('pagehide', () => cleanupInteraction?.())
  renderSelected()
})()
