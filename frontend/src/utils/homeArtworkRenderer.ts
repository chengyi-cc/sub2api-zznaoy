export interface ArtworkPoint {
  x: number
  y: number
  z: number
}

export interface ArtworkStroke {
  points: number[][]
  closed: boolean
  smooth: boolean
  width: number
  opacity: number
  fill: string
  radius: number
}

export interface HomeArtworkModel {
  id: string
  nodes: ArtworkStroke[]
  radius: number
  sides: number
  angles: number[]
  glyph: number[][]
  palette: string[]
}

interface ProjectedStroke extends Omit<ArtworkStroke, 'points'> {
  points: ArtworkPoint[]
  depth: number
}

function createArtworkRotation(angles: number[]): (point: ArtworkPoint) => ArtworkPoint {
  const [pitch, yaw, roll = 0] = angles
  const pitchCos = Math.cos(pitch)
  const pitchSin = Math.sin(pitch)
  const yawCos = Math.cos(yaw)
  const yawSin = Math.sin(yaw)
  const rollCos = Math.cos(roll)
  const rollSin = Math.sin(roll)
  return point => {
    const vertical = point.y * pitchCos - point.z * pitchSin
    const depth = point.y * pitchSin + point.z * pitchCos
    const horizontal = point.x * yawCos + depth * yawSin
    return { x: horizontal * rollCos - vertical * rollSin, y: horizontal * rollSin + vertical * rollCos, z: -point.x * yawSin + depth * yawCos }
  }
}

function projectArtworkPoint(point: ArtworkPoint): ArtworkPoint {
  const scale = 1100 / (1100 - point.z)
  return { x: 300 + point.x * scale, y: 292 + point.y * scale, z: point.z }
}

function artworkNumber(value: number): string {
  return String(Math.round(value * 100) / 100)
}

function artworkPath(points: ArtworkPoint[], closed = true, smooth = false): string {
  if (!points.length) return ''
  const commands = [`M${artworkNumber(points[0].x)} ${artworkNumber(points[0].y)}`]
  if (smooth && closed) {
    for (let index = 0; index < points.length; index += 1) {
      const previous = points[(index + points.length - 1) % points.length]
      const current = points[index]
      const next = points[(index + 1) % points.length]
      const after = points[(index + 2) % points.length]
      commands.push(`C${artworkNumber(current.x + (next.x - previous.x) / 6)} ${artworkNumber(current.y + (next.y - previous.y) / 6)} ${artworkNumber(next.x - (after.x - current.x) / 6)} ${artworkNumber(next.y - (after.y - current.y) / 6)} ${artworkNumber(next.x)} ${artworkNumber(next.y)}`)
    }
  } else {
    for (const point of points.slice(1)) commands.push(`L${artworkNumber(point.x)} ${artworkNumber(point.y)}`)
  }
  if (closed) commands.push('Z')
  return commands.join(' ')
}

function artworkStrokeMarkup(node: ProjectedStroke): string {
  if (node.radius) {
    const point = node.points[0]
    return `<circle cx="${artworkNumber(point.x)}" cy="${artworkNumber(point.y)}" r="${node.radius}" fill="var(--hf-metal-2,#e4d1a7)" fill-opacity="${node.opacity}" stroke="none"/>`
  }
  return `<path d="${artworkPath(node.points, node.closed, node.smooth)}" stroke-width="${node.width}" stroke-opacity="${node.opacity}" fill="${node.fill}"/>`
}

function sampleArtworkStroke(node: ProjectedStroke): ArtworkPoint[] {
  if (!node.closed || !node.smooth) return node.points
  const points: ArtworkPoint[] = []
  for (let index = 0; index < node.points.length; index += 1) {
    const previous = node.points[(index + node.points.length - 1) % node.points.length]
    const current = node.points[index]
    const next = node.points[(index + 1) % node.points.length]
    const after = node.points[(index + 2) % node.points.length]
    for (let sample = 0; sample < 4; sample += 1) {
      const fraction = sample / 4
      const value = (axis: keyof ArtworkPoint) => (1 - fraction) ** 3 * current[axis] + 3 * (1 - fraction) ** 2 * fraction * (current[axis] + (next[axis] - previous[axis]) / 6) + 3 * (1 - fraction) * fraction ** 2 * (next[axis] - (after[axis] - current[axis]) / 6) + fraction ** 3 * next[axis]
      points.push({ x: value('x'), y: value('y'), z: value('z') })
    }
  }
  return points
}

function artworkFrontSections(node: ProjectedStroke, normal: ArtworkPoint): ArtworkPoint[][] {
  const distance = (point: ArtworkPoint) => {
    const inverseScale = 1 - point.z / 1100
    return (point.x - 300) * inverseScale * normal.x + (point.y - 292) * inverseScale * normal.y + point.z * normal.z - 2
  }
  const intersection = (first: ArtworkPoint, second: ArtworkPoint) => {
    const fraction = distance(first) / (distance(first) - distance(second))
    const firstScale = 1 - first.z / 1100
    const secondScale = 1 - second.z / 1100
    return projectArtworkPoint({ x: (first.x - 300) * firstScale * (1 - fraction) + (second.x - 300) * secondScale * fraction, y: (first.y - 292) * firstScale * (1 - fraction) + (second.y - 292) * secondScale * fraction, z: first.z + fraction * (second.z - first.z) })
  }
  if (node.radius) return distance(node.points[0]) >= 0 ? [node.points] : []
  const points = sampleArtworkStroke(node)
  if (node.fill !== 'none') {
    const clipped: ArtworkPoint[] = []
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index]
      const next = points[(index + 1) % points.length]
      const currentFront = distance(current) >= 0
      const nextFront = distance(next) >= 0
      if (currentFront) clipped.push(current)
      if (currentFront !== nextFront) clipped.push(intersection(current, next))
    }
    return clipped.length >= 3 ? [clipped] : []
  }
  const sections: ArtworkPoint[][] = []
  let section: ArtworkPoint[] = []
  const sequence = node.closed ? [...points, points[0]] : points
  for (let index = 0; index < sequence.length - 1; index += 1) {
    const current = sequence[index]
    const next = sequence[index + 1]
    const currentFront = distance(current) >= 0
    const nextFront = distance(next) >= 0
    if (currentFront && !section.length) section.push(current)
    if (currentFront && nextFront) section.push(next)
    if (currentFront && !nextFront) {
      section.push(intersection(current, next))
      sections.push(section)
      section = []
    }
    if (!currentFront && nextFront) section.push(intersection(current, next), next)
  }
  if (section.length > 1) sections.push(section)
  return sections
}

function artworkCrossesCenter(node: ProjectedStroke, radius: number): boolean {
  if (node.fill !== 'none') return true
  if (node.radius) return Math.abs(node.points[0].x - 300) <= radius && Math.abs(node.points[0].y - 292) <= radius
  const spanCount = node.closed ? node.points.length : node.points.length - 1
  for (let index = 0; index < spanCount; index += 1) {
    const current = node.points[index]
    const next = node.points[(index + 1) % node.points.length]
    const horizontal = [current.x, next.x]
    const vertical = [current.y, next.y]
    if (node.closed && node.smooth) {
      const previous = node.points[(index + node.points.length - 1) % node.points.length]
      const after = node.points[(index + 2) % node.points.length]
      horizontal.push(current.x + (next.x - previous.x) / 6, next.x - (after.x - current.x) / 6)
      vertical.push(current.y + (next.y - previous.y) / 6, next.y - (after.y - current.y) / 6)
    }
    if (Math.min(...horizontal) <= 300 + radius && Math.max(...horizontal) >= 300 - radius && Math.min(...vertical) <= 292 + radius && Math.max(...vertical) >= 292 - radius) return true
  }
  return false
}

function trimArtworkForeground(points: ArtworkPoint[], radius: number): ArtworkPoint[][] {
  const sections: ArtworkPoint[][] = []
  let section: ArtworkPoint[] = []
  for (let index = 0; index < points.length - 1; index += 1) {
    const first = points[index]
    const second = points[index + 1]
    let start = 0
    let end = 1
    for (const [axis, center] of [['x', 300], ['y', 292]] as const) {
      const delta = second[axis] - first[axis]
      if (delta === 0) {
        if (Math.abs(first[axis] - center) > radius) end = -1
      } else {
        const before = (center - radius - first[axis]) / delta
        const after = (center + radius - first[axis]) / delta
        start = Math.max(start, Math.min(before, after))
        end = Math.min(end, Math.max(before, after))
      }
    }
    if (start > end) {
      if (section.length > 1) sections.push(section)
      section = []
      continue
    }
    const interpolate = (fraction: number) => ({ x: first.x + (second.x - first.x) * fraction, y: first.y + (second.y - first.y) * fraction, z: first.z + (second.z - first.z) * fraction })
    if (!section.length) section.push(interpolate(start))
    section.push(interpolate(end))
    if (end < 1) { sections.push(section); section = [] }
  }
  if (section.length > 1) sections.push(section)
  return sections
}

function projectHomeArtwork(model: HomeArtworkModel, pitch: number, yaw: number) {
  const fullTurn = Math.PI * 2
  const rotate = createArtworkRotation([pitch % fullTurn || 0, yaw % fullTurn || 0, 0])
  const rotateCore = createArtworkRotation(model.angles)
  const corePoint = (point: ArtworkPoint) => rotate(rotateCore(point))
  const originalNormal = corePoint({ x: 0, y: 0, z: 1 })
  const facing = originalNormal.z >= 0 ? 1 : -1
  const normal = { x: originalNormal.x * facing, y: originalNormal.y * facing, z: originalNormal.z * facing }
  const contour = (depth: number, expansion = 0) => Array.from({ length: model.sides }, (_unused, index) => {
    const position = index * Math.PI * 2 / model.sides - Math.PI / 2
    return projectArtworkPoint(corePoint({ x: (model.radius + expansion) * Math.cos(position), y: (model.radius + expansion) * Math.sin(position), z: depth * facing }))
  })
  const front = artworkPath(contour(2))
  const back = artworkPath(contour(-2))
  const nodes: ProjectedStroke[] = model.nodes.map(node => {
    const points = node.points.map(([horizontal, vertical, depth]) => projectArtworkPoint(rotate({ x: horizontal, y: vertical, z: depth })))
    return { ...node, points, depth: points.reduce((sum, point) => sum + point.z, 0) / points.length }
  }).sort((first, second) => first.depth - second.depth)
  const radius = model.radius + 24
  const foreground = nodes.filter(node => artworkCrossesCenter(node, radius)).flatMap(node => artworkFrontSections(node, normal).flatMap(points => {
    const sections = node.fill !== 'none' || node.radius ? [points] : trimArtworkForeground(points, radius)
    return sections.map(section => ({ ...node, points: section, closed: node.fill !== 'none', smooth: false }))
  }))
  const glyph = artworkPath(model.glyph.map(([horizontal, vertical]) => projectArtworkPoint(corePoint({ x: horizontal, y: vertical, z: 2.2 * facing }))), false)
  return { front, back, glyph, window: artworkPath(contour(2, 1.5)), nodes, foreground }
}

function artworkFrameMarkup(model: HomeArtworkModel, frame: ReturnType<typeof projectHomeArtwork>): string {
  const { front, back, glyph, nodes, foreground } = frame
  const prefix = `hc-live-${model.id}`
  const gradient = `<linearGradient id="${prefix}-metal" x1="115" y1="88" x2="480" y2="515" gradientUnits="userSpaceOnUse">${model.palette.map((color, index) => `<stop offset="${index / (model.palette.length - 1)}" stop-color="${color}" style="stop-color:var(--hf-metal-${index + 1},${color})"/>`).join('')}</linearGradient>`
  return `<defs>${gradient}<mask id="${prefix}-cutout" maskUnits="userSpaceOnUse" x="0" y="0" width="600" height="590" style="mask-type:luminance"><rect width="600" height="590" fill="white"/><path d="${front}" fill="black"/><path d="${back}" fill="black"/></mask><clipPath id="${prefix}-window"><path d="${frame.window}"/></clipPath></defs><g fill="none" stroke="url(#${prefix}-metal)" stroke-linecap="round" stroke-linejoin="round"><g data-depth-layer="back" mask="url(#${prefix}-cutout)">${nodes.map(artworkStrokeMarkup).join('')}</g><g data-depth-layer="core" data-world-center="0 0 0"><path d="${back}" stroke-width=".8" stroke-opacity=".65"/><path d="${front}" stroke-width=".75" stroke-opacity=".6"/><path d="${glyph}" stroke-width="${model.id === '05-atelier' ? 2 : 1.65}"/></g><g data-depth-layer="front" clip-path="url(#${prefix}-window)">${foreground.map(artworkStrokeMarkup).join('')}</g></g>`
}

export function renderHomeArtwork(model: HomeArtworkModel, pitch: number, yaw: number): string {
  return artworkFrameMarkup(model, projectHomeArtwork(model, pitch, yaw))
}

function updateArtworkAttribute(element: Element, name: string, value: string) {
  if (element.getAttribute(name) !== value) element.setAttribute(name, value)
}

function createArtworkLayerUpdater(layer: Element) {
  const pool: Record<'path' | 'circle', Element[]> = { path: [], circle: [] }
  return (nodes: ProjectedStroke[]) => {
    const used = { path: 0, circle: 0 }
    nodes.forEach((node, index) => {
      const tag = node.radius ? 'circle' : 'path'
      const position = used[tag]++
      const element = pool[tag][position] ??= layer.ownerDocument.createElementNS('http://www.w3.org/2000/svg', tag)
      if (node.radius) {
        updateArtworkAttribute(element, 'cx', artworkNumber(node.points[0].x))
        updateArtworkAttribute(element, 'cy', artworkNumber(node.points[0].y))
        updateArtworkAttribute(element, 'r', String(node.radius))
        updateArtworkAttribute(element, 'fill', 'var(--hf-metal-2,#e4d1a7)')
        updateArtworkAttribute(element, 'fill-opacity', String(node.opacity))
        updateArtworkAttribute(element, 'stroke', 'none')
      } else {
        updateArtworkAttribute(element, 'd', artworkPath(node.points, node.closed, node.smooth))
        updateArtworkAttribute(element, 'stroke-width', String(node.width))
        updateArtworkAttribute(element, 'stroke-opacity', String(node.opacity))
        updateArtworkAttribute(element, 'fill', node.fill)
      }
      if (layer.children[index] !== element) layer.insertBefore(element, layer.children[index] ?? null)
    })
    while (layer.children.length > nodes.length) layer.lastElementChild!.remove()
  }
}

export function createHomeArtworkRenderer(svg: SVGSVGElement, model: HomeArtworkModel): (pitch: number, yaw: number) => void {
  let layers: {
    anchor: Element
    paths: Element[]
    back: (nodes: ProjectedStroke[]) => void
    front: (nodes: ProjectedStroke[]) => void
  } | null = null
  return (pitch, yaw) => {
    const frame = projectHomeArtwork(model, pitch, yaw)
    if (!layers || !svg.contains(layers.anchor)) {
      svg.innerHTML = artworkFrameMarkup(model, { ...frame, nodes: [], foreground: [] })
      const back = svg.querySelector('[data-depth-layer="back"]')!
      const front = svg.querySelector('[data-depth-layer="front"]')!
      layers = {
        anchor: back,
        paths: [...svg.querySelectorAll('mask path, clipPath path, [data-depth-layer="core"] path')],
        back: createArtworkLayerUpdater(back),
        front: createArtworkLayerUpdater(front),
      }
    }
    const paths = [frame.front, frame.back, frame.window, frame.back, frame.front, frame.glyph]
    paths.forEach((path, index) => updateArtworkAttribute(layers!.paths[index], 'd', path))
    layers.back(frame.nodes)
    layers.front(frame.foreground)
  }
}
