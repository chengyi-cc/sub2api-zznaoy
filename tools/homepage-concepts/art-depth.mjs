const cameraDistance = 1100

function cameraPoint(point) {
  const scale = 1 - point.z / cameraDistance
  return { x: (point.x - 300) * scale, y: (point.y - 292) * scale, z: point.z }
}

function signedDistance(point, normal) {
  const camera = cameraPoint(point)
  return camera.x * normal.x + camera.y * normal.y + camera.z * normal.z - 2
}

function intersection(first, second, normal) {
  const before = signedDistance(first, normal)
  const after = signedDistance(second, normal)
  const fraction = before / (before - after)
  const firstCamera = cameraPoint(first)
  const secondCamera = cameraPoint(second)
  const camera = Object.fromEntries(['x', 'y', 'z'].map(axis => [axis, firstCamera[axis] + fraction * (secondCamera[axis] - firstCamera[axis])]))
  const scale = cameraDistance / (cameraDistance - camera.z)
  return { x: 300 + camera.x * scale, y: 292 + camera.y * scale, z: camera.z }
}

function sampledPoints(node) {
  if (!node.smooth || !node.closed) return node.points
  const samples = []
  for (let index = 0; index < node.points.length; index += 1) {
    const previous = node.points[(index + node.points.length - 1) % node.points.length]
    const current = node.points[index]
    const next = node.points[(index + 1) % node.points.length]
    const after = node.points[(index + 2) % node.points.length]
    for (let sample = 0; sample < 8; sample += 1) {
      const fraction = sample / 8
      samples.push(Object.fromEntries(['x', 'y', 'z'].map(axis => {
        const firstControl = current[axis] + (next[axis] - previous[axis]) / 6
        const secondControl = next[axis] - (after[axis] - current[axis]) / 6
        return [axis, (1 - fraction) ** 3 * current[axis] + 3 * (1 - fraction) ** 2 * fraction * firstControl + 3 * (1 - fraction) * fraction ** 2 * secondControl + fraction ** 3 * next[axis]]
      })))
    }
  }
  return samples
}

export function frontSections(node, normal) {
  if (node.points.length === 1) return signedDistance(node.points[0], normal) >= 0 ? [node.points] : []
  const points = sampledPoints(node)
  if (node.fill && node.fill !== 'none') {
    const clipped = []
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index]
      const next = points[(index + 1) % points.length]
      const currentInFront = signedDistance(current, normal) >= 0
      const nextInFront = signedDistance(next, normal) >= 0
      if (currentInFront) clipped.push(current)
      if (currentInFront !== nextInFront) clipped.push(intersection(current, next, normal))
    }
    return clipped.length >= 3 ? [clipped] : []
  }
  const sections = []
  let section = []
  const sequence = node.closed ? [...points, points[0]] : points
  for (let index = 0; index < sequence.length - 1; index += 1) {
    const current = sequence[index]
    const next = sequence[index + 1]
    const currentInFront = signedDistance(current, normal) >= 0
    const nextInFront = signedDistance(next, normal) >= 0
    if (currentInFront && section.length === 0) section.push(current)
    if (currentInFront && nextInFront) section.push(next)
    if (currentInFront && !nextInFront) {
      section.push(intersection(current, next, normal))
      sections.push(section)
      section = []
    }
    if (!currentInFront && nextInFront) section.push(intersection(current, next, normal), next)
  }
  if (section.length > 1) sections.push(section)
  return sections
}

export function depthCore({ id, seal, angles, nodes, transparent = false }, { project, rotate, curve }) {
  const prefix = `hc-art-${id}`
  const normal = rotate({ x: 0, y: 0, z: 1 }, angles)
  const radius = seal === 'small' ? 31 : seal === 'diamond' ? 44 : 38
  const contour = (depth, expansion = 0) => Array.from({ length: seal === 'diamond' ? 4 : 64 }, (_unused, index) => {
    const angle = index * Math.PI * 2 / (seal === 'diamond' ? 4 : 64) - Math.PI / 2
    return project({ x: (radius + expansion) * Math.cos(angle), y: (radius + expansion) * Math.sin(angle), z: depth }, angles)
  })
  const silhouette = points => `M${points.map(point => `${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join('L')}Z`
  const front = contour(2)
  const back = contour(-2)
  const glyph = [[-14, 15], [-14, -15], [-7, -15], [8, 15], [14, 15], [14, -15]].map(([horizontal, vertical]) => project({ x: horizontal, y: vertical, z: 2.2 }, angles))
  const foreground = [...nodes].sort((first, second) => first.depth - second.depth).flatMap(node => {
    const sections = frontSections(node, normal)
    if (node.points.length === 1) return sections.length ? [node.markup] : []
    return sections.map(points => curve(points, { closed: node.fill !== 'none', smooth: false, width: node.width, opacity: node.opacity, fill: node.fill }).markup)
  }).join('')
  return {
    definitions: `<clipPath id="${prefix}-core-window"><path d="${silhouette(contour(2, 1.5))}"/></clipPath>${transparent ? `<mask id="${prefix}-core-cutout" maskUnits="userSpaceOnUse" x="0" y="0" width="600" height="590" style="mask-type:luminance"><rect width="600" height="590" fill="white"/><path d="${silhouette(back)}" fill="black"/><path d="${silhouette(front)}" fill="black"/></mask>` : ''}`,
    core: `<g data-depth-layer="core" data-world-center="0 0 0" stroke="url(#${prefix}-metal)"><path d="${silhouette(back)}" fill="${transparent ? 'none' : '#24231e'}" stroke-width=".8" stroke-opacity=".65"/><path d="${silhouette(front)}" fill="${transparent ? 'none' : '#171814'}" stroke-width=".75" stroke-opacity=".6"/>${curve(glyph, { closed: false, smooth: false, width: 1.65, opacity: 1 }).markup}</g>`,
    foreground: `<g data-depth-layer="front" clip-path="url(#${prefix}-core-window)" stroke="url(#${prefix}-metal)" stroke-linecap="round" stroke-linejoin="round">${foreground}</g>`,
  }
}
