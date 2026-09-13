import { depthCore } from './art-depth.mjs'

const format = value => Number(value.toFixed(2))

export function rotate(point, angles = [0, 0, 0]) {
  const [pitch, yaw, roll] = angles
  const vertical = point.y * Math.cos(pitch) - point.z * Math.sin(pitch)
  const depth = point.y * Math.sin(pitch) + point.z * Math.cos(pitch)
  const horizontal = point.x * Math.cos(yaw) + depth * Math.sin(yaw)
  const finalDepth = -point.x * Math.sin(yaw) + depth * Math.cos(yaw)
  return {
    x: horizontal * Math.cos(roll) - vertical * Math.sin(roll),
    y: horizontal * Math.sin(roll) + vertical * Math.cos(roll),
    z: finalDepth,
  }
}

export function project(point, angles = [0, 0, 0]) {
  const rotated = rotate(point, angles)
  const perspective = 1100 / (1100 - rotated.z)
  return { x: 300 + rotated.x * perspective, y: 292 + rotated.y * perspective, z: rotated.z }
}

export function sampleCurve(callback, count = 64) {
  return Array.from({ length: count }, (_unused, index) => callback(index * Math.PI * 2 / count))
}

export function curve(points, { closed = true, smooth = true, width = 0.8, opacity = 0.7, fill = 'none' } = {}) {
  const commands = [`M${format(points[0].x)} ${format(points[0].y)}`]
  if (smooth && closed) {
    for (let index = 0; index < points.length; index += 1) {
      const previous = points[(index + points.length - 1) % points.length]
      const current = points[index]
      const next = points[(index + 1) % points.length]
      const after = points[(index + 2) % points.length]
      commands.push(`C${format(current.x + (next.x - previous.x) / 6)} ${format(current.y + (next.y - previous.y) / 6)} ${format(next.x - (after.x - current.x) / 6)} ${format(next.y - (after.y - current.y) / 6)} ${format(next.x)} ${format(next.y)}`)
    }
  } else {
    for (const point of points.slice(1)) commands.push(`L${format(point.x)} ${format(point.y)}`)
  }
  if (closed) commands.push('Z')
  return {
    points,
    closed, smooth, width, opacity, fill,
    depth: points.reduce((sum, point) => sum + (point.z || 0), 0) / points.length,
    markup: `<path d="${commands.join(' ')}" stroke-width="${width}" stroke-opacity="${opacity}" fill="${fill}"/>`,
  }
}

export function dot(point, radius = 2, opacity = 0.8) {
  return {
    points: [point],
    depth: point.z || 0,
    markup: `<circle cx="${format(point.x)}" cy="${format(point.y)}" r="${radius}" fill="#e4d1a7" stroke="none" fill-opacity="${opacity}"/>`,
  }
}

export function artwork({ id, name, caption, nodes, seal = 'circle', angles = [0, 0, 0], transparent = false, palette = ['#6c5839', '#e0cea6', '#9b8158', '#f0dec1', '#685639'] }) {
  const prefix = `hc-art-${id}`
  const layers = depthCore({ id, seal, angles, nodes, transparent }, { project, rotate, curve })
  return `<div class="hc-gold-art"${transparent ? ' data-art-surface="transparent"' : ''} data-art-style="${id}" role="img" aria-label="${name}：${caption}"><svg viewBox="0 0 600 590" fill="none" aria-hidden="true"><defs><linearGradient id="${prefix}-metal" x1="115" y1="88" x2="480" y2="515" gradientUnits="userSpaceOnUse">${palette.map((color, index) => `<stop offset="${index / (palette.length - 1)}" stop-color="${color}"${transparent ? ` style="stop-color:var(--hf-metal-${index + 1},${color})"` : ''}/>`).join('')}</linearGradient><radialGradient id="${prefix}-air"><stop stop-color="${palette[1]}" stop-opacity=".045"/><stop offset="1" stop-color="${palette[1]}" stop-opacity="0"/></radialGradient>${layers.definitions}</defs>${transparent ? '' : `<circle cx="300" cy="292" r="247" fill="url(#${prefix}-air)"/>`}<g data-depth-layer="back"${transparent ? ` mask="url(#${prefix}-core-cutout)"` : ''} stroke="url(#${prefix}-metal)" stroke-linecap="round" stroke-linejoin="round">${[...nodes].sort((first, second) => first.depth - second.depth).map(node => node.markup).join('')}</g>${layers.core}${layers.foreground}</svg><span class="hc-gold-art-label">${caption}</span><span class="hc-gold-coordinate">${name} / FORM EXPLORATION</span></div>`
}
