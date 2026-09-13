import { artwork, curve, dot, project, rotate, sampleCurve } from './art-geometry.mjs'
import { concepts } from './variants.mjs'
import { originalGoldArt } from './concepts.mjs'

function armillary() {
  const nodes = []
  const sphereAngles = [0.36, -0.38, -0.14]
  for (let latitude = -4; latitude <= 4; latitude += 1) {
    const angle = latitude * Math.PI / 12
    const radius = 120 * Math.cos(angle)
    nodes.push(curve(sampleCurve(position => project({ x: radius * Math.cos(position), y: 120 * Math.sin(angle), z: radius * Math.sin(position) }, sphereAngles), 36), { width: 0.65, opacity: 0.38 }))
  }
  for (let meridian = 0; meridian < 10; meridian += 1) {
    const angle = meridian * Math.PI / 10
    nodes.push(curve(sampleCurve(position => project({ x: 120 * Math.cos(position) * Math.cos(angle), y: 120 * Math.sin(position), z: 120 * Math.cos(position) * Math.sin(angle) }, sphereAngles), 36), { width: 0.65, opacity: 0.45 }))
  }
  const orbits = [
    { radius: 180, angles: [0.97, 0.3, -0.48] },
    { radius: 206, angles: [0.46, 0.73, 0.42] },
    { radius: 228, angles: [0.91, -0.59, 0.2] },
  ]
  for (const orbit of orbits) {
    for (const offset of [-2.2, 2.2]) {
      nodes.push(curve(sampleCurve(angle => project({ x: (orbit.radius + offset) * Math.cos(angle), y: (orbit.radius + offset) * Math.sin(angle), z: 0 }, orbit.angles), 48), { width: offset > 0 ? 0.9 : 0.6, opacity: offset > 0 ? 0.85 : 0.34 }))
    }
    const position = orbit.radius / 37
    nodes.push(dot(project({ x: orbit.radius * Math.cos(position), y: orbit.radius * Math.sin(position), z: 0 }, orbit.angles), 3.2))
  }
  return nodes
}

function crystal() {
  const ratio = (1 + Math.sqrt(5)) / 2
  const vertices = []
  for (const first of [-1, 1]) {
    for (const second of [-ratio, ratio]) {
      vertices.push({ x: 0, y: first, z: second }, { x: first, y: second, z: 0 }, { x: second, y: 0, z: first })
    }
  }
  const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y, first.z - second.z)
  const angles = [-0.24, 0.53, 0.11]
  const nodes = []
  const outer = vertices.map(point => project({ x: point.x * 119, y: point.y * 130, z: point.z * 119 }, angles))
  const inner = vertices.map(point => project({ x: point.x * 55, y: point.y * 60, z: point.z * 55 }, angles))
  for (let first = 0; first < vertices.length; first += 1) {
    nodes.push(dot(outer[first], 1.7, 0.65))
    nodes.push(curve([outer[first], inner[first]], { closed: false, smooth: false, width: 0.6, opacity: 0.28 }))
    for (let second = first + 1; second < vertices.length; second += 1) {
      if (Math.abs(distance(vertices[first], vertices[second]) - 2) > 0.01) continue
      nodes.push(curve([outer[first], outer[second]], { closed: false, smooth: false, width: 1.05, opacity: 0.8 }))
      nodes.push(curve([inner[first], inner[second]], { closed: false, smooth: false, width: 0.65, opacity: 0.42 }))
      for (let third = second + 1; third < vertices.length; third += 1) {
        if (Math.abs(distance(vertices[first], vertices[third]) - 2) > 0.01 || Math.abs(distance(vertices[second], vertices[third]) - 2) > 0.01) continue
        const face = [outer[first], outer[second], outer[third]]
        const depth = face.reduce((sum, point) => sum + point.z, 0) / 3
        nodes.push(curve(face, { smooth: false, opacity: 0, fill: depth > 0 ? '#d6d3bb09' : '#d6d3bb03' }))
      }
    }
  }
  return nodes
}

function corona() {
  const nodes = []
  const pointOnRing = (around, tube) => project({
    x: (150 + 55 * Math.cos(tube)) * Math.cos(around),
    y: (150 + 55 * Math.cos(tube)) * Math.sin(around),
    z: 55 * Math.sin(tube),
  }, [0.48, -0.22, -0.18])
  for (let longitude = 0; longitude < 46; longitude += 1) {
    nodes.push(curve(sampleCurve(tube => pointOnRing(longitude * Math.PI * 2 / 46, tube), 24), { width: 0.72, opacity: 0.57 }))
  }
  for (let latitude = 0; latitude < 18; latitude += 1) {
    nodes.push(curve(sampleCurve(around => pointOnRing(around, latitude * Math.PI * 2 / 18), 64), { width: 0.72, opacity: 0.49 }))
  }
  return nodes
}

function trefoil() {
  const centerline = angle => ({
    x: (132 + 45 * Math.cos(3 * angle)) * Math.cos(2 * angle),
    y: (132 + 45 * Math.cos(3 * angle)) * Math.sin(2 * angle),
    z: 45 * Math.sin(3 * angle),
  })
  const normalize = point => {
    const length = Math.hypot(point.x, point.y, point.z)
    return { x: point.x / length, y: point.y / length, z: point.z / length }
  }
  const cross = (first, second) => ({ x: first.y * second.z - first.z * second.y, y: first.z * second.x - first.x * second.z, z: first.x * second.y - first.y * second.x })
  const strandPoint = (angle, around) => {
    const center = centerline(angle)
    const before = centerline(angle - 0.001)
    const after = centerline(angle + 0.001)
    const tangent = normalize({ x: after.x - before.x, y: after.y - before.y, z: after.z - before.z })
    const normal = normalize(cross(tangent, { x: 0, y: 0, z: 1 }))
    const binormal = cross(tangent, normal)
    const point = {
      x: center.x + 22 * (normal.x * Math.cos(around) + binormal.x * Math.sin(around)),
      y: center.y + 22 * (normal.y * Math.cos(around) + binormal.y * Math.sin(around)),
      z: center.z + 22 * (normal.z * Math.cos(around) + binormal.z * Math.sin(around)),
    }
    return project(point, [0.28, -0.21, -0.45])
  }
  const nodes = []
  for (let strand = 0; strand < 16; strand += 1) {
    nodes.push(curve(sampleCurve(angle => strandPoint(angle, strand * Math.PI * 2 / 16), 144), { width: 0.76, opacity: 0.62 }))
  }
  for (let section = 0; section < 42; section += 1) {
    nodes.push(curve(sampleCurve(around => strandPoint(section * Math.PI * 2 / 42, around), 16), { width: 0.45, opacity: 0.19 }))
  }
  return nodes
}

function lattice() {
  const vertices = []
  for (const horizontal of [-1, 1]) {
    for (const vertical of [-1, 1]) {
      for (const depth of [-1, 1]) vertices.push({ x: horizontal, y: vertical, z: depth })
    }
  }
  const nodes = []
  for (let layer = 0; layer < 8; layer += 1) {
    const radius = 58 + layer * 10
    const angles = [-0.43, 0.56, -0.12]
    const points = vertices.map(point => project(rotate({ x: point.x * radius, y: point.y * radius, z: point.z * radius }, [0, (layer - 3.5) * 0.046, 0]), angles))
    for (let first = 0; first < vertices.length; first += 1) {
      if (layer === 7) nodes.push(dot(points[first], 1.8, 0.7))
      for (let second = first + 1; second < vertices.length; second += 1) {
        const distance = Math.abs(vertices[first].x - vertices[second].x) + Math.abs(vertices[first].y - vertices[second].y) + Math.abs(vertices[first].z - vertices[second].z)
        if (distance !== 2) continue
        const depth = (points[first].z + points[second].z) / 2
        nodes.push(curve([points[first], points[second]], { closed: false, smooth: false, width: layer === 7 ? 1 : 0.6, opacity: depth > 0 ? 0.7 : 0.31 }))
      }
    }
  }
  return nodes
}

const designs = [
  { id: '06-armillary', angles: [0.36, -0.38, -0.14], name: '天仪星轨', caption: 'CELESTIAL MECHANICS', tag: '天文仪器 · 轨道层次', description: '以古典天球仪为灵感：中央经纬球与多组倾斜金属轨道，强调精密仪器般的立体层次。', nodes: armillary(), seal: 'small' },
  { id: '07-crystal', angles: [-0.24, 0.53, 0.11], name: '钻切晶核', caption: 'PRECISION IN EVERY FACET', tag: '切割晶体 · 棱角分明', description: '外层多面晶体包裹内层晶核，冷金属细线与半透明切面交错。更像悬浮的高级珠宝，而不是圆环。', nodes: crystal(), seal: 'small', palette: ['#687273', '#d7ddcd', '#8c9794', '#e5d8ba', '#686958'] },
  { id: '08-corona', angles: [0.48, -0.22, -0.18], name: '日冕织环', caption: 'A SCULPTURE OF LIGHT', tag: '金属编织 · 饱满体积', description: '保留环形与饱满体积，用经纬金丝编织出厚实的立体环体。五款中更接近原版的交织气质，但结构更有秩序。', nodes: corona() },
  { id: '09-trefoil', angles: [0.28, -0.21, -0.45], name: '无界三叶', caption: 'CONTINUITY WITHOUT END', tag: '三叶结体 · 艺术雕塑', description: '连续金属管线在空间中穿插成三叶结，轮廓更有雕塑感。不是上次的扁平丝带，而是具有管状厚度的立体结构。', nodes: trefoil(), seal: 'small', palette: ['#775942', '#e5c8a4', '#ac7956', '#efd4ba', '#70543e'] },
  { id: '10-lattice', angles: [-0.43, 0.56, -0.12], name: '悬浮方界', caption: 'ARCHITECTURE OF POSSIBILITY', tag: '空间构架 · 建筑秩序', description: '多层轻微扭转的立方框架向中心递进，像悬浮的微型建筑。用直线与透视代替曲线，更理性、更有秩序感。', nodes: lattice(), seal: 'diamond' },
]

const base = concepts.find(concept => concept.id === '05-atelier')

export const artDirections = designs.map(design => {
  const art = artwork(design)
  return {
    ...design,
    art,
    html: base.html.replace(originalGoldArt, art),
  }
})
