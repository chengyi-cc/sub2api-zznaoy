import { artDirections } from './art-directions.mjs'

const round = value => Number(value.toFixed(4)) || 0
const defaultPalette = ['#6c5839', '#e0cea6', '#9b8158', '#f0dec1', '#685639']

function cameraCoordinates(point) {
  const scale = 1 - point.z / 1100
  return [round((point.x - 300) * scale), round((point.y - 292) * scale), round(point.z)]
}

const originalNodes = Array.from({ length: 24 }, (_unused, index) => {
  const horizontalRadius = 100 + index * 3.3
  const verticalRadius = 226 - index * 2.1
  const rotation = index * Math.PI / 24
  const depthRadius = Math.sqrt(226 ** 2 - horizontalRadius ** 2)
  return {
    points: Array.from({ length: 64 }, (_point, position) => {
      const angle = position * Math.PI * 2 / 64
      const horizontal = horizontalRadius * Math.cos(angle)
      const vertical = verticalRadius * Math.sin(angle)
      return cameraCoordinates({ x: 300 + horizontal * Math.cos(rotation) - vertical * Math.sin(rotation), y: 292 + horizontal * Math.sin(rotation) + vertical * Math.cos(rotation), z: depthRadius * Math.cos(angle) })
    }),
    closed: true, smooth: true, width: 0.8, opacity: round(0.28 + index * 0.023), fill: 'none', radius: 0,
  }
})

export const interactionModels = new Map([
  ['05-atelier', { id: '05-atelier', nodes: originalNodes, radius: 52, sides: 64, angles: [0, 0, 0], glyph: [[-18, 17], [-18, -17], [-9, -17], [11, 17], [18, 17], [18, -17]], palette: ['#504331', '#ecd5a6', '#9c8157', '#3a3326'] }],
  ...artDirections.map(design => [design.id, {
    id: design.id,
    nodes: design.nodes.map(node => ({
      points: node.points.map(cameraCoordinates),
      closed: node.closed || false, smooth: node.smooth || false,
      width: node.width ?? 0, opacity: node.opacity ?? Number(node.markup.match(/fill-opacity="([^"]+)"/)?.[1] || 1),
      fill: node.fill || 'none', radius: Number(node.markup.match(/ r="([^"]+)"/)?.[1] || 0),
    })),
    radius: design.seal === 'small' ? 31 : design.seal === 'diamond' ? 44 : 38,
    sides: design.seal === 'diamond' ? 4 : 64,
    angles: design.angles,
    glyph: [[-14, 15], [-14, -15], [-7, -15], [8, 15], [14, 15], [14, -15]],
    palette: design.palette || defaultPalette,
  }]),
])
