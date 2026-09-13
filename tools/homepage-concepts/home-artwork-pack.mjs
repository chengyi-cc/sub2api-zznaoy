import { artDirections } from './art-directions.mjs'
import { transparentGoldArt } from './transparent-artwork.mjs'
import { artwork } from './art-geometry.mjs'
import { interactionModels } from './interaction-models.mjs'

export const homeArtworks = [
  { id: '05-atelier', name: '金色轨道', markup: transparentGoldArt },
  ...artDirections.map(design => ({ id: design.id, name: design.name, markup: artwork({ ...design, transparent: true }) })),
].map(art => ({ ...art, model: interactionModels.get(art.id) }))
