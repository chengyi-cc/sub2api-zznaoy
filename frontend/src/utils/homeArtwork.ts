import artworks from '@/assets/home-artworks.json'
import { applyRandomHomeArtwork } from './homeArtworkSelection'
import { initializeHomeArtworkInteractions } from './homeArtworkInteraction'

export function initializeHomeArtwork(root: ParentNode): () => void {
  applyRandomHomeArtwork(root, artworks)
  return initializeHomeArtworkInteractions(root, artworks)
}
