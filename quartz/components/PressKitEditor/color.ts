import type { CardLayer } from '../scripts/press-kit-layers'
import type { EditorState } from './layout'

/** A highlighted layer's highlighter colour sits beside its ink in `colors`, under this key. */
export const backgroundKey = (id: string) => `${id}:bg`

export function colorImage(state: EditorState, layer: CardLayer, source: CardLayer['image']) {
  const colors = state.draft.value.colors
  const color = colors[layer.id]
  const background = layer.highlight ? colors[backgroundKey(layer.id)] : undefined
  if ((!color && !background) || layer.recolor === false) return source
  // Without its own repaint a layer can only take a flat ink fill.
  if (!color && !layer.recolor) return source
  const key = `${color ?? ''}|${background ?? ''}`
  const cached = state.tints.get(layer)
  if (cached?.source === source && cached.color === key) return cached.image
  const image = layer.recolor ? layer.recolor(color, background) : document.createElement('canvas')
  if (!layer.recolor && color) {
    image.width = source instanceof HTMLImageElement ? source.naturalWidth : source.width
    image.height = source instanceof HTMLImageElement ? source.naturalHeight : source.height
    const context = image.getContext('2d')
    if (!context) return source
    context.drawImage(source, 0, 0)
    context.globalCompositeOperation = 'source-in'
    context.fillStyle = color
    context.fillRect(0, 0, image.width, image.height)
  }
  state.tints.set(layer, { source, color: key, image })
  return image
}
