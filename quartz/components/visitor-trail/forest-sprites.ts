export type ForestSprite = { id: string; bottom: number; shape: string; render: () => string }

// Depth sorting moves sprites out of their generation holders. Keep their identity
// here so a population change never needs to bring the whole forest back first.
export function createForestSprites(document: Document) {
  const view = document.defaultView
  if (!view) throw new Error('Forest sprites need their scene window')
  const entries = new Map<string, { element: SVGElement; shape: string; bottom: number }>()
  const parser = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  return {
    update(sprite: ForestSprite, owner: Element) {
      const previous = entries.get(sprite.id)
      if (previous?.element.isConnected && previous.shape === sprite.shape) return false
      parser.innerHTML = sprite.render()
      const next = parser.firstElementChild
      if (!(next instanceof view.SVGElement))
        throw new Error('A forest sprite needs an SVG element')
      const element = previous?.element.isConnected ? previous.element : next
      if (element === next) {
        let before: SVGElement | null = null
        let bottom = Infinity
        for (const entry of entries.values())
          if (
            entry.element.parentElement === owner &&
            entry.bottom > sprite.bottom &&
            entry.bottom < bottom
          ) {
            before = entry.element
            bottom = entry.bottom
          }
        owner.insertBefore(element, before)
      } else {
        for (const attribute of next.attributes)
          if (element.getAttribute(attribute.name) !== attribute.value)
            element.setAttribute(attribute.name, attribute.value)
        element.replaceChildren(...next.childNodes)
      }
      entries.set(sprite.id, { element, shape: sprite.shape, bottom: sprite.bottom })
      return true
    },
    retain(ids: Set<string>) {
      for (const [id, entry] of entries)
        if (!ids.has(id)) {
          entry.element.remove()
          entries.delete(id)
        }
    },
    clear() {
      entries.clear()
    },
  }
}
