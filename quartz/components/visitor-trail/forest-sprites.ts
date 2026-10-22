export type ForestSprite = { id: string; bottom: number; shape: string; render: () => string }

// Depth sorting moves sprites out of their generation holders. Keep their identity
// here so a population change never needs to bring the whole forest back first.
export function createForestSprites(document: Document) {
  const view = document.defaultView
  if (!view) throw new Error('Forest sprites need their scene window')
  const entries = new Map<
    string,
    { element: SVGElement; shape: string; bottom: number; authored: Set<string> }
  >()
  const parser = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  const needsUpdate = (sprite: Pick<ForestSprite, 'id' | 'shape'>) => {
    const previous = entries.get(sprite.id)
    return !previous?.element.isConnected || previous.shape !== sprite.shape
  }
  return {
    needsUpdate,
    update(sprite: ForestSprite, owner: Element) {
      if (!needsUpdate(sprite)) return false
      const previous = entries.get(sprite.id)
      parser.innerHTML = sprite.render()
      const next = parser.firstElementChild
      if (!(next instanceof view.SVGElement))
        throw new Error('A forest sprite needs an SVG element')
      const authored = new Set(Array.from(next.attributes, attribute => attribute.name))
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
        // Remove attributes owned by the previous sprite, retaining state attached by controllers.
        for (const name of previous?.authored ?? [])
          if (!authored.has(name)) element.removeAttribute(name)
        for (const attribute of next.attributes)
          if (element.getAttribute(attribute.name) !== attribute.value)
            element.setAttribute(attribute.name, attribute.value)
        element.replaceChildren(...next.childNodes)
      }
      entries.set(sprite.id, { element, shape: sprite.shape, bottom: sprite.bottom, authored })
      return true
    },
    remove(id: string) {
      entries.get(id)?.element.remove()
      entries.delete(id)
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
