export type ForestSprite = { id: string; bottom: number; shape: string; render: () => string }

// Depth sorting moves sprites out of their generation holders. Keep their identity
// here so a population change never needs to bring the whole forest back first.
export function createForestSprites(document: Document) {
  const view = document.defaultView
  if (!view) throw new Error('Forest sprites need their scene window')
  const entries = new Map<string, { element: SVGElement; shape: string; bottom: number }>()
  const parser = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  const attributes = new WeakMap<SVGElement, Set<string>>()
  const styles = new WeakMap<SVGElement, Set<string>>()
  const remember = (element: SVGElement) => {
    attributes.set(element, new Set(element.getAttributeNames()))
    styles.set(element, new Set(element.style))
    for (const child of element.children) if (child instanceof view.SVGElement) remember(child)
  }
  const reconcile = (element: SVGElement, next: SVGElement) => {
    // Controllers own wind, lighting and motion properties absent from generated markup.
    for (const name of attributes.get(element) ?? [])
      if (name !== 'style' && !next.hasAttribute(name)) element.removeAttribute(name)
    for (const attribute of next.attributes)
      if (attribute.name !== 'style' && element.getAttribute(attribute.name) !== attribute.value)
        element.setAttribute(attribute.name, attribute.value)
    const nextStyles = new Set(next.style)
    for (const name of styles.get(element) ?? [])
      if (!nextStyles.has(name)) element.style.removeProperty(name)
    for (const name of nextStyles) {
      const value = next.style.getPropertyValue(name)
      const priority = next.style.getPropertyPriority(name)
      if (
        element.style.getPropertyValue(name) !== value ||
        element.style.getPropertyPriority(name) !== priority
      )
        element.style.setProperty(name, value, priority)
    }
    if (element.getAttribute('style') === '') element.removeAttribute('style')
    attributes.set(element, new Set(next.getAttributeNames()))
    styles.set(element, nextStyles)
    reconcileChildren(element, next)
  }
  const reconcileChildren = (element: SVGElement, next: SVGElement) => {
    let index = 0
    for (const child of Array.from(next.childNodes)) {
      const previous = element.childNodes[index++]
      if (
        previous instanceof view.SVGElement &&
        child instanceof view.SVGElement &&
        previous.localName === child.localName &&
        previous.getAttribute('class') === child.getAttribute('class')
      )
        reconcile(previous, child)
      else if (!previous || !previous.isEqualNode(child)) {
        if (child instanceof view.SVGElement) remember(child)
        if (previous) element.replaceChild(child, previous)
        else element.append(child)
      }
    }
    while (element.childNodes.length > index) element.lastChild?.remove()
  }
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
      const element = previous?.element.isConnected ? previous.element : next
      if (element === next) {
        remember(element)
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
        // Keep matching texture boundaries and their computed styles when only geometry changes.
        reconcile(element, next)
      }
      entries.set(sprite.id, { element, shape: sprite.shape, bottom: sprite.bottom })
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
