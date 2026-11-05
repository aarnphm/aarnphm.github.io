import type { Element as HastElement, RootContent } from 'hast'

const idReferences = new Set([
  'ariadescribedby',
  'arialabelledby',
  'ariacontrols',
  'ariaowns',
  'ariadetails',
  'ariaerrormessage',
  'ariaactivedescendant',
  'ariaflowto',
  'for',
  'htmlfor',
  'headers',
  'list',
  'form',
])

function rewrite(name: string, value: string, ids: ReadonlyMap<string, string>): string {
  const key = name.replaceAll('-', '').toLowerCase()
  if (key === 'id') return ids.get(value) ?? value
  if (idReferences.has(key)) return value.replace(/\S+/g, id => ids.get(id) ?? id)
  if (key === 'href' || key === 'xlinkhref')
    return value.startsWith('#') && ids.has(value.slice(1)) ? `#${ids.get(value.slice(1))}` : value
  if (['fill', 'stroke', 'clippath', 'mask', 'filter', 'style'].includes(key))
    return value.replace(/url\((['"]?)#([^)'"\s]+)\1\)/g, (raw, quote: string, id: string) =>
      ids.has(id) ? `url(${quote}#${ids.get(id)}${quote})` : raw,
    )
  return value
}

/** Scope a complete, newly built tree before its handlers mount; external targets stay unchanged. */
export function scopeHastIds(root: HastElement, prefix: string): void {
  const elements: HastElement[] = []
  function collect(node: RootContent): void {
    if (node.type === 'element') elements.push(node)
    if ('children' in node) for (const child of node.children) collect(child)
  }
  collect(root)
  const ids = new Map<string, string>()
  for (const element of elements) {
    const id = element.properties.id
    if (typeof id === 'string') ids.set(id, `${prefix}-${id}`)
  }
  for (const element of elements)
    for (const [name, value] of Object.entries(element.properties)) {
      if (typeof value === 'string') element.properties[name] = rewrite(name, value, ids)
      else if (Array.isArray(value) && idReferences.has(name.replaceAll('-', '').toLowerCase()))
        element.properties[name] = value.map(id =>
          typeof id === 'string' ? (ids.get(id) ?? id) : id,
        )
    }
}

export function scopeDomIds(root: Element, prefix: string): void {
  const elements = [root, ...root.querySelectorAll('*')]
  const ids = new Map<string, string>()
  for (const element of elements) if (element.id) ids.set(element.id, `${prefix}-${element.id}`)
  for (const element of elements)
    for (const attribute of element.attributes) {
      const value = rewrite(attribute.name, attribute.value, ids)
      if (value !== attribute.value) element.setAttribute(attribute.name, value)
    }
}
