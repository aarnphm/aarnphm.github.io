import type { Element, Nodes } from 'hast'

export type ElementPredicate = (element: Element) => boolean

export function isElement(node: Nodes): node is Element {
  return node.type === 'element'
}

export function childNodes(node: Nodes): readonly Nodes[] {
  return 'children' in node ? node.children : []
}

function propertyValue(element: Element, key: string): unknown {
  return element.properties?.[key]
}

export function attributeValue(element: Element, key: string): string | undefined {
  const value = propertyValue(element, key) ?? propertyValue(element, kebabToCamel(key))
  return typeof value === 'string' || typeof value === 'number' ? `${value}` : undefined
}

function kebabToCamel(value: string): string {
  return value.replace(/-([a-z])/g, (_, char: string) => char.toUpperCase())
}

export function classList(element: Element): string[] {
  const value = propertyValue(element, 'className')
  if (typeof value === 'string') return value.split(/\s+/).filter(Boolean)
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string')
}

export function hasClass(element: Element, className: string): boolean {
  return classList(element).includes(className)
}

export function hasId(element: Element, id: string): boolean {
  return attributeValue(element, 'id') === id
}

export function findElement(node: Nodes, predicate: ElementPredicate): Element | undefined {
  if (isElement(node) && predicate(node)) return node
  for (const child of childNodes(node)) {
    const match = findElement(child, predicate)
    if (match) return match
  }
  return undefined
}

export function findElements(
  node: Nodes,
  predicate: ElementPredicate,
  matches: Element[] = [],
): Element[] {
  if (isElement(node) && predicate(node)) matches.push(node)
  for (const child of childNodes(node)) {
    findElements(child, predicate, matches)
  }
  return matches
}

export function textContent(node: Nodes): string {
  if (node.type === 'text') return node.value
  return childNodes(node).map(textContent).join('')
}

export function normalizeText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

export function firstClassText(root: Nodes, className: string): string | undefined {
  const element = findElement(root, item => hasClass(item, className))
  const text = element ? normalizeText(textContent(element)) : ''
  return text.length > 0 ? text : undefined
}

function isMetaFor(element: Element, key: string): boolean {
  if (element.tagName !== 'meta') return false
  return attributeValue(element, 'property') === key || attributeValue(element, 'name') === key
}

export function metaContent(root: Nodes, key: string): string | undefined {
  const element = findElement(root, item => isMetaFor(item, key))
  const content = element ? attributeValue(element, 'content')?.trim() : undefined
  return content && content.length > 0 ? content : undefined
}

export function metaContents(root: Nodes, key: string): string[] {
  return findElements(root, item => isMetaFor(item, key))
    .map(element => attributeValue(element, 'content')?.trim() ?? '')
    .filter(content => content.length > 0)
}
