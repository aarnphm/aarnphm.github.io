import type { Element, ElementContent, Root, RootContent } from 'hast'
import type { StreamEntry } from '../plugins/transformers/stream'
import { clone } from './clone'
import { normalizeHastElement, type FullSlug } from './path'

const isElement = (node: ElementContent | RootContent): node is Element => node.type === 'element'

const isFootnoteReference = (node: Element): boolean =>
  node.tagName === 'a' && node.properties?.dataFootnoteRef != null

const isCitationReference = (node: Element): boolean =>
  node.tagName === 'a' &&
  typeof node.properties?.href === 'string' &&
  node.properties.href.startsWith('#bib')

const isFootnoteSection = (node: RootContent): boolean =>
  isElement(node) && node.tagName === 'section' && node.properties?.dataFootnotes === ''

const isReferenceSection = (node: RootContent): boolean =>
  isElement(node) && node.tagName === 'section' && node.properties?.dataReferences === ''

function hasNode(nodes: readonly ElementContent[], predicate: (node: Element) => boolean): boolean {
  const stack: ElementContent[] = [...nodes]

  while (stack.length > 0) {
    const node = stack.pop()
    if (!node || !isElement(node)) continue
    if (predicate(node)) return true
    stack.push(...node.children)
  }

  return false
}

function hasContentNode(
  entries: readonly StreamEntry[],
  predicate: (node: Element) => boolean,
): boolean {
  return entries.some(entry => hasNode(entry.content, predicate))
}

function cloneStreamEntry(entry: StreamEntry): StreamEntry {
  return { ...entry, metadata: clone(entry.metadata), content: clone(entry.content) }
}

export function cloneStreamEntries(entries: readonly StreamEntry[]): StreamEntry[] {
  return entries.map(cloneStreamEntry)
}

export function rebaseStreamEntries(
  entries: readonly StreamEntry[],
  targetSlug: FullSlug,
  sourceSlug: FullSlug,
): StreamEntry[] {
  return entries.map(entry => ({
    ...entry,
    metadata: clone(entry.metadata),
    content: entry.content.map(node =>
      isElement(node) ? normalizeHastElement(node, targetSlug, sourceSlug) : clone(node),
    ),
  }))
}

export function buildStreamRouteTree(entries: readonly StreamEntry[], sourceTree: Root): Root {
  const needsFootnotes = hasContentNode(entries, isFootnoteReference)
  const needsReferences = hasContentNode(entries, isCitationReference)
  const children: RootContent[] = entries.flatMap(entry => entry.content)

  if (needsFootnotes || needsReferences) {
    for (const child of sourceTree.children) {
      if (
        (needsFootnotes && isFootnoteSection(child)) ||
        (needsReferences && isReferenceSection(child))
      ) {
        children.push(child)
      }
    }
  }

  return { type: 'root', children }
}
