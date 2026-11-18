import type { Element, ElementContent, Root, RootContent } from 'hast'
import type { FullSlug } from '../../util/path'
import type { TranscludeScope } from './transclude-scope'
import { normalizeHastElement, resolveRelative } from '../../util/path'
import { rebaseReferenceElement } from '../../util/references/hast'
import { hasHastClass, readTranscludeTarget, transcludeVisitKey } from '../../util/transclude-props'

const idRefProperties = [
  'ariaDescribedBy',
  'ariaLabelledBy',
  'ariaControls',
  'ariaOwns',
  'ariaDetails',
  'ariaErrorMessage',
  'ariaActiveDescendant',
  'htmlFor',
  'headers',
  'list',
  'form',
]

function walk(node: Root | RootContent, visit: (node: Element) => void): void {
  if (node.type === 'element') visit(node)
  if ('children' in node) for (const child of node.children) walk(child, visit)
}

export function occurrenceContent(
  scope: TranscludeScope,
  nodes: ElementContent[],
  target: FullSlug,
): ElementContent[] | undefined {
  let size = 0
  for (const node of nodes)
    walk(node, element => {
      size++
      if (element.tagName === 'section' && element.properties.dataFootnotes != null)
        scope.features.hasFootnotes = true
      if (element.tagName === 'section' && element.properties.dataReferences != null)
        scope.features.hasReferences = true
    })
  if (size > scope.expansion.remainingNodes) return undefined
  scope.expansion.remainingNodes -= size
  nodes = nodes.map(node =>
    node.type === 'element' ? normalizeHastElement(node, scope.slug, target) : node,
  )
  for (const node of nodes) walk(node, element => rebaseReferenceElement(element, scope.slug))
  const prefix = `transclude-${scope.expansion.occurrence}-`
  const ids = new Map<string, string>()
  for (const node of nodes)
    walk(node, element => {
      const id = element.properties.id
      if (typeof id !== 'string') return
      let next = `${prefix}${id}`
      while (scope.expansion.ids.has(next)) next += '-1'
      scope.expansion.ids.add(next)
      if (!ids.has(id)) ids.set(id, next)
      element.properties.id = next
    })
  for (const node of nodes)
    walk(node, element => {
      for (const property of idRefProperties) {
        const value = element.properties[property]
        if (Array.isArray(value))
          element.properties[property] = value.map(id =>
            typeof id === 'string' ? (ids.get(id) ?? id) : id,
          )
        else if (typeof value === 'string')
          element.properties[property] = value
            .split(/\s+/)
            .map(id => ids.get(id) ?? id)
            .join(' ')
      }
      for (const property of ['href', 'xLinkHref']) {
        const value = element.properties[property]
        if (typeof value !== 'string' || !value.startsWith('#')) continue
        const id = value.slice(1)
        const local = ids.get(id)
        if (local) element.properties[property] = `#${local}`
        else if (element.tagName === 'a')
          element.properties[property] = `${resolveRelative(scope.slug, target)}#${id}`
      }
      for (const property of ['fill', 'stroke', 'clipPath', 'mask', 'filter', 'style']) {
        const value = element.properties[property]
        if (typeof value === 'string')
          element.properties[property] = value.replace(/url\(#([^)]*)\)/g, (raw, id: string) =>
            ids.has(id) ? `url(#${ids.get(id)})` : raw,
          )
      }
    })
  return nodes
}

export function expandTransclusionTree(
  scope: TranscludeScope,
  render: (node: Element) => void,
): void {
  function expand(node: Root | RootContent, depth: number): void {
    if (
      node.type === 'element' &&
      node.tagName === 'blockquote' &&
      hasHastClass(node, 'transclude')
    ) {
      const target = readTranscludeTarget(node)
      if (!target || scope.opts.skipTranscludes) return
      const key = transcludeVisitKey(target)
      if (scope.visited.has(key)) {
        node.properties['data-transclude-status'] = 'cycle'
        return
      }
      if (scope.expansion.occurrence >= 256 || depth >= 32 || scope.expansion.remainingNodes <= 0) {
        node.properties['data-transclude-status'] = 'budget'
        return
      }
      scope.expansion.occurrence++
      scope.visited.add(key)
      const children = node.children
      try {
        render(node)
        if (children !== node.children) {
          node.properties['data-transclude-occurrence'] = scope.expansion.occurrence
          node.properties['data-transclude-origin'] = target.targetSlug
          for (const child of node.children) expand(child, depth + 1)
        }
      } finally {
        scope.visited.delete(key)
      }
      return
    }
    if ('children' in node) for (const child of node.children) expand(child, depth)
  }
  expand(scope.root, 0)
}

export function documentIds(root: Root): Set<string> {
  const ids = new Set<string>()
  walk(root, node => {
    if (typeof node.properties.id === 'string') ids.add(node.properties.id)
  })
  return ids
}

/** An embed can begin inside emphasis; split every phrasing ancestor at its block boundary. */
export function placeTransclusionBlocks(root: Root): void {
  const phrasing = new Set([
    'p',
    'em',
    'strong',
    'a',
    'span',
    'code',
    'mark',
    'sup',
    'del',
    'small',
    'label',
  ])
  const blocks = new Set([
    'blockquote',
    'div',
    'section',
    'ul',
    'ol',
    'p',
    'pre',
    'table',
    'figure',
    'details',
    'dl',
    'hr',
    'article',
    'aside',
    'nav',
    'form',
    'fieldset',
    'header',
    'footer',
    'main',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
  ])
  const interactive = new Set([
    'a',
    'button',
    'select',
    'textarea',
    'details',
    'embed',
    'iframe',
    'label',
  ])
  function flowBoundary(node: ElementContent): boolean {
    return (
      node.type === 'element' &&
      (blocks.has(node.tagName) || (node.tagName === 'a' && node.children.some(flowBoundary)))
    )
  }
  function requiresOwnLinkBoundary(node: ElementContent): boolean {
    if (node.type !== 'element') return false
    if (
      interactive.has(node.tagName) ||
      (node.tagName === 'input' && node.properties.type !== 'hidden') ||
      (['audio', 'video'].includes(node.tagName) &&
        node.properties.controls != null &&
        node.properties.controls !== false) ||
      (node.tagName === 'img' && node.properties.useMap != null) ||
      node.properties['data-transclude-occurrence'] != null ||
      node.properties.dataTranscludeOccurrence != null ||
      (node.tagName === 'blockquote' && hasHastClass(node, 'transclude'))
    )
      return true
    return node.children.some(requiresOwnLinkBoundary)
  }
  function split(node: RootContent): RootContent[] {
    if (node.type !== 'element') return [node]
    node.children = node.children.flatMap(child =>
      split(child).filter((entry): entry is ElementContent => entry.type !== 'doctype'),
    )
    // Anchors have a transparent content model: a Base view row may validly contain divs.
    const boundary = (child: ElementContent): boolean =>
      flowBoundary(child) && (node.tagName !== 'a' || requiresOwnLinkBoundary(child))
    if (!phrasing.has(node.tagName) || !node.children.some(boundary)) return [node]
    const element = node
    const result: ElementContent[] = []
    let children: ElementContent[] = []
    let first = true
    function flush(): void {
      if (!children.length) return
      const properties = { ...element.properties }
      if (!first) delete properties.id
      result.push({ ...element, properties, children })
      children = []
      first = false
    }
    for (const child of node.children) {
      if (boundary(child)) {
        flush()
        result.push(child)
      } else children.push(child)
    }
    flush()
    const id = element.properties.id
    if (
      typeof id === 'string' &&
      !result.some(child => child.type === 'element' && child.properties.id === id)
    ) {
      result.unshift({ type: 'element', tagName: 'span', properties: { id }, children: [] })
    }
    return result
  }
  root.children = root.children.flatMap(split)
}
