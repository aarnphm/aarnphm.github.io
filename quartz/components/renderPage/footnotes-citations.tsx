import type { Element, ElementContent, Root, RootContent, Text } from 'hast'
import { h } from 'hastscript'

export const getFootnotesList = (node: Element) =>
  node.children.find(
    (child): child is Element => child.type === 'element' && child.tagName === 'ol',
  )!

export const getBibList = (node: Element) =>
  node.children.find(
    (child): child is Element => child.type === 'element' && child.tagName === 'ul',
  )!

export const isTrailingNoteSection = (node: ElementContent): boolean =>
  node.type === 'element' &&
  node.tagName === 'section' &&
  (node.properties.dataFootnotes != null || node.properties.dataReferences != null)

function walk(node: Root | RootContent, callback: (node: Element) => void): void {
  if (node.type === 'element') callback(node)
  if ('children' in node) for (const child of node.children) walk(child, callback)
}

function removeSections(root: Root, sections: Set<Element>): void {
  function prune(node: Root | RootContent): void {
    if (!('children' in node)) return
    node.children = node.children.filter(child => child.type !== 'element' || !sections.has(child))
    for (const child of node.children) prune(child)
  }
  prune(root)
}

function noteSections(root: Root, property: 'dataFootnotes' | 'dataReferences'): Element[] {
  const sections: Element[] = []
  walk(root, node => {
    if (node.tagName === 'section' && node.properties[property] != null) sections.push(node)
  })
  const direct = new Set(root.children)
  return [
    ...sections.filter(section => direct.has(section)),
    ...sections.filter(section => !direct.has(section)),
  ]
}

function mergeSectionLabels(root: Root, sections: Element[]): string | undefined {
  const labels = sections.flatMap(section =>
    section.children.flatMap(node =>
      node.type === 'element' &&
      /^h[1-6]$/.test(node.tagName) &&
      typeof node.properties.id === 'string'
        ? [node.properties.id]
        : [],
    ),
  )
  const label = labels[0]
  if (!label) return undefined
  const removed = new Set(labels.slice(1))
  walk(root, node => {
    for (const property of ['href', 'xLinkHref']) {
      const value = node.properties[property]
      if (typeof value === 'string' && value.startsWith('#') && removed.has(value.slice(1)))
        node.properties[property] = '#' + label
    }
    for (const property of ['ariaDescribedBy', 'ariaLabelledBy', 'ariaControls']) {
      const value = node.properties[property]
      if (Array.isArray(value))
        node.properties[property] = value.map(id =>
          typeof id === 'string' && removed.has(id) ? label : id,
        )
      else if (typeof value === 'string')
        node.properties[property] = value
          .split(/\s+/)
          .map(id => (removed.has(id) ? label : id))
          .join(' ')
    }
  })
  return label
}

export function mergeReferences(root: Root): void {
  const sections = noteSections(root, 'dataReferences')
  if (sections.length < 2) return
  const entries = new Map<string, ElementContent>()
  for (const section of sections) {
    for (const entry of getBibList(section)?.children ?? []) {
      if (entry.type === 'element' && typeof entry.properties.id === 'string')
        entries.set(entry.properties.id, entry)
    }
  }
  const section = sections[0]
  const list = getBibList(section)
  if (!list) return
  mergeSectionLabels(root, sections)
  list.children = [...entries.values()]
  removeSections(root, new Set(sections))
  root.children.push(section)
}

type Footnote = { id: string; number: number; references: Element[]; entry: Element }

export function mergeFootnotes(root: Root): void {
  const sections = noteSections(root, 'dataFootnotes')
  const section = sections[0]
  if (!section) return
  const list = getFootnotesList(section)
  if (!list) return
  const definitions = new Map<string, Element>()
  for (const section of sections) {
    for (const entry of getFootnotesList(section)?.children ?? []) {
      if (entry.type === 'element' && typeof entry.properties.id === 'string')
        definitions.set(entry.properties.id, entry)
    }
  }
  const notes = new Map<string, Footnote>()
  const reservedIds = new Set<string>()
  const owners = new Map<string, Element[]>()
  const footnoteNodes = new Set(definitions.values())
  walk(root, node => {
    if (node.tagName === 'a' && node.properties.dataFootnoteRef != null) footnoteNodes.add(node)
    const id = node.properties.id
    if (typeof id === 'string') {
      reservedIds.add(id)
      const candidates = owners.get(id) ?? []
      candidates.push(node)
      owners.set(id, candidates)
    }
  })
  function allocate(id: string): string {
    while (reservedIds.has(id)) id += '-1'
    reservedIds.add(id)
    return id
  }
  function retainOrAllocate(node: Element, fallback: string): string {
    const id = node.properties.id
    if (typeof id !== 'string' || id.length === 0) return allocate(fallback)
    const candidates = owners.get(id) ?? []
    const retained = candidates.find(candidate => !footnoteNodes.has(candidate)) ?? candidates[0]
    return retained === node ? id : allocate(id)
  }
  const labelId = mergeSectionLabels(root, sections)
  walk(root, node => {
    if (
      node.tagName !== 'a' ||
      node.properties.dataFootnoteRef == null ||
      typeof node.properties.href !== 'string'
    )
      return
    const originalId = node.properties.href.replace(/^#/, '')
    const entry = definitions.get(originalId)
    if (!entry) return
    let note = notes.get(originalId)
    if (!note) {
      const number = notes.size + 1
      note = { id: retainOrAllocate(entry, `fn-${number}`), number, references: [], entry }
      notes.set(originalId, note)
    }
    note.references.push(node)
    node.properties.href = `#${note.id}`
    node.properties.id = retainOrAllocate(node, `fnref-${note.number}-${note.references.length}`)
    if (labelId) node.properties.ariaDescribedBy = [labelId]
    const number = note.number
    function replaceNumber(child: ElementContent): void {
      if (child.type === 'text') child.value = String(number)
      else if ('children' in child) for (const nested of child.children) replaceNumber(nested)
    }
    for (const child of node.children) replaceNumber(child)
  })
  if (notes.size === 0) return
  for (const note of notes.values()) {
    note.entry.properties.id = note.id
    function removeBacklinks(node: Element): void {
      node.children = node.children.filter(
        child =>
          child.type !== 'element' ||
          child.tagName !== 'a' ||
          child.properties.dataFootnoteBackref == null,
      )
      for (const child of node.children) if (child.type === 'element') removeBacklinks(child)
    }
    removeBacklinks(note.entry)
    let container = note.entry
    for (const child of note.entry.children) if (child.type === 'element') container = child
    for (const [ordinal, ref] of note.references.entries()) {
      container.children.push({ type: 'text', value: ' ' } satisfies Text)
      container.children.push(
        h(
          'a',
          { href: `#${ref.properties.id}`, dataFootnoteBackref: '', ariaLabel: 'Back to content' },
          `↩︎${ordinal === 0 ? '' : ordinal + 1}`,
        ),
      )
    }
  }
  list.children = [...notes.values()].map(note => note.entry)
  removeSections(root, new Set(sections))
  root.children.push(section)
}
