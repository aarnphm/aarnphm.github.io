import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { QuartzComponentProps } from '../../types/component'
import { usesApparatus } from '../../util/apparatus'
import { renderDataFor } from '../../util/ctx'
import { FullSlug, resolveRelative, simplifySlug } from '../../util/path'
import { Apparatus, firstAnchor } from './apparatus'
import { anchorIn, isElement, remove, takeNotes } from './apparatus-notes'

type Ctx = QuartzComponentProps['ctx']

const LABELS = {
  notes: 'notes',
  bib: 'bibliographie',
  seealso: 'see also',
  cite: 'cited by',
  listing: 'listing',
}
type Kind = keyof typeof LABELS
const ORDER: Record<Kind, number> = { notes: 0, bib: 1, seealso: 2, cite: 3, listing: 4 }
// the margin names the first few notes that cite this one
const CITED_BY = 5

/**
 * The bibliography, each reference in the group of the block that first cites it (the entry keeps
 * the reference's id, so every citation of it still lands there); references no block cites stay in
 * the list at the end, and an emptied list leaves with its section.
 */
function takeBibliography(tree: Root, apparatus: Apparatus) {
  let section: Element | undefined
  let list: Element | undefined
  const refs = new Map<string, Element>()
  const cites = new Map<Element, string[]>()
  const visit = (node: Element | Root, path: Element[]) => {
    for (const child of node.children) {
      if (!isElement(child)) continue
      if (child.tagName === 'section' && child.properties?.dataReferences != null) {
        section = child
        list = child.children.find((c): c is Element => isElement(c) && c.tagName === 'ul')
        for (const item of list?.children.filter(isElement) ?? [])
          if (item.properties?.id) refs.set(String(item.properties.id), item)
        continue
      }
      const href = child.tagName === 'a' ? String(child.properties?.href ?? '') : ''
      const block = anchorIn(path)
      if (href.startsWith('#bib-') && block) {
        const keys = cites.get(block)
        if (keys) keys.push(href.slice(1))
        else cites.set(block, [href.slice(1)])
      }
      visit(child, [...path, child])
    }
  }
  visit(tree, [])
  // a citation inside a note already taken to the margin counts for the note's block
  const scan = (node: ElementContent, block: Element) => {
    if (!isElement(node)) return
    const href = node.tagName === 'a' ? String(node.properties?.href ?? '') : ''
    if (href.startsWith('#bib-')) {
      const keys = cites.get(block)
      if (keys) keys.push(href.slice(1))
      else cites.set(block, [href.slice(1)])
    }
    for (const child of node.children) scan(child, block)
  }
  for (const [block, entries] of apparatus.pending)
    for (const entry of entries) scan(entry.body, block)
  if (!section || !list) return
  const taken = new Set<string>()
  for (const [block, keys] of cites) {
    const items: Element[] = []
    for (const key of keys) {
      const ref = refs.get(key)
      if (!ref || taken.has(key)) continue
      taken.add(key)
      items.push(h('span.fr-gloss', { id: key }, ref.children))
      remove(list, ref)
    }
    if (items.length > 0)
      apparatus.put(block, { kind: 'bib', label: LABELS.bib, body: h('span.fr-margin', items) })
  }
  if (!list.children.some(isElement)) {
    const parent = (function find(node: Element | Root): Element | Root | undefined {
      for (const child of node.children) {
        if (!isElement(child)) continue
        if (child === section) return node
        const hit = find(child)
        if (hit) return hit
      }
    })(tree)
    if (parent) remove(parent as Element, section)
  }
}

const linkLine = (href: string, label: string, slug?: string) =>
  h('span.fr-gloss', [h('a.internal', { href, ...(slug ? { dataSlug: slug } : {}) }, label)])

/** Code listings with a title: the title becomes the entry, the listing's caption keeps the number. */
function takeListings(tree: Root, apparatus: Apparatus) {
  const visit = (node: Element | Root) => {
    for (const child of node.children) {
      if (!isElement(child)) continue
      if (child.tagName === 'figcaption' && child.properties?.dataRehypePrettyCodeTitle != null) {
        if (!toString(child).trim()) continue
        const title = child.children
        child.children = []
        apparatus.put(child, {
          kind: 'listing',
          label: LABELS.listing,
          body: h('span.fr-margin', [h('span.fr-meta', title)]),
        })
        continue
      }
      visit(child)
    }
  }
  visit(tree)
}

/**
 * The apparatus of a technical note (`usesApparatus`), on the /fr contract (apparatus.ts): notes per
 * block, then `seealso`/`related` links and the notes that cite this one on the first block, then
 * code listing titles on their own caption.
 */
export function renderTechnicalApparatus(
  tree: Root,
  componentData: QuartzComponentProps,
  ctx: Ctx,
) {
  const { fileData, allFiles } = componentData
  if (!usesApparatus(fileData)) return
  const slug = fileData.slug as FullSlug
  const apparatus = new Apparatus()
  takeNotes(tree, apparatus, { kind: 'notes', label: LABELS.notes, sidenotes: true })
  takeBibliography(tree, apparatus)
  takeListings(tree, apparatus)

  const first = firstAnchor(tree)
  if (first) {
    const data = renderDataFor(ctx, allFiles)
    const titleOf = (target: FullSlug, alias?: string) =>
      alias?.trim() || data.bySlug.get(target)?.frontmatter?.title || target.split('/').at(-1)!
    const links = [
      ...(fileData.frontmatterLinks?.seealso ?? []),
      ...(fileData.frontmatterLinks?.related ?? []),
    ]
    const seen = new Set<string>()
    const seeAlso = links.flatMap(link => {
      if (seen.has(link.slug)) return []
      seen.add(link.slug)
      if (link.slug.startsWith('@')) {
        const key = link.slug.slice(1)
        return [linkLine(`#bib-${key.toLowerCase()}`, link.alias?.trim() || key)]
      }
      return [linkLine(resolveRelative(slug, link.slug), titleOf(link.slug, link.alias), link.slug)]
    })
    if (seeAlso.length > 0)
      apparatus.put(first, {
        kind: 'seealso',
        label: LABELS.seealso,
        body: h('span.fr-margin', seeAlso),
      })

    // the first few notes that cite this one; the rest is a count that opens the backlinks table at
    // the end of the page
    const citing = data.backlinksBySlug.get(simplifySlug(slug)) ?? []
    if (citing.length > 0) {
      const lines = citing
        .slice(0, CITED_BY)
        .map(file =>
          linkLine(
            resolveRelative(slug, file.slug as FullSlug),
            titleOf(file.slug as FullSlug),
            file.slug,
          ),
        )
      if (citing.length > CITED_BY)
        lines.push(
          h('span.fr-gloss.fr-meta', [
            h('a', { href: '#backlinks-label' }, `and ${citing.length - CITED_BY} more`),
          ]),
        )
      apparatus.put(first, { kind: 'cite', label: LABELS.cite, body: h('span.fr-margin', lines) })
    }
  }

  apparatus.hang(tree, kind => ORDER[kind as Kind])
}
