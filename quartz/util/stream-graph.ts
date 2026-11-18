import type { Element, Nodes, Root as HastRoot } from 'hast'
import type { StreamEntry } from '../plugins/transformers/stream'
import type { InquiryLinkKind, InquiryMapData, InquiryNode, InquiryPreview } from './inquiry-map'
import { descriptionToPlainText } from './description'
import { parseGraphMemberships, type GraphMembership } from './graph-membership'
import { childNodes, firstClassText, hasClass, isElement, normalizeText } from './hast-query'
import {
  buildStreamEntryPathFromIso,
  formatStreamDate,
  isDraftEntry,
  isRestrictedEntry,
} from './stream'

const SNIPPET_MAX = 180
const SKIP_TAGS = new Set(['audio', 'video', 'svg', 'button', 'script', 'style', 'noscript'])

const isPublished = (entry: StreamEntry): boolean =>
  !isDraftEntry(entry) && !isRestrictedEntry(entry)

const entryAt = (entry: StreamEntry, membership: GraphMembership): string | undefined => {
  if (membership.at) return membership.at
  if (entry.date) return entry.date
  if (typeof entry.timestamp === 'number') return new Date(entry.timestamp).toISOString()
  return undefined
}

const entryLabel = (entry: StreamEntry, membership: GraphMembership): string => {
  if (membership.label) return membership.label
  const title = entry.title?.trim()
  if (title) return title
  const description = descriptionToPlainText(entry.description, 'stream')
  if (description) return description.length > 40 ? `${description.slice(0, 39)}…` : description
  const at = entryAt(entry, membership)
  if (at) {
    const date = new Date(at)
    if (!Number.isNaN(date.getTime())) return date.toISOString().slice(0, 10)
  }
  return entry.id
}

const entryHref = (entry: StreamEntry): string => {
  const path = buildStreamEntryPathFromIso(entry.date, entry.id)
  return path ? path.replace(/^\//, '') : ''
}

const tagsOf = (entry: StreamEntry): string[] => {
  const tags = entry.metadata.tags
  if (!Array.isArray(tags)) return []
  return tags.flatMap(tag => {
    if (typeof tag !== 'string') return []
    const trimmed = tag.trim()
    return trimmed ? [trimmed] : []
  })
}

const membershipsFor = (entry: StreamEntry, graphId: string): GraphMembership[] =>
  parseGraphMemberships(entry.metadata.graphs).filter(m => m.id === graphId)

const isPlayerChrome = (el: Element): boolean =>
  hasClass(el, 'audio-player') ||
  hasClass(el, 'ap-row') ||
  hasClass(el, 'ap-time') ||
  hasClass(el, 'ap-bars') ||
  hasClass(el, 'ap-play') ||
  hasClass(el, 'ap-repeat') ||
  hasClass(el, 'ap-caption') ||
  el.tagName === 'figcaption'

const clipSnippet = (text: string): string | undefined => {
  const cleaned = normalizeText(text)
  if (!cleaned) return undefined
  if (cleaned.length <= SNIPPET_MAX) return cleaned
  return `${cleaned.slice(0, SNIPPET_MAX - 1)}…`
}

const entryMedia = (root: HastRoot): string | undefined =>
  firstClassText(root, 'ap-caption') ?? firstClassText(root, 'figure-caption')

const entryBodyText = (root: HastRoot): string => {
  const parts: string[] = []
  const walk = (node: Nodes) => {
    if (node.type === 'text') {
      const value = node.value.trim()
      if (value) parts.push(value)
      return
    }
    if (!isElement(node)) {
      for (const child of childNodes(node)) walk(child)
      return
    }
    if (SKIP_TAGS.has(node.tagName) || isPlayerChrome(node)) return
    if (node.tagName === 'img' || node.tagName === 'iframe') return
    for (const child of childNodes(node)) walk(child)
  }
  for (const child of root.children) walk(child)
  return parts.join(' ')
}

const entryPreview = (entry: StreamEntry, membership: GraphMembership): InquiryPreview => {
  const title = entry.title?.trim() || entryLabel(entry, membership)
  const description = descriptionToPlainText(entry.description, 'stream') || undefined
  const at = entryAt(entry, membership)
  const date = at ? (formatStreamDate(at) ?? at.slice(0, 10)) : undefined
  const tags = tagsOf(entry)
  const href = entryHref(entry)
  const root: HastRoot = { type: 'root', children: entry.content }
  const media = entryMedia(root)
  const snippet = clipSnippet(entryBodyText(root))
  const unique = (value: string | undefined, ...seen: (string | undefined)[]) =>
    value && !seen.some(item => item && item === value) ? value : undefined
  return {
    title,
    date,
    description: unique(description, title),
    media: unique(media, description, title),
    snippet: unique(snippet, description, media, title),
    tags: tags.length > 0 ? tags : undefined,
    href: href || undefined,
  }
}

export const listStreamGraphIds = (entries: StreamEntry[]): string[] => {
  const ids = new Set<string>()
  for (const entry of entries) {
    if (!isPublished(entry)) continue
    for (const m of parseGraphMemberships(entry.metadata.graphs)) ids.add(m.id)
  }
  return [...ids].sort((a, b) => a.localeCompare(b))
}

export interface StreamGraphSummary {
  id: string
  entries: number
  earliest?: string
  latest?: string
}

export const listStreamGraphSummaries = (entries: StreamEntry[]): StreamGraphSummary[] => {
  const byId = new Map<string, { members: Set<string>; times: number[] }>()
  for (const entry of entries) {
    if (!isPublished(entry)) continue
    for (const membership of parseGraphMemberships(entry.metadata.graphs)) {
      let row = byId.get(membership.id)
      if (!row) {
        row = { members: new Set(), times: [] }
        byId.set(membership.id, row)
      }
      row.members.add(entry.id)
      const at = entryAt(entry, membership)
      if (!at) continue
      const time = Date.parse(at)
      if (!Number.isNaN(time)) row.times.push(time)
    }
  }
  return [...byId.entries()]
    .map(([id, row]) => {
      const times = row.times.sort((a, b) => a - b)
      return {
        id,
        entries: row.members.size,
        earliest: times.length > 0 ? new Date(times[0]).toISOString() : undefined,
        latest: times.length > 0 ? new Date(times[times.length - 1]).toISOString() : undefined,
      }
    })
    .sort((a, b) => a.id.localeCompare(b.id))
}

export const streamEntryGraphIds = (entry: StreamEntry): string[] =>
  parseGraphMemberships(entry.metadata.graphs).map(m => m.id)

interface MemberRow {
  entry: StreamEntry
  membership: GraphMembership
  at?: string
  sort: number
  preview: InquiryPreview
}

/** Build InquiryMapData for one stream graph id. Tag→hub is on by default. */
export function buildStreamInquiryGraph(
  graphId: string,
  entries: StreamEntry[],
): InquiryMapData | null {
  const members: MemberRow[] = []
  for (const entry of entries) {
    if (!isPublished(entry)) continue
    for (const membership of membershipsFor(entry, graphId)) {
      const at = entryAt(entry, membership)
      const sort = at ? Date.parse(at) : Number.POSITIVE_INFINITY
      members.push({
        entry,
        membership,
        at,
        sort: Number.isNaN(sort) ? Number.POSITIVE_INFINITY : sort,
        preview: entryPreview(entry, membership),
      })
    }
  }
  if (members.length === 0) return null
  members.sort((a, b) => a.sort - b.sort || a.entry.id.localeCompare(b.entry.id))

  const nodes: InquiryNode[] = []
  const index = new Map<string, number>()
  const add = (node: InquiryNode): number => {
    const known = index.get(node.id)
    if (known !== undefined) {
      if (node.at && !nodes[known].at) nodes[known].at = node.at
      if (node.label && nodes[known].label === nodes[known].id) nodes[known].label = node.label
      if (node.previews?.length) {
        const existing = nodes[known].previews ?? []
        const seen = new Set(existing.map(p => p.href ?? p.title))
        for (const preview of node.previews) {
          const key = preview.href ?? preview.title
          if (key && seen.has(key)) continue
          if (key) seen.add(key)
          existing.push(preview)
        }
        nodes[known].previews = existing
      }
      return known
    }
    index.set(node.id, nodes.length)
    nodes.push(node)
    return nodes.length - 1
  }

  const PRIORITY: Record<InquiryLinkKind, number> = {
    edge: 6,
    seealso: 5,
    card: 4,
    wikilink: 3,
    hub: 2,
    tag: 1,
    chrono: 0,
  }
  const linkSet = new Map<string, [number, number, InquiryLinkKind]>()
  const link = (a: number, b: number, kind: InquiryLinkKind) => {
    if (a === b) return
    const key = a < b ? `${a}|${b}` : `${b}|${a}`
    const known = linkSet.get(key)
    if (known && PRIORITY[known[2]] >= PRIORITY[kind]) return
    linkSet.set(key, [a, b, kind])
  }

  const allPreviews = members.map(row => row.preview)
  const graphHub = add({
    id: `hub:${graphId}`,
    label: graphId,
    cls: 'mvt',
    kind: 'hub',
    w: 0,
    href: '',
    previews: allPreviews,
  })

  const tagHubs = new Map<string, number>()
  const tagMembers = new Map<string, InquiryPreview[]>()
  for (const row of members) {
    for (const tag of tagsOf(row.entry)) {
      const list = tagMembers.get(tag) ?? []
      list.push(row.preview)
      tagMembers.set(tag, list)
      if (tagHubs.has(tag)) continue
      tagHubs.set(
        tag,
        add({
          id: `hub:${tag}`,
          label: tag,
          cls: 'mvt',
          kind: 'hub',
          w: 0,
          href: '',
          previews: tagMembers.get(tag),
        }),
      )
    }
  }
  for (const [tag, hub] of tagHubs) {
    nodes[hub].previews = tagMembers.get(tag)
  }

  const segmentIndexes: number[] = []
  for (const row of members) {
    const nodeId = `stream:${row.entry.id}`
    const seg = add({
      id: nodeId,
      label: entryLabel(row.entry, row.membership),
      cls: row.membership.cls ?? 'pg',
      kind: row.membership.kind ?? 'entry',
      w: 1,
      href: entryHref(row.entry),
      at: row.at,
      previews: [row.preview],
    })
    segmentIndexes.push(seg)
    link(graphHub, seg, 'hub')
    for (const tag of tagsOf(row.entry)) {
      const hub = tagHubs.get(tag)
      if (hub !== undefined) link(hub, seg, 'tag')
    }
    for (const rel of row.membership.links ?? []) {
      const target =
        index.get(rel.to) ??
        index.get(`hub:${rel.to}`) ??
        index.get(`stream:${rel.to}`) ??
        add({
          id: rel.to.startsWith('hub:') || rel.to.includes('/') ? rel.to : `hub:${rel.to}`,
          label: rel.to.replace(/^hub:/, ''),
          cls: 'mvt',
          kind: 'hub',
          w: 0,
          href: rel.to.includes('/') ? rel.to.replace(/^\//, '') : '',
        })
      link(seg, target, rel.as)
    }
  }

  for (let i = 0; i < segmentIndexes.length - 1; i++) {
    link(segmentIndexes[i], segmentIndexes[i + 1], 'chrono')
  }

  const base = nodes.map(n => n.w)
  const links = [...linkSet.values()]
  for (const [a, b] of links) {
    for (const [m, c] of [
      [a, b],
      [b, a],
    ] as const) {
      if (nodes[m].cls === 'mvt' && nodes[c].cls !== 'mvt') nodes[m].w += base[c]
    }
  }

  return { nodes, links }
}

export function buildStreamInquiryGraphs(
  entries: StreamEntry[],
): { id: string; data: InquiryMapData }[] {
  return listStreamGraphIds(entries).flatMap(id => {
    const data = buildStreamInquiryGraph(id, entries)
    return data ? [{ id, data }] : []
  })
}

export function streamGraphsIndexSlug(): string {
  return 'stream/graphs'
}

export function streamGraphsIndexHref(): string {
  return `/${streamGraphsIndexSlug()}`
}

export function streamGraphSlug(graphId: string): string {
  return `stream/graphs/${encodeURIComponent(graphId)}`
}

export function streamGraphHref(graphId: string, fromEntryId?: string): string {
  const base = `/${streamGraphSlug(graphId)}`
  if (!fromEntryId) return base
  return `${base}?from=${encodeURIComponent(fromEntryId)}`
}
