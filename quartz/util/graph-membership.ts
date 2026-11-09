import type { InquiryClass, InquiryLinkKind } from './inquiry-map'
import { isRecord } from './type-guards'

/** Short or rich membership in a named thematic graph (`graphs:`). */
export interface GraphMembership {
  id: string
  label?: string
  kind?: string
  cls?: InquiryClass
  at?: string
  links?: GraphMembershipLink[]
  edges?: string
}

export interface GraphMembershipLink {
  to: string
  as: InquiryLinkKind
}

export interface GraphHost {
  id: string
  title?: string
  chronology: string | false
  hubs: { id: string; label?: string; cls?: InquiryClass }[]
  edges?: string
  members: 'scan' | 'stream' | string
}

const CLASSES = new Set<InquiryClass>(['theme', 'mvt', 'pg'])
const LINK_KINDS = new Set<InquiryLinkKind>([
  'edge',
  'seealso',
  'card',
  'wikilink',
  'chrono',
  'hub',
  'tag',
])

const asClass = (value: unknown): InquiryClass | undefined =>
  typeof value === 'string' && CLASSES.has(value as InquiryClass)
    ? (value as InquiryClass)
    : undefined

const asLinkKind = (value: unknown, fallback: InquiryLinkKind): InquiryLinkKind =>
  typeof value === 'string' && LINK_KINDS.has(value as InquiryLinkKind)
    ? (value as InquiryLinkKind)
    : fallback

const parseLink = (item: unknown): GraphMembershipLink | null => {
  if (typeof item === 'string') {
    const to = item.trim()
    return to ? { to, as: 'hub' } : null
  }
  if (!isRecord(item) || typeof item.to !== 'string') return null
  const to = item.to.trim()
  if (!to) return null
  return { to, as: asLinkKind(item.as, 'hub') }
}

const parseMembershipItem = (item: unknown): GraphMembership | null => {
  if (typeof item === 'string') {
    const id = item.trim()
    return id ? { id } : null
  }
  if (!isRecord(item) || typeof item.id !== 'string') return null
  const id = item.id.trim()
  if (!id) return null
  const links = Array.isArray(item.links)
    ? item.links.flatMap(link => {
        const parsed = parseLink(link)
        return parsed ? [parsed] : []
      })
    : undefined
  return {
    id,
    label: typeof item.label === 'string' ? item.label.trim() : undefined,
    kind: typeof item.kind === 'string' ? item.kind.trim() : undefined,
    cls: asClass(item.cls),
    at: typeof item.at === 'string' ? item.at.trim() : undefined,
    links: links && links.length > 0 ? links : undefined,
    edges: typeof item.edges === 'string' ? item.edges : undefined,
  }
}

/** Parse `graphs:` from stream meta or note frontmatter. */
export function parseGraphMemberships(value: unknown): GraphMembership[] {
  if (typeof value === 'string') {
    const id = value.trim()
    return id ? [{ id }] : []
  }
  if (!Array.isArray(value)) return []
  return value.flatMap(item => {
    const parsed = parseMembershipItem(item)
    return parsed ? [parsed] : []
  })
}

export function parseGraphHost(value: unknown): GraphHost | null {
  if (!isRecord(value) || typeof value.id !== 'string') return null
  const id = value.id.trim()
  if (!id) return null
  const chronology =
    value.chronology === false
      ? false
      : typeof value.chronology === 'string'
        ? value.chronology.trim() || 'date'
        : 'date'
  const hubs = Array.isArray(value.hubs)
    ? value.hubs.flatMap(hub => {
        if (!isRecord(hub) || typeof hub.id !== 'string') return []
        const hubId = hub.id.trim()
        if (!hubId) return []
        return [
          {
            id: hubId,
            label: typeof hub.label === 'string' ? hub.label.trim() : undefined,
            cls: asClass(hub.cls),
          },
        ]
      })
    : []
  const members =
    value.members === 'scan' || value.members === 'stream'
      ? value.members
      : typeof value.members === 'string' && value.members.trim()
        ? value.members.trim()
        : 'scan'
  return {
    id,
    title: typeof value.title === 'string' ? value.title.trim() : undefined,
    chronology,
    hubs,
    edges: typeof value.edges === 'string' ? value.edges : undefined,
    members,
  }
}
