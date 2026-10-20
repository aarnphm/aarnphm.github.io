import { isRecord, readString } from './type-guards'

const DOT_ESCAPE = '___DOT___'
export const NOTES_HOSTNAME = 'notes.aarnphm.xyz'

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (const byte of bytes) {
    binary += String.fromCharCode(byte)
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function base64ToBytes(hash: string): Uint8Array | null {
  const base64 = hash.replace(/-/g, '+').replace(/_/g, '/')
  const padding = (4 - (base64.length % 4)) % 4
  let binary: string
  try {
    binary = atob(`${base64}${'='.repeat(padding)}`)
  } catch {
    return null
  }
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

export function normalizeStackedNoteSlug(raw: string | null): string | null {
  if (!raw) return null
  let decoded: string
  try {
    decoded = decodeURIComponent(raw)
  } catch {
    return null
  }
  const slug = decoded.replace(/^\/+|\/+$/g, '')
  if (!slug) return 'index'
  if (slug.split('/').some(part => part === '' || part === '.' || part === '..')) return null
  if (slug.includes('?') || slug.includes('#') || slug.includes('\\')) return null
  if ([...slug].some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) return null
  return slug
}

export function hashStackedNoteSlug(slug: string): string {
  const safePath = slug.toString().replace(/\./g, DOT_ESCAPE)
  return bytesToBase64(new TextEncoder().encode(safePath))
}

export function decodeStackedNoteHash(hash: string): string | null {
  const bytes = base64ToBytes(hash)
  if (!bytes) return null

  let decoded: string
  try {
    decoded = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    return null
  }

  return normalizeStackedNoteSlug(decoded.replace(/___DOT___/g, '.'))
}

export function stackedNoteRootSlug(url: URL): string | null {
  if (url.hostname === NOTES_HOSTNAME && url.pathname === '/') return 'notes'
  const slug = normalizeStackedNoteSlug(url.pathname.replace(/\.html?$/, ''))
  if (!slug || url.hostname !== NOTES_HOSTNAME) return slug
  const decoded = decodeStackedNoteHash(slug)
  return decoded && hashStackedNoteSlug(decoded) === slug ? decoded : slug
}

export function stackedNoteSlugs(url: URL): string[] {
  const root = stackedNoteRootSlug(url)
  const slugs = [root, ...url.searchParams.getAll('stackedNotes').map(decodeStackedNoteHash)]
  return [...new Set(slugs.filter((slug): slug is string => slug !== null))]
}

export function stackedNotesUrl(base: URL, slugs: string[]): URL {
  const url = new URL(base)
  if (url.hostname === NOTES_HOSTNAME && slugs[0]) {
    url.pathname = `/${hashStackedNoteSlug(slugs[0])}`
  }
  url.searchParams.delete('stackedNotes')
  for (const slug of [...new Set(slugs)].slice(1)) {
    url.searchParams.append('stackedNotes', hashStackedNoteSlug(slug))
  }
  return url
}

export type StackedNoteState = 'pending' | 'ready' | 'protected' | 'failed'

export interface StackedNotePayload {
  slug: string
  title: string
  content: string
  state: StackedNoteState
}

export interface NoteDocument {
  slug: string
  title: string
  hash?: string
  bodyHtml: string
  state: StackedNoteState
}

export interface MountedNote {
  shell: HTMLElement
  bodyHost: HTMLElement
  titleRail: HTMLElement
  mounted: boolean
  scrollTop: number
}

export interface VirtualRange {
  first: number
  last: number
}

export interface DagNode {
  slug: string
  title: string
  document: NoteDocument
  mounted: MountedNote
  anchor?: HTMLElement | null
}

export class Dag {
  private readonly nodes = new Map<string, DagNode>()
  private readonly order: string[] = []

  addNode(node: DagNode): DagNode {
    const existing = this.nodes.get(node.slug)
    if (existing) return existing
    this.nodes.set(node.slug, node)
    this.order.push(node.slug)
    return node
  }

  getOrderedNodes(): DagNode[] {
    return this.order.flatMap(slug => {
      const node = this.nodes.get(slug)
      return node ? [node] : []
    })
  }

  truncateAfter(slug: string): void {
    const index = this.order.indexOf(slug)
    if (index === -1) return
    for (const removed of this.order.splice(index + 1)) this.nodes.delete(removed)
  }

  clear(): void {
    this.nodes.clear()
    this.order.length = 0
  }

  has(slug: string): boolean {
    return this.nodes.has(slug)
  }

  get(slug: string): DagNode | undefined {
    return this.nodes.get(slug)
  }

  getTail(): DagNode | undefined {
    const lastSlug = this.order.at(-1)
    return lastSlug ? this.nodes.get(lastSlug) : undefined
  }
}

export const MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES = 24

function isStackedNoteState(value: unknown): value is StackedNoteState {
  return value === 'pending' || value === 'ready' || value === 'protected' || value === 'failed'
}

function sharedStackedNotePayloadCache(): Map<string, StackedNotePayload> {
  window.stackedNotePayloadCache ??= new Map()
  return window.stackedNotePayloadCache
}

export function stackedNotePayloadUrl(slug: string): URL {
  const url = new URL('/api/stacked-note', window.location.toString())
  url.searchParams.set('slug', slug)
  return url
}

export function getCachedStackedNotePayload(slug: string): StackedNotePayload | null {
  const cache = window.stackedNotePayloadCache
  if (!cache) return null
  const cached = cache.get(slug)
  if (!cached) return null
  cache.delete(slug)
  cache.set(slug, cached)
  return cached
}

export function cacheStackedNotePayload(payload: StackedNotePayload): void {
  if (payload.state !== 'ready') return
  const cache = sharedStackedNotePayloadCache()
  cache.delete(payload.slug)
  while (cache.size >= MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES) {
    const oldest = cache.keys().next().value
    if (oldest === undefined) break
    cache.delete(oldest)
  }
  cache.set(payload.slug, payload)
}

export function clearStackedNotePayloadCache(): void {
  window.stackedNotePayloadCache?.clear()
}

export function readStackedNotePayload(value: unknown): StackedNotePayload | null {
  if (!isRecord(value)) return null
  const slug = readString(value, 'slug')
  const title = readString(value, 'title')
  const content = readString(value, 'content')
  const state = readString(value, 'state')
  if (!slug || !title || !content || !isStackedNoteState(state)) return null
  return { slug, title, content, state }
}
