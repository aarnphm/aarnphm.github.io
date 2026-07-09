import { isRecord, readString } from './type-guards'

const DOT_ESCAPE = '___DOT___'
export const STACKED_NOTE_METADATA_CLASSES = ['modified-time', 'published-time', 'reading-time']

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

function fragmentHasClass(fragment: string, className: string): boolean {
  return (
    fragment.includes(`"${className}`) ||
    fragment.includes(` ${className}`) ||
    fragment.includes(`${className}"`)
  )
}

export function stackedNoteMetadataHtml(items: string[]): string {
  const ordered: string[] = []
  for (const className of STACKED_NOTE_METADATA_CLASSES) {
    for (const item of items) {
      if (!fragmentHasClass(item, className) || ordered.includes(item)) continue
      ordered.push(item)
    }
  }

  if (ordered.length === 0) return ''

  return `<footer class="stacked-note-footer" aria-label="note metadata">
  <ul class="content-meta stacked-note-content-meta">
${ordered.map(item => `    ${item}`).join('\n')}
  </ul>
</footer>`
}

function tagHasClass(tag: string, className: string): boolean {
  const classMatch = tag.match(/\sclass=(["'])(.*?)\1/i)
  if (!classMatch) return false
  return classMatch[2].split(/\s+/).includes(className)
}

function findPageFooterIndex(html: string): number {
  let offset = 0
  while (offset < html.length) {
    const start = html.indexOf('<', offset)
    if (start === -1) return -1
    const end = html.indexOf('>', start + 1)
    if (end === -1) return -1
    const tag = html.slice(start, end + 1)
    if (!tag.startsWith('</') && tagHasClass(tag, 'page-footer')) return start
    offset = end + 1
  }
  return -1
}

export function withStackedNoteMetadata(content: string, metadata: string | undefined): string {
  const footer = metadata?.trim()
  if (!footer) return content

  const pageFooterIndex = findPageFooterIndex(content)
  if (pageFooterIndex === -1) return `${content}\n${footer}`

  return `${content.slice(0, pageFooterIndex)}${footer}\n${content.slice(pageFooterIndex)}`
}

export type StackedNoteState = 'pending' | 'ready' | 'protected' | 'failed'

export interface StackedNotePayload {
  slug: string
  title: string
  content: string
  metadata?: string
  state: StackedNoteState
}

export interface NoteDocument {
  slug: string
  title: string
  hash?: string
  bodyHtml: string
  metadataHtml?: string
  state: StackedNoteState
}

export interface MountedNote {
  shell: HTMLElement
  bodyHost: HTMLElement
  titleRail: HTMLElement
  mounted: boolean
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
  return window.stackedNotePayloadCache?.get(slug) ?? null
}

export function cacheStackedNotePayload(payload: StackedNotePayload): void {
  if (payload.state === 'ready') sharedStackedNotePayloadCache().set(payload.slug, payload)
}

export function readStackedNotePayload(value: unknown): StackedNotePayload | null {
  if (!isRecord(value)) return null
  const slug = readString(value, 'slug')
  const title = readString(value, 'title')
  const content = readString(value, 'content')
  const metadata = readString(value, 'metadata')
  const state = readString(value, 'state')
  if (!slug || !title || !content || !isStackedNoteState(state)) return null
  return { slug, title, content, metadata, state }
}
