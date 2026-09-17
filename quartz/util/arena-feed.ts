import type { ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { toString } from 'hast-util-to-string'
import type { ArenaBlock, ArenaChannel } from '../plugins/transformers/arena'
import { arenaArxivPdfUrl, isArenaPdfUrl } from './arena-embed'
import { hostnameMatches } from './url'
import { buildYouTubeEmbed } from './youtube'

export interface ArenaFeedOccurrence {
  channelSlug: string
  channelName: string
  blockId: string
  parentBlockId: string | null
  notesHtml: string | null
}

export interface CuriusFeedOccurrence {
  userId: number
  linkId: number
}

export interface ArenaFeedEntry {
  articleId: string
  sourceUrl: string
  title: string
  kind: 'html' | 'pdf' | 'video' | 'internal'
  later: boolean
  tags: string[]
  savedAt: string | null
  occurrences: ArenaFeedOccurrence[]
  curius?: CuriusFeedOccurrence[]
}

export interface ArenaFeedManifest {
  schemaVersion: 1
  revision: string
  entries: ArenaFeedEntry[]
}

const ARTICLE_ID_PATTERN = /^article-v1-[0-9a-f]{64}$/
const REVISION_PATTERN = /^feed-v1-[0-9a-f]{64}$/
const TRACKING_PARAMETERS = new Set([
  'curius',
  'fbclid',
  'gclid',
  'dclid',
  'msclkid',
  'mc_cid',
  'mc_eid',
  'igshid',
  'mkt_tok',
  '_hsenc',
  '_hsmi',
])

function isTrackingParameter(rawParameter: string): boolean {
  const name = new URLSearchParams(rawParameter).keys().next().value?.toLowerCase()
  return name !== undefined && (TRACKING_PARAMETERS.has(name) || /^utm(?:$|[_-])/.test(name))
}

export function normalizeArenaFeedUrl(rawUrl: string, baseUrl?: string): string | null {
  let url: URL
  try {
    url = new URL(rawUrl, baseUrl)
  } catch {
    return null
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null

  // Keep authored query order and encoding, including duplicate content parameters.
  const parameters = url.search.slice(1).split('&')
  const remaining = parameters.filter(parameter => !isTrackingParameter(parameter))
  if (remaining.length !== parameters.length) url.search = remaining.join('&')
  return url.href
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

export function assertArenaFeedRouteAvailable(channels: readonly ArenaChannel[]): void {
  const reserved = channels.find(
    channel => channel.slug === 'feed' || channel.slug.startsWith('feed/'),
  )
  if (reserved) {
    throw new Error(`Arena channel "${reserved.name}" uses reserved reader route /arena/feed`)
  }
}

function firstContentNode(node: ElementContent): ElementContent | undefined {
  if (node.type === 'text') return node.value.trim() ? node : undefined
  if (node.type !== 'element') return undefined
  if (node.tagName === 'ul' || node.tagName === 'ol') return undefined
  if (node.tagName === 'a') return node
  for (const child of node.children) {
    const first = firstContentNode(child)
    if (first) return first
  }
  return undefined
}

function savedBlockUrl(block: ArenaBlock, baseUrl: string): string | null {
  let rawUrl =
    block.url ??
    block.internalHref ??
    (block.internalSlug ? `/${block.internalSlug}${block.internalHash ?? ''}` : undefined)
  if (!rawUrl) return null

  if (block.htmlNode) {
    const first = firstContentNode(block.htmlNode)
    if (!first) return null
    // The existing parser also sets block.url for incidental links inside prose.
    if (first.type === 'text' && !/^https?:\/\/\S+/.test(first.value.trim())) return null
    if (first.type === 'element' && first.tagName !== 'a') return null
    if (first.type === 'element' && typeof first.properties.href === 'string')
      rawUrl = first.properties.href
  }

  return normalizeArenaFeedUrl(rawUrl, baseUrl)
}

function explicitLater(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value
  if (typeof value !== 'string') return undefined
  const normalized = value.trim().toLowerCase()
  if (normalized === 'true' || normalized === 'yes') return true
  if (normalized === 'false' || normalized === 'no') return false
  return undefined
}

export function arenaFeedSavedDate(value: unknown): string | null {
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value.toISOString()
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  const iso = /^(\d{4})[-/](\d{2})[-/](\d{2})$/.exec(trimmed)
  const us = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(trimmed)
  if (iso || us) {
    const year = iso?.[1] ?? us?.[3]
    const month = iso?.[2] ?? us?.[1]
    const day = iso?.[3] ?? us?.[2]
    if (!year || !month || !day) return null
    const date = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
    const parsed = new Date(`${date}T00:00:00.000Z`)
    return Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date
      ? null
      : date
  }
  if (!/^\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})$/.test(trimmed)) return null
  const parsed = new Date(trimmed)
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString()
}

function noteNode(block: ArenaBlock, baseUrl: string): ElementContent {
  const body: ElementContent = block.htmlNode ?? { type: 'text', value: block.content }
  const children = noteChildren(block, baseUrl)
  return {
    type: 'element',
    tagName: 'li',
    properties: {},
    children:
      children.length > 0
        ? [body, { type: 'element', tagName: 'ul', properties: {}, children }]
        : [body],
  }
}

function noteChildren(block: ArenaBlock, baseUrl: string): ElementContent[] {
  return (block.subItems ?? [])
    .filter(child => !savedBlockUrl(child, baseUrl))
    .map(child => noteNode(child, baseUrl))
}

function savedNotes(block: ArenaBlock, baseUrl: string): string | null {
  const children = noteChildren(block, baseUrl)
  return children.length > 0
    ? toHtml({ type: 'element', tagName: 'ul', properties: {}, children })
    : null
}

export function arenaFeedContentKind(sourceUrl: string, baseUrl: string): ArenaFeedEntry['kind'] {
  const url = new URL(sourceUrl)
  if (url.origin === new URL(baseUrl).origin) return 'internal'
  if (isArenaPdfUrl(sourceUrl) || arenaArxivPdfUrl(sourceUrl)) return 'pdf'
  if (
    buildYouTubeEmbed(sourceUrl) ||
    (hostnameMatches(url, 'vimeo.com') && /^\/(?:video\/)?\d+/.test(url.pathname))
  )
    return 'video'
  return 'html'
}

export async function arenaFeedArticleId(sourceUrl: string): Promise<string> {
  return `article-v1-${await sha256(`arena-article-v1\0${sourceUrl}`)}`
}

export async function finalizeArenaFeedManifest(
  entries: readonly ArenaFeedEntry[],
): Promise<ArenaFeedManifest> {
  const sorted = [...entries].sort((a, b) => compareText(a.articleId, b.articleId))
  const revision = `feed-v1-${await sha256(JSON.stringify(sorted))}`
  return { schemaVersion: 1, revision, entries: sorted }
}

export function arenaFeedSourceNames(entry: ArenaFeedEntry): string[] {
  return [
    ...new Set([
      ...entry.occurrences.map(occurrence => occurrence.channelName),
      ...(entry.curius?.length ? ['curius'] : []),
    ]),
  ]
}

export function isArenaReadingEntry(entry: ArenaFeedEntry): boolean {
  if (
    entry.kind === 'video' ||
    entry.occurrences.some(occurrence => occurrence.channelSlug === 'video')
  )
    return false
  const url = new URL(entry.sourceUrl)
  if (/\/watch\/?$/.test(url.pathname) && /^[\w-]{11}$/.test(url.searchParams.get('v') ?? ''))
    return false
  return (
    !['youtube.com', 'youtube-nocookie.com', 'youtu.be', 'vimeo.com'].some(host =>
      hostnameMatches(url, host),
    ) && !/\.(?:mp4|m4v|mov|webm|ogv|avi|mkv|m3u8|mpd)$/i.test(url.pathname)
  )
}

export async function buildArenaFeedManifest(
  channels: readonly ArenaChannel[],
  baseUrl: string,
): Promise<ArenaFeedManifest> {
  assertArenaFeedRouteAvailable(channels)
  const base = new URL(/^https?:\/\//i.test(baseUrl) ? baseUrl : `https://${baseUrl}`)
  base.pathname = '/'
  base.search = ''
  base.hash = ''
  const entriesByUrl = new Map<string, Omit<ArenaFeedEntry, 'articleId'>>()

  const visitBlock = (
    block: ArenaBlock,
    channel: ArenaChannel,
    parent: ArenaBlock | null,
    inheritedLater: boolean,
  ) => {
    const later = block.later ?? explicitLater(block.metadata?.later) ?? inheritedLater
    const sourceUrl = savedBlockUrl(block, base.href)
    if (sourceUrl) {
      const occurrence: ArenaFeedOccurrence = {
        channelSlug: channel.slug,
        channelName: channel.name,
        blockId: block.id,
        parentBlockId: parent?.id ?? null,
        notesHtml: savedNotes(block, base.href),
      }
      const date = arenaFeedSavedDate(block.metadata?.date)
      const title =
        (block.titleHtmlNode ? toString(block.titleHtmlNode).trim() : '') ||
        block.title?.trim() ||
        block.internalTitle?.trim() ||
        block.content.trim() ||
        sourceUrl
      const tags = [...new Set([...(channel.tags ?? []), ...(block.tags ?? [])])].sort(compareText)
      const existing = entriesByUrl.get(sourceUrl)
      if (existing) {
        existing.later ||= later
        existing.tags = [...new Set([...existing.tags, ...tags])].sort(compareText)
        existing.occurrences.push(occurrence)
        if (existing.title === sourceUrl && title !== sourceUrl) existing.title = title
        if (date && (!existing.savedAt || date < existing.savedAt)) existing.savedAt = date
      } else {
        entriesByUrl.set(sourceUrl, {
          sourceUrl,
          title,
          kind: arenaFeedContentKind(sourceUrl, base.href),
          later,
          tags,
          savedAt: date,
          occurrences: [occurrence],
        })
      }
    }
    for (const child of block.subItems ?? []) visitBlock(child, channel, block, later)
  }

  for (const channel of channels) {
    for (const block of channel.blocks) visitBlock(block, channel, null, false)
  }

  const entries = await Promise.all(
    [...entriesByUrl.values()].map(async entry => ({
      articleId: await arenaFeedArticleId(entry.sourceUrl),
      ...entry,
    })),
  )
  return finalizeArenaFeedManifest(entries)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function isOccurrence(value: unknown): value is ArenaFeedOccurrence {
  return (
    isRecord(value) &&
    typeof value.channelSlug === 'string' &&
    value.channelSlug.length > 0 &&
    typeof value.channelName === 'string' &&
    value.channelName.length > 0 &&
    typeof value.blockId === 'string' &&
    value.blockId.length > 0 &&
    isNullableString(value.parentBlockId) &&
    isNullableString(value.notesHtml)
  )
}

export function isCuriusFeedOccurrence(value: unknown): value is CuriusFeedOccurrence {
  return (
    isRecord(value) &&
    typeof value.userId === 'number' &&
    Number.isSafeInteger(value.userId) &&
    value.userId > 0 &&
    typeof value.linkId === 'number' &&
    Number.isSafeInteger(value.linkId) &&
    value.linkId > 0
  )
}

function isEntry(value: unknown): value is ArenaFeedEntry {
  return (
    isRecord(value) &&
    typeof value.articleId === 'string' &&
    ARTICLE_ID_PATTERN.test(value.articleId) &&
    typeof value.sourceUrl === 'string' &&
    normalizeArenaFeedUrl(value.sourceUrl) === value.sourceUrl &&
    typeof value.title === 'string' &&
    value.title.length > 0 &&
    (value.kind === 'html' ||
      value.kind === 'pdf' ||
      value.kind === 'video' ||
      value.kind === 'internal') &&
    typeof value.later === 'boolean' &&
    Array.isArray(value.tags) &&
    value.tags.every(tag => typeof tag === 'string') &&
    isNullableString(value.savedAt) &&
    (value.savedAt === null || arenaFeedSavedDate(value.savedAt) !== null) &&
    Array.isArray(value.occurrences) &&
    value.occurrences.every(isOccurrence) &&
    (value.curius === undefined ||
      (Array.isArray(value.curius) && value.curius.every(isCuriusFeedOccurrence))) &&
    (value.occurrences.length > 0 || (Array.isArray(value.curius) && value.curius.length > 0))
  )
}

export function parseArenaFeedManifest(value: unknown): ArenaFeedManifest | null {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    typeof value.revision !== 'string' ||
    !REVISION_PATTERN.test(value.revision) ||
    !Array.isArray(value.entries) ||
    !value.entries.every(isEntry)
  )
    return null
  const identities = new Set(value.entries.map(entry => entry.articleId))
  const urls = new Set(value.entries.map(entry => entry.sourceUrl))
  if (identities.size !== value.entries.length || urls.size !== value.entries.length) return null
  return { schemaVersion: 1, revision: value.revision, entries: value.entries }
}

function shuffleRank(seed: string, articleId: string): number {
  let hash = 2166136261
  for (const character of `${seed}\0${articleId}`) {
    hash = Math.imul(hash ^ character.charCodeAt(0), 16777619)
  }
  return hash >>> 0
}

export function orderArenaFeedEntries(
  entries: readonly ArenaFeedEntry[],
  seed: string,
): ArenaFeedEntry[] {
  return entries
    .map(entry => ({ entry, rank: shuffleRank(seed, entry.articleId) }))
    .sort(
      (a, b) =>
        Number(b.entry.later) - Number(a.entry.later) ||
        a.rank - b.rank ||
        compareText(a.entry.articleId, b.entry.articleId),
    )
    .map(({ entry }) => entry)
}
