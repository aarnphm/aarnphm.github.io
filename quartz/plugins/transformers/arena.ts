import Slugger from 'github-slugger'
import { Element, ElementContent, Root as HastRoot, RootContent } from 'hast'
import { toString } from 'hast-util-to-string'
import { createHash } from 'node:crypto'
import { QuartzTransformerPlugin } from '../../types/plugin'
import {
  ArenaExternalEmbedMode,
  defaultArenaExternalEmbedMode,
  readArenaExternalEmbedMode,
} from '../../util/arena-embed'
import { parseTwitterPostUrl } from '../../util/twitter'
import { fetchTwitterPost } from '../../util/twitter-content'
import { buildYouTubeEmbed } from '../../util/youtube'
import {
  populateArenaBlockContent,
  readArenaBlockText,
  parseLinkTitle,
  stripTrailingMarkers,
} from './arena/block-content'
import {
  extractMetadataFromList,
  normalizeChannelMetadata,
  parseCoordinateMetadata,
  extractNestedList,
  getFirstTextContent,
} from './arena/metadata'

export interface ArenaBlock {
  id: string
  entryId?: string
  content: string
  url?: string
  title?: string
  titleHtmlNode?: ElementContent
  subItems?: ArenaBlock[]
  highlighted?: boolean
  pinned?: boolean
  later?: boolean
  htmlNode?: ElementContent
  embedHtml?: string
  metadata?: Record<string, unknown>
  coordinates?: { lat: number; lon: number }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  internalTitle?: string
  tags?: string[]
  embedMode?: ArenaExternalEmbedMode
  importance?: number
}

export interface ArenaChannel {
  id: string
  name: string
  slug: string
  blocks: ArenaBlock[]
  titleHtmlNode?: ElementContent
  metadata?: Record<string, string | boolean>
  tags?: string[]
}

export interface ArenaData {
  channels: ArenaChannel[]
}

export interface ArenaBlockSearchable {
  id: string
  entryId?: string
  channelSlug: string
  channelName: string
  content: string
  title?: string
  titleHtml?: string
  blockHtml?: string
  url?: string
  highlighted: boolean
  pinned: boolean
  later: boolean
  embedHtml?: string
  metadata?: Record<string, unknown>
  coordinates?: { lat: number; lon: number }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  internalTitle?: string
  tags?: string[]
  subItems?: ArenaBlockSearchable[]
  hasModalInDom: boolean
  embedMode?: ArenaExternalEmbedMode
  importance?: number
}

export interface ArenaChannelSearchable {
  id: string
  name: string
  slug: string
  blockCount: number
}

export interface ArenaSearchIndex {
  version: string
  blocks: ArenaBlockSearchable[]
  channels: ArenaChannelSearchable[]
}

declare module 'vfile' {
  interface DataMap {
    arenaData?: ArenaData
    arenaChannel?: ArenaChannel
    arenaEntry?: ArenaBlock
  }
}

const WAYBACK_PREFIX = 'https://web.archive.org/web/*/'
const REMOVE_PAYWALL_PATTERN = /^https?:\/\/(?:www\.)?removepaywalls\.com\/(?=https?:\/\/)/i

const archiveUrl = (url: string): string => {
  const original = url.replace(REMOVE_PAYWALL_PATTERN, '')
  return /^https?:\/\/web\.archive\.org\//i.test(original)
    ? original
    : `${WAYBACK_PREFIX}${original}`
}

const elementContainsAnchor = (node: ElementContent): boolean => {
  if (node.type !== 'element') return false
  if (node.tagName === 'a') return true
  return node.children.some(child => elementContainsAnchor(child))
}

const isElement = (node: RootContent | ElementContent): node is Element => node.type === 'element'

const isH2 = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === 'h2'

const isUl = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === 'ul'

const isLi = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === 'li'

export const Arena: QuartzTransformerPlugin = () => {
  return {
    name: 'Arena',
    htmlPlugins(ctx) {
      return [
        () => {
          return (tree: HastRoot, file) => {
            if (file.data.slug !== 'are.na') return
            return (async () => {
              const channels: ArenaChannel[] = []
              const slugger = new Slugger()
              let blockCounter = 0
              const embedPromises: Promise<void>[] = []

              const parseBlock = (li: Element, depth = 0): ArenaBlock | null => {
                const { highlighted, embedDisabledFromMarker, strippedContent } =
                  readArenaBlockText(li)

                let url: string | undefined
                let titleCandidate: string | undefined

                const findFirstLink = (node: Element): Element | null => {
                  for (const child of node.children) {
                    if (isElement(child) && child.tagName === 'a') return child
                    if (isElement(child)) {
                      const found = findFirstLink(child)
                      if (found) return found
                    }
                  }
                  return null
                }

                const firstLink = findFirstLink(li)

                if (firstLink && depth === 0) {
                  const linkText = stripTrailingMarkers(toString(firstLink).trim())
                  if (linkText.length > 0) titleCandidate = linkText

                  const href = firstLink.properties?.href
                  if (typeof href === 'string' && /^https?:\/\//.test(href)) {
                    url = href
                  }
                }

                if (depth === 0 && strippedContent.toLowerCase().startsWith('http')) {
                  const parsed = parseLinkTitle(strippedContent)
                  if (parsed) {
                    url = parsed.url
                    let parsedTitle = parsed.title
                    if (parsedTitle) {
                      parsedTitle = stripTrailingMarkers(parsedTitle)
                    }
                    titleCandidate = parsedTitle ?? parsed.url ?? titleCandidate
                  }
                }

                if (depth > 0 && firstLink && !url) {
                  const href = firstLink.properties?.href
                  if (typeof href === 'string' && /^https?:\/\//.test(href)) {
                    url = href
                  }
                }

                if (!titleCandidate || titleCandidate.length === 0) {
                  titleCandidate = strippedContent.length > 0 ? strippedContent : undefined
                }
                if ((!titleCandidate || titleCandidate.length === 0) && url) {
                  titleCandidate = url
                }

                const defaultEmbedMode = defaultArenaExternalEmbedMode(url, embedDisabledFromMarker)
                const block: ArenaBlock = {
                  id: `block-${blockCounter++}`,
                  content: titleCandidate || strippedContent || url || '',
                  title: titleCandidate,
                  url: url && REMOVE_PAYWALL_PATTERN.test(url) ? archiveUrl(url) : url,
                  highlighted,
                  embedMode: defaultEmbedMode,
                }

                const nestedList = extractNestedList(li)
                if (nestedList) {
                  const meta = extractMetadataFromList(nestedList)
                  if (meta.metadata) {
                    block.metadata = meta.metadata

                    const coordValue = block.metadata?.coord
                    if (typeof coordValue === 'string' && coordValue.length > 0) {
                      const parsedCoords = parseCoordinateMetadata(coordValue)
                      if (parsedCoords) {
                        block.coordinates = parsedCoords
                        delete block.metadata?.coord
                      }
                    }

                    const pinnedValue = block.metadata?.pinned
                    if (pinnedValue !== undefined && depth === 0) {
                      const normalizedPinned =
                        typeof pinnedValue === 'string'
                          ? pinnedValue.toLowerCase().trim()
                          : pinnedValue === true
                            ? 'true'
                            : pinnedValue === false
                              ? 'false'
                              : undefined
                      if (normalizedPinned === 'true' || normalizedPinned === 'yes') {
                        block.pinned = true
                        delete block.metadata?.pinned
                      } else if (normalizedPinned === 'false' || normalizedPinned === 'no') {
                        delete block.metadata?.pinned
                      }
                    }

                    const laterValue = block.metadata?.later
                    if (laterValue !== undefined && depth === 0) {
                      const normalizedLater =
                        typeof laterValue === 'string'
                          ? laterValue.toLowerCase().trim()
                          : laterValue === true
                            ? 'true'
                            : laterValue === false
                              ? 'false'
                              : undefined
                      if (normalizedLater === 'true' || normalizedLater === 'yes') {
                        block.later = true
                        delete block.metadata?.later
                      } else if (normalizedLater === 'false' || normalizedLater === 'no') {
                        delete block.metadata?.later
                      }
                    }

                    const importanceValue = block.metadata?.importance
                    if (importanceValue !== undefined) {
                      const parsed =
                        typeof importanceValue === 'number'
                          ? importanceValue
                          : typeof importanceValue === 'string'
                            ? Number.parseFloat(importanceValue)
                            : Number.NaN
                      if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
                        block.importance = parsed
                      }
                      delete block.metadata?.importance
                    }

                    const embedValue = block.metadata?.embed
                    const embedMode = readArenaExternalEmbedMode(embedValue)
                    if (embedMode) {
                      block.embedMode = embedMode
                      delete block.metadata?.embed
                    }

                    const unlockedValue = block.metadata?.unlocked
                    if (unlockedValue !== undefined) {
                      const normalizedUnlocked =
                        typeof unlockedValue === 'string'
                          ? unlockedValue.toLowerCase().trim()
                          : unlockedValue === true
                            ? 'true'
                            : unlockedValue === false
                              ? 'false'
                              : undefined
                      if (
                        (normalizedUnlocked === 'true' || normalizedUnlocked === 'yes') &&
                        block.url
                      ) {
                        block.url = archiveUrl(block.url)
                      }
                      delete block.metadata?.unlocked
                    }

                    if (block.metadata && Object.keys(block.metadata).length === 0) {
                      delete block.metadata
                    }
                  }
                  if (meta.tags) block.tags = meta.tags

                  if (nestedList.children.length > 0 && (meta.metadata || meta.tags)) {
                    const firstItem = nestedList.children.find(isLi)
                    if (firstItem) {
                      const label = getFirstTextContent(firstItem).trim().toLowerCase()
                      if (/^\[meta\](?:\s*[:\-–—])?$/.test(label)) {
                        const metaIndex = nestedList.children.indexOf(firstItem)
                        if (metaIndex !== -1) {
                          nestedList.children.splice(metaIndex, 1)
                        }
                      }
                    }
                  }

                  const subItems: ArenaBlock[] = []
                  for (const child of nestedList.children) {
                    if (isLi(child)) {
                      const subBlock = parseBlock(child, depth + 1)
                      if (subBlock) subItems.push(subBlock)
                    }
                  }
                  if (subItems.length > 0) block.subItems = subItems
                }

                if (url && parseTwitterPostUrl(url)) {
                  embedPromises.push(
                    fetchTwitterPost(url)
                      .then(html => {
                        if (html) block.embedHtml = html
                      })
                      .catch(() => undefined),
                  )
                }

                if (url && !block.embedHtml) {
                  const youtubeEmbed = buildYouTubeEmbed(url)
                  if (youtubeEmbed) {
                    block.embedHtml = `<iframe class="arena-modal-iframe arena-modal-iframe-youtube" title="YouTube embed: ${(block.title ?? block.content ?? block.id).replace(/"/g, '&quot;')}" loading="lazy" data-block-id="${block.id}" src="${youtubeEmbed.src}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>`
                  }
                }

                populateArenaBlockContent(ctx, file, block, li, url)

                return block
              }

              const bodyChildren = tree.children.filter(
                child => child.type !== 'doctype',
              ) as RootContent[]

              for (let i = 0; i < bodyChildren.length; i++) {
                const node = bodyChildren[i]

                if (isH2(node)) {
                  let name = toString(node).trim()

                  const linkInH2 = node.children.find(ch => isElement(ch) && ch.tagName === 'a')
                  if (linkInH2 && isElement(linkInH2)) {
                    const href = linkInH2.properties?.href
                    if (typeof href === 'string' && !/^https?:\/\//i.test(href)) {
                      try {
                        const parts = decodeURI(href).split('/')
                        const base = parts.at(-1)
                        if (base && base.trim().length > 0) {
                          name = base.trim()
                        }
                      } catch {}
                    }
                  }

                  if (name.length === 0) name = `Channel ${channels.length + 1}`
                  const slug = slugger.slug(name || `channel-${channels.length + 1}`)

                  const channel: ArenaChannel = {
                    id: `channel-${channels.length}`,
                    name,
                    slug,
                    blocks: [],
                  }

                  if (elementContainsAnchor(node)) {
                    const span: Element = {
                      type: 'element',
                      tagName: 'span',
                      properties: {},
                      children: (node.children ?? []).map(child =>
                        structuredClone(child as ElementContent),
                      ),
                    }
                    channel.titleHtmlNode = span
                  }

                  channels.push(channel)
                } else if (isUl(node)) {
                  if (channels.length > 0) {
                    const currentChannel = channels[channels.length - 1]
                    const ulElement = node as Element

                    const channelMeta = extractMetadataFromList(ulElement)
                    if (channelMeta.metadata) {
                      const normalizedMeta = normalizeChannelMetadata(channelMeta.metadata)
                      if (Object.keys(normalizedMeta).length > 0) {
                        currentChannel.metadata = normalizedMeta
                      }
                    }
                    if (channelMeta.tags) {
                      currentChannel.tags = channelMeta.tags
                    }
                    if (channelMeta.metadata || channelMeta.tags) {
                      const firstItem = ulElement.children.find(isLi)
                      if (firstItem) {
                        const label = getFirstTextContent(firstItem).trim().toLowerCase()
                        if (/^\[meta\](?:\s*[:\-–—])?$/.test(label)) {
                          const metaIndex = ulElement.children.indexOf(firstItem)
                          if (metaIndex !== -1) {
                            ulElement.children.splice(metaIndex, 1)
                          }
                        }
                      }
                    }

                    for (const child of ulElement.children as ElementContent[]) {
                      if (isLi(child)) {
                        const block = parseBlock(child)
                        if (block) currentChannel.blocks.push(block)
                      }
                    }
                  }
                }
              }

              if (embedPromises.length > 0) {
                await Promise.all(embedPromises)
              }

              const routes = new Set<string>()
              for (const channel of channels) {
                const occurrences = new Map<string, number>()
                for (const block of channel.blocks) {
                  const source = block.url ?? block.internalHref ?? block.content
                  const occurrence = occurrences.get(source) ?? 0
                  occurrences.set(source, occurrence + 1)
                  const digest = createHash('sha256')
                    .update(`arena-entry-v1\0${channel.slug}\0${source}\0${occurrence}`)
                    .digest('hex')
                  block.entryId = `entry-v1-${digest}`
                  const route = `${channel.slug}/${block.entryId}`
                  if (routes.has(route)) throw new Error(`Duplicate Arena entry route: ${route}`)
                  routes.add(route)
                }
              }

              file.data.arenaData = { channels }
            })()
          }
        },
      ]
    },
  }
}
