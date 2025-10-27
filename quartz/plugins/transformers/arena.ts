import { QuartzTransformerPlugin } from "../types"
import { Root as MdastRoot, List, ListItem, Paragraph, Link } from "mdast"
import { Element, ElementContent } from "hast"
import { QuartzPluginData } from "../vfile"
import { toString } from "mdast-util-to-string"
import Slugger from "github-slugger"
import { visit, CONTINUE } from "unist-util-visit"
import { externalLinkRegex } from "./ofm"
import { createHash } from "crypto"
import { fetchTwitterEmbed, twitterUrlRegex } from "./twitter"
import {
  splitAnchor,
  transformLink,
  stripSlashes,
  resolveRelative,
  FullSlug,
} from "../../util/path"
import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { createWikilinkRegex, parseWikilink, resolveWikilinkTarget } from "../../util/wikilinks"

const datasetKey = (attr: string): string =>
  attr.replace(/-([a-z])/g, (_match, ch: string) => ch.toUpperCase())

const getDataAttr = <T = unknown>(
  props: Record<string, unknown> | undefined,
  attr: string,
): T | undefined => {
  if (!props) return undefined
  return (props[attr] ?? props[datasetKey(attr)]) as T | undefined
}

const deleteDataAttr = (props: Record<string, unknown> | undefined, attr: string): void => {
  if (!props) return
  delete props[attr]
  delete props[datasetKey(attr)]
}
import { buildYouTubeEmbed, YouTubeEmbedSpec } from "../../util/youtube"

export interface ArenaBlock {
  id: string
  content: string
  url?: string
  title?: string
  titleHtmlNode?: ElementContent
  snapshotKey?: string
  subItems?: ArenaBlock[]
  highlighted?: boolean
  htmlNode?: ElementContent
  embedHtml?: string
  metadata?: Record<string, string>
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  tags?: string[]
  embedDisabled?: boolean
}

export interface ArenaChannel {
  id: string
  name: string
  slug: string
  blocks: ArenaBlock[]
  titleHtmlNode?: ElementContent
}

export interface ArenaData {
  channels: ArenaChannel[]
}

/**
 * Serializable version of ArenaBlock for search index.
 * ElementContent fields (titleHtmlNode, htmlNode) are converted to HTML strings
 * for JSON serialization. Enables full-text search and modal reconstruction
 * without requiring build-time ElementContent access.
 */
export interface ArenaBlockSearchable {
  /** Unique block identifier matching DOM data-block-id */
  id: string

  /** Slug of the channel containing this block */
  channelSlug: string

  /** Display name of the channel for search result context */
  channelName: string

  /** Plain text content (fallback if title unavailable) */
  content: string

  /** Block title if available */
  title?: string

  /** HTML string serialized from titleHtmlNode via hast-util-to-html */
  titleHtml?: string

  /** HTML string serialized from htmlNode for modal sidebar display */
  blockHtml?: string

  /** External URL if block references one */
  url?: string

  /** Whether block should be visually highlighted */
  highlighted: boolean

  /** Prerendered embed HTML (Twitter, YouTube iframe, etc) */
  embedHtml?: string

  /** Key-value metadata pairs (accessed date, author, etc) */
  metadata?: Record<string, string>

  /** Internal note slug reference if block links internally */
  internalSlug?: string

  /** Internal note resolved href */
  internalHref?: string

  /** Internal note anchor fragment */
  internalHash?: string

  /** Associated tags for filtering and search */
  tags?: string[]

  /** Nested sub-items (notes/annotations on this block) */
  subItems?: ArenaBlockSearchable[]

  /**
   * True if this block has a prerendered modal div in the index page DOM.
   * Optimization flag: if true, client can reuse existing modal;
   * if false, client must reconstruct from JSON data.
   */
  hasModalInDom: boolean

  /** True if embed rendering is explicitly disabled */
  embedDisabled?: boolean

  /** Snapshot key for caching (hash of url + title) */
  snapshotKey?: string
}

/**
 * Channel metadata for search index.
 * Lightweight summary without full block data.
 */
export interface ArenaChannelSearchable {
  /** Channel identifier */
  id: string

  /** Channel display name */
  name: string

  /** Channel URL slug */
  slug: string

  /** Total number of blocks in this channel */
  blockCount: number
}

/**
 * Complete search index for arena content.
 * Generated at build time in arenaPage.tsx emitter.
 * Emitted as static/arena-search.json for client-side consumption.
 */
export interface ArenaSearchIndex {
  /**
   * Schema version for cache invalidation.
   * Increment when making breaking changes to the index structure.
   */
  version: string

  /**
   * All blocks from all channels in a flat array.
   * Each block includes channel context (channelSlug, channelName).
   * Enables efficient linear search and filtering.
   */
  blocks: ArenaBlockSearchable[]

  /**
   * Channel metadata for quick lookups and navigation.
   * Sorted by block count descending (same as index page).
   */
  channels: ArenaChannelSearchable[]
}

declare module "vfile" {
  interface DataMap {
    arenaData?: ArenaData
    arenaChannel?: ArenaChannel
  }
}

// Matches trailing section containing one or more markers in any order
const TRAILING_MARKERS_PATTERN = /(?:\s*\[(?:\*\*|--|—)\])+\s*$/
const HIGHLIGHT_MARKER = /\[\*\*\]/
const EMBED_DISABLED_MARKER = /\[(?:--|—)\]/

const parseLinkTitle = (text: string): { url: string; title?: string } | undefined => {
  const match = text.match(/^(https?:\/\/[^\s]+)\s*(?:(?:--|—)\s*(.+))?$/)
  if (!match) {
    return undefined
  }

  return {
    url: match[1],
    title: match[2]?.trim(),
  }
}

const stripTrailingMarkers = (value: string): string =>
  value.replace(TRAILING_MARKERS_PATTERN, "").trim()

type NodeWithData = {
  data?: {
    hProperties?: Record<string, unknown>
  }
}

const setDataAttribute = <T extends NodeWithData>(node: T, key: string, value: string): void => {
  if (!node.data) {
    node.data = {}
  }
  if (!node.data.hProperties) {
    node.data.hProperties = {}
  }
  node.data.hProperties[key] = value
}

const cloneElementContent = <T extends ElementContent>(node: T): T => {
  // structuredClone preserves prototypes and avoids JSON serialization issues
  return typeof structuredClone === "function"
    ? structuredClone(node)
    : (JSON.parse(JSON.stringify(node)) as T)
}

const elementContainsAnchor = (node: ElementContent): boolean => {
  if (node.type !== "element") {
    return false
  }

  if (node.tagName === "a") {
    return true
  }

  if (!("children" in node) || !node.children) {
    return false
  }

  return (node.children as ElementContent[]).some((child) => elementContainsAnchor(child))
}

const makeSnapshotKey = (url: string, title: string): string => {
  const hash = createHash("sha256")
  hash.update(url)
  hash.update("::")
  hash.update(title)
  return hash.digest("hex").slice(0, 32)
}

export const Arena: QuartzTransformerPlugin = () => {
  return {
    name: "Arena",
    markdownPlugins(ctx) {
      const localeConfig = ctx.cfg.configuration.locale ?? "en"
      const locale = localeConfig.split("-")[0] ?? "en"
      return [
        () => {
          return async (tree: MdastRoot, file) => {
            if (file.data.slug !== "are.na") return

            const slugger = new Slugger()
            let blockCounter = 0
            let channelCounter = 0
            let pendingChannelId: string | null = null
            const embedPromises: Promise<void>[] = []

            const tryExtractMetadata = (list: List, owner: ListItem): void => {
              if (!list.children || list.children.length === 0) return

              const firstChild = list.children[0]
              if (firstChild?.type !== "listItem") return

              const metaParagraph = firstChild.children.find(
                (child): child is Paragraph => child.type === "paragraph",
              )
              const metaHeading = metaParagraph ? toString(metaParagraph).trim().toLowerCase() : ""
              const metaMarker = metaHeading.match(/^\[meta\](?:\s*[:\-–—])?/)
              if (!metaMarker) return

              const metaList = firstChild.children.find(
                (child): child is List => child.type === "list",
              )
              // remove marker regardless so it doesn't leak downstream
              list.children.shift()
              if (!metaList || metaList.children.length === 0) return

              const metadata: Record<string, string> = {}
              const tags: string[] = []
              const seenTags = new Set<string>()

              const pushTags = (values: string[]) => {
                for (const entry of values) {
                  const trimmed = entry.trim()
                  if (trimmed.length === 0) continue
                  if (seenTags.has(trimmed)) continue
                  seenTags.add(trimmed)
                  tags.push(trimmed)
                }
              }

              const extractTagTokens = (raw: string): string[] => {
                let normalized = raw.trim()
                if (normalized.length === 0) {
                  return []
                }
                normalized = normalized.replace(/^[-•]\s*/, "")

                const first = normalized.charAt(0)
                const last = normalized.charAt(normalized.length - 1)
                if ((first === "[" && last === "]") || (first === "(" && last === ")")) {
                  normalized = normalized.slice(1, -1)
                }

                const rawTokens = normalized
                  .split(/[\n,;|]/)
                  .map((segment) => segment.trim())
                  .filter((segment) => segment.length > 0)

                const cleaned: string[] = []
                for (const token of rawTokens.length > 0 ? rawTokens : [normalized]) {
                  const stripped = token
                    .replace(/^[-•]\s*/, "")
                    .replace(/^['"]/, "")
                    .replace(/['"]$/, "")
                    .trim()
                  if (stripped.length > 0) {
                    cleaned.push(stripped)
                  }
                }

                return cleaned
              }

              for (const item of metaList.children) {
                if (item.type !== "listItem") continue
                const itemParagraph = item.children.find(
                  (child): child is Paragraph => child.type === "paragraph",
                )
                const raw = itemParagraph ? toString(itemParagraph).trim() : toString(item).trim()
                const sublist = item.children.find((child): child is List => child.type === "list")

                let keySource: string | undefined
                let value: string | undefined

                if (raw.length > 0) {
                  const delimiterIndex = raw.indexOf(":")
                  if (delimiterIndex !== -1) {
                    keySource = raw.slice(0, delimiterIndex).trim()
                    value = raw.slice(delimiterIndex + 1).trim()
                  } else if (sublist) {
                    const normalized = raw.trim().toLowerCase()
                    if (normalized === "tags" || normalized === "[tags]") {
                      keySource = "tags"
                      value = ""
                    }
                  }
                } else if (sublist) {
                  keySource = "tags"
                  value = ""
                }

                if (!keySource) continue

                const normalizedKey = keySource.toLowerCase().replace(/\s+/g, "_")

                if (normalizedKey === "tags") {
                  const candidateStrings: string[] = []
                  if (typeof value === "string" && value.length > 0) {
                    candidateStrings.push(value)
                  }
                  if (sublist) {
                    for (const child of sublist.children) {
                      if (child.type !== "listItem") continue
                      const childParagraph = child.children.find(
                        (grand): grand is Paragraph => grand.type === "paragraph",
                      )
                      const tagText = childParagraph
                        ? toString(childParagraph).trim()
                        : toString(child).trim()
                      if (tagText.length > 0) {
                        candidateStrings.push(tagText)
                      }
                    }
                  }
                  for (const candidate of candidateStrings) {
                    const tokens = extractTagTokens(candidate)
                    if (tokens.length > 0) {
                      pushTags(tokens)
                    }
                  }
                  continue
                }

                if (!value || value.length === 0) continue
                const key = normalizedKey
                if (key.length === 0) continue
                metadata[key] = value
              }

              const hasTags = tags.length > 0
              const hasMetadata = Object.keys(metadata).length > 0

              if (hasTags || hasMetadata) {
                const payload = hasTags
                  ? JSON.stringify({ ...(hasMetadata ? { metadata } : {}), tags })
                  : JSON.stringify(metadata)
                setDataAttribute(owner as NodeWithData, "data-arena-block-meta", payload)
              }
            }

            const registerBlock = (listItem: ListItem, depth = 0) => {
              const paragraph = listItem.children.find(
                (child): child is Paragraph => child.type === "paragraph",
              )

              const textContent = paragraph ? toString(paragraph).trim() : toString(listItem).trim()

              // Extract trailing markers section and check for both markers
              const trailingMatch = textContent.match(TRAILING_MARKERS_PATTERN)
              const trailingSection = trailingMatch ? trailingMatch[0] : ""

              const highlighted = HIGHLIGHT_MARKER.test(trailingSection)
              const embedDisabled = EMBED_DISABLED_MARKER.test(trailingSection)
              const strippedContent = stripTrailingMarkers(textContent)

              const firstLink = paragraph?.children.find(
                (child): child is Link => child.type === "link",
              )

              let url: string | undefined
              let titleCandidate: string | undefined

              if (firstLink && depth === 0) {
                let linkText = toString(firstLink).trim()
                linkText = stripTrailingMarkers(linkText)
                if (linkText.length > 0) {
                  titleCandidate = linkText
                }

                if (firstLink.url && /^https?:\/\//.test(firstLink.url)) {
                  url = firstLink.url
                }
              }

              if (depth === 0 && strippedContent.toLowerCase().startsWith("http")) {
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

              if (depth > 0 && firstLink && !url && /^https?:\/\//.test(firstLink.url)) {
                url = firstLink.url
              }

              if (!titleCandidate || titleCandidate.length === 0) {
                titleCandidate = strippedContent.length > 0 ? strippedContent : undefined
              }
              if ((!titleCandidate || titleCandidate.length === 0) && url) {
                titleCandidate = url
              }

              const fallbackContent = titleCandidate || strippedContent || url || ""

              const blockId = `block-${blockCounter++}`
              const snapshotKey = url ? makeSnapshotKey(url, fallbackContent) : undefined

              setDataAttribute(listItem as NodeWithData, "data-arena-block-id", blockId)
              if (paragraph) {
                setDataAttribute(paragraph as NodeWithData, "data-arena-block-paragraph", blockId)
              }

              const info: Record<string, unknown> = {
                content: fallbackContent,
                highlighted: highlighted || undefined,
                embedDisabled: embedDisabled || undefined,
                snapshotKey,
                title: titleCandidate,
                url,
              }
              setDataAttribute(
                listItem as NodeWithData,
                "data-arena-block-info",
                JSON.stringify(info),
              )

              const nestedList = listItem.children.find(
                (child): child is List => child.type === "list",
              )
              if (nestedList) {
                tryExtractMetadata(nestedList, listItem)
                for (const child of nestedList.children) {
                  if (child.type === "listItem") {
                    registerBlock(child as ListItem, depth + 1)
                  }
                }
              }

              if (url && twitterUrlRegex.test(url)) {
                embedPromises.push(
                  fetchTwitterEmbed(url, locale)
                    .then((html) => {
                      if (html) {
                        setDataAttribute(
                          listItem as NodeWithData,
                          "data-arena-block-embed-html",
                          html,
                        )
                      }
                    })
                    .catch(() => undefined),
                )
              }

              if (url) {
                const youtubeEmbed = buildYouTubeEmbed(url)
                if (youtubeEmbed) {
                  setDataAttribute(
                    listItem as NodeWithData,
                    "data-arena-block-youtube",
                    JSON.stringify(youtubeEmbed),
                  )
                }
              }
            }

            visit(tree, (node, _index, parent) => {
              if (parent !== tree) {
                return CONTINUE
              }

              if (node.type === "heading") {
                const depth = node.depth as number | undefined
                if (depth === 2) {
                  let name = toString(node).trim()
                  const linkChild = node.children?.find((ch: any) => ch.type === "link") as
                    | Link
                    | undefined
                  if (linkChild && !/^https?:\/\//i.test(linkChild.url)) {
                    try {
                      const parts = decodeURI(linkChild.url).split("/")
                      const base = parts.at(-1)
                      if (base && base.trim().length > 0) {
                        name = base.trim()
                      }
                    } catch {}
                  }
                  if (name.length === 0) name = `Channel ${channelCounter + 1}`
                  const slug = slugger.slug(name || `channel-${channelCounter + 1}`)

                  const channelId = `channel-${channelCounter++}`
                  setDataAttribute(node as NodeWithData, "data-arena-channel-id", channelId)
                  setDataAttribute(node as NodeWithData, "data-arena-channel-name", name)
                  setDataAttribute(node as NodeWithData, "data-arena-channel-slug", slug)

                  pendingChannelId = channelId
                  return CONTINUE
                }

                pendingChannelId = null
                return CONTINUE
              }

              if (node.type === "list" && pendingChannelId) {
                setDataAttribute(
                  node as NodeWithData,
                  "data-arena-channel-blocks",
                  pendingChannelId,
                )
                for (const child of node.children) {
                  if (child.type === "listItem") {
                    registerBlock(child as ListItem)
                  }
                }
                pendingChannelId = null
                return CONTINUE
              }

              return CONTINUE
            })

            if (embedPromises.length > 0) {
              await Promise.all(embedPromises)
            }
          }
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => {
          return (tree, file) => {
            if (file.data.slug !== "are.na") return

            const channels: ArenaChannel[] = []
            const channelById = new Map<string, ArenaChannel>()

            const parseJsonAttr = <T>(value: unknown): T | undefined => {
              if (typeof value !== "string" || value.length === 0) return undefined
              try {
                return JSON.parse(value) as T
              } catch {
                return undefined
              }
            }

            const applyLinkProcessing = (node?: ElementContent): ElementContent | undefined => {
              if (!node) return undefined

              const visitEl = (el: ElementContent) => {
                if (el.type !== "element") return
                const e = el as Element

                const processTextNode = (value: string): ElementContent[] => {
                  const results: ElementContent[] = []
                  const regex = createWikilinkRegex()
                  let lastIndex = 0
                  let match: RegExpExecArray | null

                  while ((match = regex.exec(value)) !== null) {
                    const start = match.index
                    if (start > lastIndex) {
                      results.push({ type: "text", value: value.slice(lastIndex, start) })
                    }

                    const parsed = parseWikilink(match[0])
                    const resolved =
                      parsed && file.data.slug
                        ? resolveWikilinkTarget(parsed, file.data.slug as FullSlug)
                        : null

                    if (parsed && resolved) {
                      const hrefBase = resolveRelative(file.data.slug!, resolved.slug)
                      const href = parsed.anchor ? `${hrefBase}${parsed.anchor}` : hrefBase
                      results.push({
                        type: "element",
                        tagName: "a",
                        properties: {
                          href,
                          className: ["internal"],
                          "data-slug": resolved.slug,
                          "data-no-popover": true,
                        },
                        children: [
                          {
                            type: "text",
                            value: parsed.alias ?? parsed.target ?? match[0],
                          },
                        ],
                      })
                    } else {
                      results.push({
                        type: "text",
                        value: parsed?.alias ?? parsed?.target ?? match[0],
                      })
                    }

                    lastIndex = regex.lastIndex
                  }

                  if (lastIndex < value.length) {
                    results.push({ type: "text", value: value.slice(lastIndex) })
                  }

                  return results
                }

                if (e.tagName === "a" && typeof e.properties?.href === "string") {
                  let dest = e.properties.href as string
                  const classes = Array.isArray(e.properties.className)
                    ? (e.properties.className as string[])
                    : typeof e.properties.className === "string"
                      ? [e.properties.className]
                      : []

                  const isExternal = externalLinkRegex.test(dest)

                  if (!isExternal && !dest.startsWith("#")) {
                    dest = transformLink(file.data.slug!, dest, {
                      strategy: "absolute",
                      allSlugs: ctx.allSlugs,
                    })
                    const url = new URL(
                      dest,
                      "https://base.com/" + stripSlashes(file.data.slug!, true),
                    )
                    let canonical = url.pathname
                    let [destCanonical] = splitAnchor(canonical)
                    if (destCanonical.endsWith("/")) destCanonical += "index"
                    const full = decodeURIComponent(stripSlashes(destCanonical, true))
                    const canonicalHref = `${url.pathname}${url.search}${url.hash}` || "/"
                    e.properties.href = canonicalHref
                    e.properties["data-slug"] = full

                    if (!classes.includes("internal")) classes.push("internal")
                  } else if (isExternal) {
                    if (!classes.includes("external")) classes.push("external")
                  }

                  if (classes.length > 0) {
                    e.properties.className = classes
                  }
                }

                if (el.children) {
                  const newChildren: ElementContent[] = []
                  for (const child of el.children as ElementContent[]) {
                    if (child.type === "text") {
                      newChildren.push(...processTextNode(child.value))
                    } else {
                      visitEl(child)
                      newChildren.push(child)
                    }
                  }
                  el.children = newChildren as ElementContent[]
                }

                // Also transform text nodes stored directly on this element if it's a text node
                // @ts-ignore
                if (el.type === "text") {
                  return
                }
              }

              const cloned = cloneElementContent(node)
              visitEl(cloned)
              return cloned
            }

            type InternalLinkInfo = {
              slug: string
              href: string
              hash?: string
            }

            const classListFromProps = (props?: Element["properties"]): string[] => {
              if (!props) return []
              const classProp = props.className
              if (Array.isArray(classProp)) {
                return classProp.map((cls) => cls.toString())
              }
              if (typeof classProp === "string") {
                return classProp.split(/\s+/).filter((cls) => cls.length > 0)
              }
              return []
            }

            const findInternalLink = (node?: ElementContent): InternalLinkInfo | undefined => {
              if (!node) return undefined

              if (node.type === "element") {
                const elementNode = node as Element
                const classList = classListFromProps(elementNode.properties)

                if (elementNode.tagName === "a" && classList.includes("internal")) {
                  const slug = elementNode.properties?.["data-slug"]
                  if (typeof slug === "string" && slug.length > 0) {
                    const rawHref = elementNode.properties?.href
                    const hrefString = typeof rawHref === "string" ? rawHref : undefined
                    const [, anchor] = hrefString ? splitAnchor(hrefString) : ["", ""]
                    const canonicalSlug = stripSlashes(slug, true)
                    const canonicalHref = `/${canonicalSlug}${anchor ?? ""}`
                    return {
                      slug,
                      href: canonicalHref,
                      hash: anchor && anchor.length > 0 ? anchor : undefined,
                    }
                  }
                }

                if (elementNode.children) {
                  for (const child of elementNode.children as ElementContent[]) {
                    const found = findInternalLink(child)
                    if (found) {
                      return found
                    }
                  }
                }
              }

              if ("children" in node && node.children) {
                for (const child of node.children as ElementContent[]) {
                  const found = findInternalLink(child)
                  if (found) {
                    return found
                  }
                }
              }

              return undefined
            }

            const aggregateListItem = (li: Element): ElementContent => {
              const collected: ElementContent[] = []
              for (const child of (li.children ?? []) as ElementContent[]) {
                const cloned = cloneElementContent(child)
                if (cloned.type === "element" && cloned.tagName === "ul") continue
                if (cloned.type === "element" && cloned.properties) {
                  deleteDataAttr(cloned.properties, "data-arena-block-paragraph")
                  deleteDataAttr(cloned.properties, "data-arena-block-id")
                }
                collected.push(cloned)
              }
              const wrapped: ElementContent = {
                type: "element",
                tagName: "div",
                properties: {},
                children: collected,
              }
              return applyLinkProcessing(wrapped) ?? wrapped
            }

            const buildBlocksFromList = (list: Element): ArenaBlock[] => {
              const results: ArenaBlock[] = []
              for (const child of list.children ?? []) {
                if (typeof child !== "object" || child === null) continue
                if ((child as Element).type !== "element") continue
                const el = child as Element
                if (el.tagName !== "li" || !el.properties) continue
                const blockId = getDataAttr<string>(el.properties, "data-arena-block-id")
                if (typeof blockId !== "string") continue

                const info =
                  parseJsonAttr<{
                    content?: string
                    highlighted?: boolean
                    embedDisabled?: boolean
                    snapshotKey?: string
                    title?: string
                    url?: string
                  }>(getDataAttr(el.properties, "data-arena-block-info")) ?? {}

                const block: ArenaBlock = {
                  id: blockId,
                  content: info.content ?? "",
                  url: typeof info.url === "string" ? info.url : undefined,
                  title: typeof info.title === "string" ? info.title : undefined,
                  snapshotKey: typeof info.snapshotKey === "string" ? info.snapshotKey : undefined,
                  highlighted: Boolean(info.highlighted),
                  embedDisabled: Boolean(info.embedDisabled),
                }

                const metaPayload = parseJsonAttr<unknown>(
                  getDataAttr(el.properties, "data-arena-block-meta"),
                )
                if (metaPayload && typeof metaPayload === "object" && !Array.isArray(metaPayload)) {
                  const metaObject = metaPayload as Record<string, unknown>
                  const hasMetadataProp = Object.prototype.hasOwnProperty.call(
                    metaObject,
                    "metadata",
                  )
                  const rawTags = metaObject.tags
                  const tagsArray = Array.isArray(rawTags)

                  if (hasMetadataProp) {
                    const rawMetadata = metaObject.metadata
                    if (
                      rawMetadata &&
                      typeof rawMetadata === "object" &&
                      !Array.isArray(rawMetadata)
                    ) {
                      const entries: [string, string][] = []
                      for (const [key, value] of Object.entries(rawMetadata)) {
                        if (typeof value === "string" && value.length > 0) {
                          entries.push([key, value])
                        }
                      }
                      if (entries.length > 0) {
                        block.metadata = Object.fromEntries(entries)
                      }
                    }
                  }

                  if (!hasMetadataProp && !tagsArray) {
                    const entries: [string, string][] = []
                    for (const [key, value] of Object.entries(metaObject)) {
                      if (typeof value === "string" && value.length > 0) {
                        entries.push([key, value])
                      }
                    }
                    if (entries.length > 0) {
                      block.metadata = Object.fromEntries(entries)
                    }
                  }

                  if (tagsArray) {
                    const deduped: string[] = []
                    const seen = new Set<string>()
                    for (const tag of rawTags as unknown[]) {
                      if (typeof tag !== "string") continue
                      const trimmed = tag.trim()
                      if (trimmed.length === 0 || seen.has(trimmed)) continue
                      seen.add(trimmed)
                      deduped.push(trimmed)
                    }
                    if (deduped.length > 0) {
                      block.tags = deduped
                    }
                  }
                }

                const embedHtml = getDataAttr<string>(el.properties, "data-arena-block-embed-html")
                if (typeof embedHtml === "string" && embedHtml.length > 0) {
                  block.embedHtml = embedHtml
                }

                const youtubeSpec = parseJsonAttr<YouTubeEmbedSpec>(
                  getDataAttr(el.properties, "data-arena-block-youtube"),
                )
                if (!block.embedHtml && youtubeSpec?.src) {
                  const frameTitle = block.title ?? block.content ?? block.id
                  block.embedHtml = `<iframe class="arena-modal-iframe arena-modal-iframe-youtube" title="YouTube embed: ${frameTitle.replace(/"/g, "&quot;")}" loading="lazy" data-block-id="${block.id}" src="${youtubeSpec.src}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>`
                }

                const paragraphId = getDataAttr<string>(el.properties, "data-arena-block-paragraph")
                let paragraphNode: Element | undefined
                if (typeof paragraphId === "string") {
                  const match = el.children?.find((childNode) => {
                    if (typeof childNode !== "object" || childNode === null) return false
                    if ((childNode as Element).type !== "element") return false
                    const elementChild = childNode as Element
                    return (
                      getDataAttr<string>(elementChild.properties, "data-arena-block-paragraph") ===
                      paragraphId
                    )
                  }) as Element | undefined
                  if (match) {
                    paragraphNode = match
                    if (match.properties) {
                      deleteDataAttr(match.properties, "data-arena-block-paragraph")
                    }
                  }
                }

                if (paragraphNode) {
                  const paragraphClone = cloneElementContent(paragraphNode)
                  if (paragraphClone.type === "element") {
                    if (paragraphClone.tagName === "p") {
                      paragraphClone.tagName = "span"
                    }
                    if (paragraphClone.properties) {
                      deleteDataAttr(paragraphClone.properties, "data-arena-block-id")
                    }
                  }
                  block.titleHtmlNode =
                    applyLinkProcessing(paragraphClone as ElementContent) ??
                    (paragraphClone as ElementContent)
                }

                block.htmlNode = aggregateListItem(el)

                const internalLinkInfo =
                  findInternalLink(block.titleHtmlNode) ?? findInternalLink(block.htmlNode)
                if (internalLinkInfo) {
                  block.internalSlug = internalLinkInfo.slug
                  block.internalHref = internalLinkInfo.href
                  block.internalHash = internalLinkInfo.hash
                }

                const sublist = el.children?.find((childNode) => {
                  if (typeof childNode !== "object" || childNode === null) return false
                  if ((childNode as Element).type !== "element") return false
                  return (childNode as Element).tagName === "ul"
                }) as Element | undefined

                if (sublist) {
                  const subBlocks = buildBlocksFromList(sublist)
                  if (subBlocks.length > 0) {
                    block.subItems = subBlocks
                  }
                }

                deleteDataAttr(el.properties, "data-arena-block-id")
                deleteDataAttr(el.properties, "data-arena-block-info")
                deleteDataAttr(el.properties, "data-arena-block-meta")
                deleteDataAttr(el.properties, "data-arena-block-embed-html")
                deleteDataAttr(el.properties, "data-arena-block-youtube")

                results.push(block)
              }
              return results
            }

            visit(tree, "element", (node: Element) => {
              if (!node.properties) return

              const channelId = getDataAttr<string>(node.properties, "data-arena-channel-id")
              if (typeof channelId === "string") {
                const name = getDataAttr<string>(node.properties, "data-arena-channel-name")
                const slug = getDataAttr<string>(node.properties, "data-arena-channel-slug")
                if (!channelById.has(channelId)) {
                  const channel: ArenaChannel = {
                    id: channelId,
                    name: typeof name === "string" ? name : channelId,
                    slug: typeof slug === "string" ? slug : channelId,
                    blocks: [],
                  }

                  const headingClone = cloneElementContent(node)
                  if (headingClone.type === "element" && elementContainsAnchor(headingClone)) {
                    const span: Element = {
                      type: "element",
                      tagName: "span",
                      properties: {},
                      children: (headingClone.children ?? []).map((child) =>
                        cloneElementContent(child as ElementContent),
                      ),
                    }
                    channel.titleHtmlNode = applyLinkProcessing(span) ?? span
                  }

                  channels.push(channel)
                  channelById.set(channelId, channel)
                }

                deleteDataAttr(node.properties, "data-arena-channel-id")
                deleteDataAttr(node.properties, "data-arena-channel-name")
                deleteDataAttr(node.properties, "data-arena-channel-slug")
              }

              const blocksChannelId = getDataAttr<string>(
                node.properties,
                "data-arena-channel-blocks",
              )
              if (typeof blocksChannelId === "string") {
                const channel = channelById.get(blocksChannelId)
                if (channel) {
                  const blocks = buildBlocksFromList(node)
                  channel.blocks = blocks
                }
                deleteDataAttr(node.properties, "data-arena-channel-blocks")
              }
            })

            file.data.arenaData = { channels }

            if (ctx.argv.watch && ctx.argv.force) {
              try {
                const debugPath = join(process.cwd(), "content", "arena_hast.json")
                writeFileSync(debugPath, JSON.stringify(tree, null, 2))
              } catch {
                // ignore write failures during debugging
              }
            }
          }
        },
      ]
    },
  }
}
