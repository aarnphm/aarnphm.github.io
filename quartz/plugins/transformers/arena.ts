import { QuartzTransformerPlugin } from "../types"
import { Element, ElementContent, Root as HastRoot, RootContent } from "hast"
import { toString } from "hast-util-to-string"
import Slugger from "github-slugger"
import { externalLinkRegex } from "./ofm"
import { fetchTwitterEmbed, twitterUrlRegex } from "./twitter"
import { splitAnchor, transformLink, stripSlashes, FullSlug } from "../../util/path"
import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { createWikilinkRegex, parseWikilink, resolveWikilinkTarget } from "../../util/wikilinks"
import { buildYouTubeEmbed } from "../../util/youtube"

export interface ArenaBlock {
  id: string
  content: string
  url?: string
  title?: string
  titleHtmlNode?: ElementContent
  subItems?: ArenaBlock[]
  highlighted?: boolean
  htmlNode?: ElementContent
  embedHtml?: string
  metadata?: Record<string, string>
  coordinates?: {
    lon: number
    lat: number
  }
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
  metadata?: Record<string, string>
  tags?: string[]
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

  /** Geographic coordinates for map rendering */
  coordinates?: {
    lon: number
    lat: number
  }

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

// Get text content from li excluding nested ul elements
const getTextContentExcludingNestedUl = (li: Element): string => {
  let text = ""
  for (const child of li.children as ElementContent[]) {
    // skip nested ul elements
    if (isElement(child) && child.tagName === "ul") continue
    text += toString(child)
  }
  return text.trim()
}

const cloneElementContent = <T extends ElementContent>(node: T): T => {
  // structuredClone preserves prototypes and avoids JSON serialization issues
  return typeof structuredClone === "function"
    ? structuredClone(node)
    : (JSON.parse(JSON.stringify(node)) as T)
}

const COORDINATE_NUMBER_PATTERN = /-?\d+(?:\.\d+)?/g

const parseCoordinateMetadata = (value: string): { lon: number; lat: number } | null => {
  if (!value) return null

  const matches = value.match(COORDINATE_NUMBER_PATTERN)
  if (!matches || matches.length < 2) {
    return null
  }

  const lon = Number.parseFloat(matches[0])
  const lat = Number.parseFloat(matches[1])

  if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
    return null
  }

  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    return null
  }

  return { lon, lat }
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

const isElement = (node: RootContent | ElementContent): node is Element => node.type === "element"

const isH2 = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "h2"

const isUl = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "ul"

const isLi = (node: RootContent | ElementContent): node is Element =>
  isElement(node) && node.tagName === "li"

const getFirstTextContent = (node: Element): string => {
  for (const child of node.children) {
    if (child.type === "text") {
      return child.value
    }
    if (isElement(child)) {
      const text = getFirstTextContent(child)
      if (text) return text
    }
  }
  return ""
}

const extractNestedList = (li: Element): Element | null => {
  for (const child of li.children) {
    if (isUl(child)) return child
  }
  return null
}

export const Arena: QuartzTransformerPlugin = () => {
  return {
    name: "Arena",
    htmlPlugins(ctx) {
      const localeConfig = ctx.cfg.configuration.locale ?? "en"
      const locale = localeConfig.split("-")[0] ?? "en"

      return [
        () => {
          return async (tree: HastRoot, file) => {
            if (file.data.slug !== "are.na") return

            const channels: ArenaChannel[] = []
            const slugger = new Slugger()
            let blockCounter = 0
            const embedPromises: Promise<void>[] = []

            const extractMetadataFromList = (
              list: Element,
            ): {
              metadata?: Record<string, string>
              tags?: string[]
            } => {
              if (list.children.length === 0) return {}

              const firstItem = list.children.find(isLi)
              if (!firstItem) return {}

              const label = getFirstTextContent(firstItem).trim().toLowerCase()
              const metaMatch = label.match(/^\[meta\](?:\s*[:\-–—])?$/)
              if (!metaMatch) return {}

              const metaList = extractNestedList(firstItem)
              if (!metaList || metaList.children.length === 0) return {}

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
                if (normalized.length === 0) return []
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
                  if (stripped.length > 0) cleaned.push(stripped)
                }

                return cleaned
              }

              for (const item of metaList.children) {
                if (!isLi(item)) continue
                const raw = getFirstTextContent(item).trim()
                const sublist = extractNestedList(item)

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
                      if (!isLi(child)) continue
                      const tagText = getFirstTextContent(child).trim()
                      if (tagText.length > 0) candidateStrings.push(tagText)
                    }
                  }
                  for (const candidate of candidateStrings) {
                    const tokens = extractTagTokens(candidate)
                    if (tokens.length > 0) pushTags(tokens)
                  }
                  continue
                }

                if (!value || value.length === 0) continue
                const key = normalizedKey
                if (key.length === 0) continue
                metadata[key] = value
              }

              const result: { metadata?: Record<string, string>; tags?: string[] } = {}
              if (Object.keys(metadata).length > 0) result.metadata = metadata
              if (tags.length > 0) result.tags = tags
              return result
            }

            // parse block from li element
            const parseBlock = (li: Element, depth = 0): ArenaBlock | null => {
              // get text excluding nested ul to detect markers correctly
              const textContent = getTextContentExcludingNestedUl(li)

              // extract trailing markers
              const trailingMatch = textContent.match(TRAILING_MARKERS_PATTERN)
              const trailingSection = trailingMatch ? trailingMatch[0] : ""
              const highlighted = HIGHLIGHT_MARKER.test(trailingSection)
              const embedDisabled = EMBED_DISABLED_MARKER.test(trailingSection)
              const strippedContent = stripTrailingMarkers(textContent)

              // find first <a> element
              let url: string | undefined
              let titleCandidate: string | undefined

              const findFirstLink = (node: Element): Element | null => {
                for (const child of node.children) {
                  if (isElement(child) && child.tagName === "a") return child
                  if (isElement(child)) {
                    const found = findFirstLink(child)
                    if (found) return found
                  }
                }
                return null
              }

              const firstLink = findFirstLink(li)

              if (firstLink && depth === 0) {
                const linkText = toString(firstLink).trim()
                const strippedLinkText = stripTrailingMarkers(linkText)
                if (strippedLinkText.length > 0) {
                  titleCandidate = strippedLinkText
                }

                const href = firstLink.properties?.href
                if (typeof href === "string" && /^https?:\/\//.test(href)) {
                  url = href
                }
              }

              // check for plain URL with optional title
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

              // nested blocks can have links too
              if (depth > 0 && firstLink && !url) {
                const href = firstLink.properties?.href
                if (typeof href === "string" && /^https?:\/\//.test(href)) {
                  url = href
                }
              }

              if (!titleCandidate || titleCandidate.length === 0) {
                titleCandidate = strippedContent.length > 0 ? strippedContent : undefined
              }
              if ((!titleCandidate || titleCandidate.length === 0) && url) {
                titleCandidate = url
              }

              const fallbackContent = titleCandidate || strippedContent || url || ""
              const blockId = `block-${blockCounter++}`

              const block: ArenaBlock = {
                id: blockId,
                content: fallbackContent,
                title: titleCandidate,
                url,
                highlighted,
                embedDisabled,
              }

              // handle nested list
              const nestedList = extractNestedList(li)
              if (nestedList) {
                // check for metadata first
                const meta = extractMetadataFromList(nestedList)
                if (meta.metadata) {
                  block.metadata = meta.metadata

                  const coordValue = block.metadata?.coord
                  if (coordValue) {
                    const parsedCoords = parseCoordinateMetadata(coordValue)
                    if (parsedCoords) {
                      block.coordinates = parsedCoords
                      delete block.metadata?.coord
                    }
                  }

                  if (block.metadata && Object.keys(block.metadata).length === 0) {
                    delete block.metadata
                  }
                }
                if (meta.tags) block.tags = meta.tags

                // remove meta list (first item) if it exists and was successfully parsed
                // only remove if we actually found metadata/tags and the first item matches [meta] pattern
                if (nestedList.children.length > 0 && (meta.metadata || meta.tags)) {
                  const firstItem = nestedList.children.find(isLi)
                  if (firstItem) {
                    const label = getFirstTextContent(firstItem).trim().toLowerCase()
                    const metaMatch = label.match(/^\[meta\](?:\s*[:\-–—])?$/)
                    if (metaMatch) {
                      // find the actual index and remove it
                      const metaIndex = nestedList.children.indexOf(firstItem)
                      if (metaIndex !== -1) {
                        nestedList.children.splice(metaIndex, 1)
                      }
                    }
                  }
                }

                // parse remaining items as sub-blocks
                const subItems: ArenaBlock[] = []
                for (const child of nestedList.children) {
                  if (isLi(child)) {
                    const subBlock = parseBlock(child, depth + 1)
                    if (subBlock) subItems.push(subBlock)
                  }
                }
                if (subItems.length > 0) block.subItems = subItems
              }

              // schedule twitter embed fetch
              if (url && twitterUrlRegex.test(url)) {
                const currentBlock = block
                embedPromises.push(
                  fetchTwitterEmbed(url, locale)
                    .then((html) => {
                      if (html) currentBlock.embedHtml = html
                    })
                    .catch(() => undefined),
                )
              }

              // handle youtube embed
              if (url && !block.embedHtml) {
                const youtubeEmbed = buildYouTubeEmbed(url)
                if (youtubeEmbed) {
                  const frameTitle = block.title ?? block.content ?? block.id
                  block.embedHtml = `<iframe class="arena-modal-iframe arena-modal-iframe-youtube" title="YouTube embed: ${frameTitle.replace(/"/g, "&quot;")}" loading="lazy" data-block-id="${block.id}" src="${youtubeEmbed.src}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>`
                }
              }

              // apply link processing (wikilinks, internal links)
              const applyLinkProcessing = (node: ElementContent): ElementContent => {
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
                    const resolved = parsed ? resolveWikilinkTarget(parsed, "" as FullSlug) : null

                    if (parsed && resolved) {
                      const href = (
                        parsed.anchor ? `/${resolved.slug}${parsed.anchor}` : `/${resolved.slug}`
                      ) as string
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
                          { type: "text", value: parsed.alias ?? parsed.target ?? match[0] },
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

                const visitEl = (el: ElementContent) => {
                  if (el.type !== "element") return
                  const e = el as Element

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
                }

                const cloned = cloneElementContent(node)
                visitEl(cloned)
                return cloned
              }

              // build titleHtmlNode from direct li children (with or without p wrapper)
              const buildTitleNode = (li: Element): ElementContent | undefined => {
                // Check if content is wrapped in <p>
                let contentChildren: ElementContent[] = li.children as ElementContent[]

                for (const child of contentChildren) {
                  if (isElement(child) && child.tagName === "p") {
                    contentChildren = child.children as ElementContent[]
                    break
                  }
                }

                // extract url from first <a> element
                let linkHref: string | undefined
                let linkElement: Element | undefined

                for (const child of contentChildren) {
                  if (isElement(child) && child.tagName === "a") {
                    linkElement = child
                    const href = child.properties?.href
                    if (typeof href === "string") {
                      linkHref = href
                    }
                    break
                  }
                }

                // extract title - can be text node OR element, or mixed content
                let titleText: string | undefined
                let foundSeparator = false
                const titleElements: ElementContent[] = []

                for (const child of contentChildren) {
                  if (isElement(child) && child.tagName === "ul") break

                  // Check for separator (emdash or double-hyphen)
                  if (child.type === "text") {
                    const text = child.value.trim()
                    // Check if this is just a separator
                    if (text === "—" || text === "--") {
                      foundSeparator = true
                      continue
                    }
                    // Or if it contains separator + title
                    const match = text.match(/^(?:—|--)\s*(.+)$/)
                    if (match && match[1]) {
                      titleText = stripTrailingMarkers(match[1]).trim()
                      foundSeparator = true
                      continue
                    }
                    // Collect text nodes after separator
                    if (foundSeparator && text.length > 0) {
                      titleElements.push(child)
                    }
                  }

                  // After finding separator, collect <a> elements (wikilinks)
                  if (
                    foundSeparator &&
                    isElement(child) &&
                    child.tagName === "a" &&
                    child !== linkElement
                  ) {
                    titleElements.push(child)
                  }
                }

                // Process title if we have url and title content
                if (linkHref && (titleText || titleElements.length > 0)) {
                  const wrapper: Element = {
                    type: "element",
                    tagName: "span",
                    properties: {},
                    children: [],
                  }

                  if (titleText) {
                    wrapper.children.push({ type: "text", value: titleText })
                  }

                  // Add collected elements (text nodes and wikilinks)
                  wrapper.children.push(...(titleElements as any[]))

                  // Apply link processing to handle any wikilinks in text
                  return applyLinkProcessing(wrapper)
                }

                // fallback: collect all content before nested ul
                const collected: ElementContent[] = []
                for (const child of li.children as ElementContent[]) {
                  if (isElement(child) && child.tagName === "ul") break
                  const cloned = cloneElementContent(child)
                  collected.push(cloned)
                }

                if (collected.length === 0) return undefined

                const wrapper: Element = {
                  type: "element",
                  tagName: "span",
                  properties: {},
                  children: collected,
                }

                return applyLinkProcessing(wrapper)
              }

              block.titleHtmlNode = buildTitleNode(li)

              // aggregate all non-ul children for htmlNode
              const aggregateListItem = (li: Element): ElementContent => {
                const collected: ElementContent[] = []
                for (const child of li.children as ElementContent[]) {
                  if (isElement(child) && child.tagName === "ul") continue
                  const cloned = cloneElementContent(child)
                  collected.push(cloned)
                }
                const wrapped: ElementContent = {
                  type: "element",
                  tagName: "div",
                  properties: {},
                  children: collected,
                }
                return applyLinkProcessing(wrapped)
              }

              block.htmlNode = aggregateListItem(li)

              // find internal link
              const findInternalLink = (
                node?: ElementContent,
              ):
                | {
                    slug: string
                    href: string
                    hash?: string
                  }
                | undefined => {
                if (!node || node.type !== "element") return undefined
                const el = node as Element

                if (el.tagName === "a") {
                  const classes = Array.isArray(el.properties?.className)
                    ? el.properties.className
                    : []
                  const hasInternal = classes.some((c) => c === "internal")
                  if (hasInternal) {
                    const slug = el.properties?.["data-slug"]
                    if (typeof slug === "string" && slug.length > 0) {
                      const rawHref = el.properties?.href
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
                }

                if (el.children) {
                  for (const child of el.children as ElementContent[]) {
                    const found = findInternalLink(child)
                    if (found) return found
                  }
                }

                return undefined
              }

              const internalLinkInfo =
                findInternalLink(block.titleHtmlNode) ?? findInternalLink(block.htmlNode)
              if (internalLinkInfo) {
                block.internalSlug = internalLinkInfo.slug
                block.internalHref = internalLinkInfo.href
                block.internalHash = internalLinkInfo.hash
              }

              return block
            }

            // parse structure from hast tree
            const bodyChildren = tree.children.filter(
              (child) => child.type !== "doctype",
            ) as RootContent[]

            for (let i = 0; i < bodyChildren.length; i++) {
              const node = bodyChildren[i]

              // h2 starts a new channel
              if (isH2(node)) {
                let name = toString(node).trim()

                // check if h2 contains a link and extract name from it
                const linkInH2 = node.children.find((ch) => isElement(ch) && ch.tagName === "a")
                if (linkInH2 && isElement(linkInH2)) {
                  const href = linkInH2.properties?.href
                  if (typeof href === "string" && !/^https?:\/\//i.test(href)) {
                    try {
                      const parts = decodeURI(href).split("/")
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

                // handle titleHtmlNode if h2 contains anchor
                if (elementContainsAnchor(node)) {
                  const span: Element = {
                    type: "element",
                    tagName: "span",
                    properties: {},
                    children: (node.children ?? []).map((child) =>
                      cloneElementContent(child as ElementContent),
                    ),
                  }
                  channel.titleHtmlNode = span
                }

                channels.push(channel)
              } else if (isUl(node)) {
                // ul following h2 contains blocks
                if (channels.length > 0) {
                  const currentChannel = channels[channels.length - 1]
                  const ulElement = node as Element

                  const channelMeta = extractMetadataFromList(ulElement)
                  if (channelMeta.metadata) {
                    currentChannel.metadata = channelMeta.metadata
                  }
                  if (channelMeta.tags) {
                    currentChannel.tags = channelMeta.tags
                  }
                  if (channelMeta.metadata || channelMeta.tags) {
                    const firstItem = ulElement.children.find(isLi)
                    if (firstItem) {
                      const label = getFirstTextContent(firstItem).trim().toLowerCase()
                      const metaMatch = label.match(/^\[meta\](?:\s*[:\-–—])?$/)
                      if (metaMatch) {
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

            // wait for all twitter embeds
            if (embedPromises.length > 0) {
              await Promise.all(embedPromises)
            }

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
