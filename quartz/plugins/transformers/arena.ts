import { QuartzTransformerPlugin } from "../../types/plugin"
import { Element, ElementContent, Root as HastRoot, RootContent } from "hast"
import { toString } from "hast-util-to-string"
import Slugger from "github-slugger"
import { externalLinkRegex } from "./ofm"
import { fetchTwitterEmbed, twitterUrlRegex } from "./twitter"
import { splitAnchor, transformLink, stripSlashes, FullSlug } from "../../util/path"
import { createWikilinkRegex, parseWikilink, resolveWikilinkTarget } from "../../util/wikilinks"
import { buildYouTubeEmbed } from "../../util/youtube"
import yaml from "js-yaml"

export interface ArenaBlock {
  id: string
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
  coordinates?: {
    lat: number
    lon: number
  }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  internalTitle?: string
  tags?: string[]
  embedDisabled?: boolean
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
  coordinates?: {
    lat: number
    lon: number
  }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  internalTitle?: string
  tags?: string[]
  subItems?: ArenaBlockSearchable[]
  hasModalInDom: boolean
  embedDisabled?: boolean
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

declare module "vfile" {
  interface DataMap {
    arenaData?: ArenaData
    arenaChannel?: ArenaChannel
  }
}

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

const isGithubUrl = (rawUrl: string): boolean => {
  if (!rawUrl) return false
  try {
    const { hostname } = new URL(rawUrl)
    const normalized = hostname.toLowerCase()
    return normalized === "github.com" || normalized.endsWith(".github.com")
  } catch {
    return rawUrl.toLowerCase().includes("github.com/")
  }
}

const getTextContentExcludingNestedUl = (li: Element): string => {
  let text = ""
  for (const child of li.children as ElementContent[]) {
    if (isElement(child) && child.tagName === "ul") continue
    text += toString(child)
  }
  return text.trim()
}

const cloneElementContent = <T extends ElementContent>(node: T): T => {
  return typeof structuredClone === "function"
    ? structuredClone(node)
    : (JSON.parse(JSON.stringify(node)) as T)
}

const COORDINATE_NUMBER_PATTERN = /-?\d+(?:\.\d+)?/g

const parseCoordinateMetadata = (value: string): { lat: number; lon: number } | null => {
  const matches = value.match(COORDINATE_NUMBER_PATTERN)
  if (!matches || matches.length < 2) {
    return null
  }

  const lat = Number.parseFloat(matches[0])
  const lon = Number.parseFloat(matches[1])

  if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
    return null
  }

  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    return null
  }

  return { lon, lat }
}

const elementContainsAnchor = (node: ElementContent): boolean => {
  if (node.type !== "element") return false
  if (node.tagName === "a") return true
  return node.children.some((child) => elementContainsAnchor(child))
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
      const value = child.value.trim()
      if (value.length > 0) {
        return value
      }
      continue
    }
    if (isElement(child)) {
      const text = getFirstTextContent(child)
      if (text.trim().length > 0) {
        return text
      }
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

const appendListToYaml = (list: Element, indent: number, lines: string[]): void => {
  const indentStr = "  ".repeat(indent)

  for (const child of list.children) {
    if (!isLi(child)) continue

    const nested = extractNestedList(child)
    let rawText = ""
    for (const ch of child.children as ElementContent[]) {
      if (isElement(ch) && ch.tagName === "ul") continue
      rawText += toString(ch)
    }
    rawText = rawText.trim()

    if (nested) {
      if (rawText.length > 0) {
        lines.push(`${indentStr}${rawText}`)
      } else {
        lines.push(`${indentStr}-`)
      }
      appendListToYaml(nested, indent + 1, lines)
      continue
    }

    if (rawText.length === 0) continue
    const normalized = rawText.replace(/^\-+\s*/, "").trim()

    if (normalized.includes(":")) {
      lines.push(`${indentStr}${normalized}`)
    } else {
      lines.push(`${indentStr}- ${normalized}`)
    }
  }
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
              metadata?: Record<string, unknown>
              tags?: string[]
            } => {
              if (list.children.length === 0) return {}

              const firstItem = list.children.find(isLi)
              if (!firstItem) return {}

              const label = getFirstTextContent(firstItem).trim().toLowerCase()
              if (!/^\[meta\](?:\s*[:\-–—])?$/.test(label)) return {}

              const metaList = extractNestedList(firstItem)
              if (!metaList || metaList.children.length === 0) return {}

              const metadata: Record<string, unknown> = {}
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
                    } else {
                      keySource = normalized
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

                if (sublist && !value) {
                  const yamlLines: string[] = []
                  appendListToYaml(sublist, 0, yamlLines)
                  const yamlSource = yamlLines.join("\n")

                  if (yamlSource.trim().length > 0) {
                    const parsed = yaml.load(yamlSource)
                    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                      metadata[normalizedKey] = parsed
                      continue
                    }
                  }
                }

                if (!value || value.length === 0) continue
                const key = normalizedKey
                if (key.length === 0) continue
                metadata[key] = value
              }

              const result: { metadata?: Record<string, unknown>; tags?: string[] } = {}
              if (Object.keys(metadata).length > 0) result.metadata = metadata
              if (tags.length > 0) result.tags = tags
              return result
            }

            const normalizeChannelMetadata = (
              metadata: Record<string, unknown>,
            ): Record<string, string | boolean> => {
              const normalized: Record<string, string | boolean> = {}
              for (const [key, value] of Object.entries(metadata)) {
                if (typeof value === "string" || typeof value === "boolean") {
                  normalized[key] = value
                } else if (typeof value === "number" && Number.isFinite(value)) {
                  normalized[key] = String(value)
                }
              }
              return normalized
            }

            const parseBlock = (li: Element, depth = 0): ArenaBlock | null => {
              const textContent = getTextContentExcludingNestedUl(li)

              const trailingSection = textContent.match(TRAILING_MARKERS_PATTERN)?.[0] ?? ""
              const highlighted = HIGHLIGHT_MARKER.test(trailingSection)
              const embedDisabledFromMarker = EMBED_DISABLED_MARKER.test(trailingSection)
              const strippedContent = stripTrailingMarkers(textContent)

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
                const linkText = stripTrailingMarkers(toString(firstLink).trim())
                if (linkText.length > 0) titleCandidate = linkText

                const href = firstLink.properties?.href
                if (typeof href === "string" && /^https?:\/\//.test(href)) {
                  url = href
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

              const block: ArenaBlock = {
                id: `block-${blockCounter++}`,
                content: titleCandidate || strippedContent || url || "",
                title: titleCandidate,
                url,
                highlighted,
                embedDisabled: embedDisabledFromMarker || (url ? isGithubUrl(url) : false),
              }

              const nestedList = extractNestedList(li)
              if (nestedList) {
                const meta = extractMetadataFromList(nestedList)
                if (meta.metadata) {
                  block.metadata = meta.metadata

                  const coordValue = block.metadata?.coord
                  if (typeof coordValue === "string" && coordValue.length > 0) {
                    const parsedCoords = parseCoordinateMetadata(coordValue)
                    if (parsedCoords) {
                      block.coordinates = parsedCoords
                      delete block.metadata?.coord
                    }
                  }

                  const pinnedValue = block.metadata?.pinned
                  if (pinnedValue !== undefined && depth === 0) {
                    const normalizedPinned =
                      typeof pinnedValue === "string"
                        ? pinnedValue.toLowerCase().trim()
                        : pinnedValue === true
                          ? "true"
                          : pinnedValue === false
                            ? "false"
                            : undefined
                    if (normalizedPinned === "true" || normalizedPinned === "yes") {
                      block.pinned = true
                      delete block.metadata?.pinned
                    } else if (normalizedPinned === "false" || normalizedPinned === "no") {
                      delete block.metadata?.pinned
                    }
                  }

                  const laterValue = block.metadata?.later
                  if (laterValue !== undefined && depth === 0) {
                    const normalizedLater =
                      typeof laterValue === "string"
                        ? laterValue.toLowerCase().trim()
                        : laterValue === true
                          ? "true"
                          : laterValue === false
                            ? "false"
                            : undefined
                    if (normalizedLater === "true" || normalizedLater === "yes") {
                      block.later = true
                      delete block.metadata?.later
                    } else if (normalizedLater === "false" || normalizedLater === "no") {
                      delete block.metadata?.later
                    }
                  }

                  const importanceValue = block.metadata?.importance
                  if (importanceValue !== undefined) {
                    const parsed =
                      typeof importanceValue === "number"
                        ? importanceValue
                        : typeof importanceValue === "string"
                          ? Number.parseFloat(importanceValue)
                          : Number.NaN
                    if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
                      block.importance = parsed
                    }
                    delete block.metadata?.importance
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

              if (url && twitterUrlRegex.test(url)) {
                embedPromises.push(
                  fetchTwitterEmbed(url, locale)
                    .then((html) => {
                      if (html) block.embedHtml = html
                    })
                    .catch(() => undefined),
                )
              }

              if (url && !block.embedHtml) {
                const youtubeEmbed = buildYouTubeEmbed(url)
                if (youtubeEmbed) {
                  block.embedHtml = `<iframe class="arena-modal-iframe arena-modal-iframe-youtube" title="YouTube embed: ${(block.title ?? block.content ?? block.id).replace(/"/g, "&quot;")}" loading="lazy" data-block-id="${block.id}" src="${youtubeEmbed.src}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>`
                }
              }

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

              const buildTitleNode = (li: Element): ElementContent | undefined => {
                let contentChildren: ElementContent[] = li.children as ElementContent[]

                for (const child of contentChildren) {
                  if (isElement(child) && child.tagName === "p") {
                    contentChildren = child.children as ElementContent[]
                    break
                  }
                }

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

                let titleText: string | undefined
                let foundSeparator = false
                const titleElements: ElementContent[] = []

                for (const child of contentChildren) {
                  if (isElement(child) && child.tagName === "ul") break

                  if (child.type === "text") {
                    const text = child.value.trim()
                    if (text === "—" || text === "--") {
                      foundSeparator = true
                      continue
                    }
                    const match = text.match(/^(?:—|--)\s*(.+)$/)
                    if (match && match[1]) {
                      titleText = stripTrailingMarkers(match[1]).trim()
                      foundSeparator = true
                      continue
                    }
                    if (foundSeparator && text.length > 0) {
                      titleElements.push(child)
                    }
                  }

                  if (
                    foundSeparator &&
                    isElement(child) &&
                    child.tagName === "a" &&
                    child !== linkElement
                  ) {
                    titleElements.push(child)
                  }
                }

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

                  wrapper.children.push(...(titleElements as any[]))

                  return applyLinkProcessing(wrapper)
                }

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

              const findInternalLink = (
                node?: ElementContent,
              ):
                | {
                    slug: string
                    href: string
                    title: string
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
                        title: toString(el),
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
                block.internalTitle = internalLinkInfo.title
              }

              return block
            }

            const bodyChildren = tree.children.filter(
              (child) => child.type !== "doctype",
            ) as RootContent[]

            for (let i = 0; i < bodyChildren.length; i++) {
              const node = bodyChildren[i]

              if (isH2(node)) {
                let name = toString(node).trim()

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

            file.data.arenaData = { channels }
          }
        },
      ]
    },
  }
}
