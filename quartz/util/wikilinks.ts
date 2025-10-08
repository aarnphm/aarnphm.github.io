import path from "path"
import isAbsoluteUrl from "is-absolute-url"

import { Image, Link, Html } from "mdast"
import {
  FilePath,
  FullSlug,
  getFileExtension,
  slugAnchor,
  slugifyFilePath,
  splitAnchor,
  stripSlashes,
} from "./path"
import { WikilinkParsed, Wikilink } from "./micromark-extension-wikilink/fromMarkdown"

// re-export WikilinkParsed for external consumers
export type { WikilinkParsed, Wikilink }

// !?                 -> optional embedding
// \[\[               -> open brace
// ([^\[\]\|\#]+)     -> one or more non-special characters ([,],|, or #) (name)
// (#[^\[\]\|\#]+)?   -> # then one or more non-special characters (heading link)
// (\\?\|[^\[\]\#]+)? -> optional escape \ then | then zero or more non-special characters (alias)
export const WIKILINK_PATTERN = /!?\[\[([^\[\]\|#\\]+)?(#+[^\[\]\|#\\]+)?(\\?\|[^\[\]#]*)?\]\]/

/**
 * Create a fresh wikilink regex. Consumers should prefer using this helper instead of sharing
 * a singleton RegExp to avoid `lastIndex` state across different callers.
 */
export function createWikilinkRegex(flags: string = "g"): RegExp {
  const normalizedFlags = Array.from(new Set((flags + "g").split(""))).join("")
  return new RegExp(WIKILINK_PATTERN.source, normalizedFlags)
}

const singleWikilinkRegex = new RegExp(`^${WIKILINK_PATTERN.source}$`, "i")

export function parseWikilink(raw: string): WikilinkParsed | null {
  const trimmed = raw.trim()
  const match = trimmed.match(singleWikilinkRegex)
  if (!match) {
    return null
  }

  const isEmbed = trimmed.startsWith("!")

  const [, rawFp, rawHeader, rawAlias] = match

  const [_target, anchor] = splitAnchor(`${rawFp ?? ""}${rawHeader ?? ""}`)
  const blockRef = rawHeader?.startsWith("#^") ? "^" : ""
  const displayAnchor = anchor ? `#${blockRef}${anchor.trim().replace(/^#+/, "")}` : undefined
  const alias = rawAlias ? rawAlias.replace(/^\\?\|/, "") : undefined

  return {
    raw: trimmed,
    target: rawFp ?? "",
    anchor: displayAnchor,
    alias,
    embed: isEmbed,
  }
}

export function extractWikilinks(text: string): WikilinkParsed[] {
  const links: WikilinkParsed[] = []
  const regex = createWikilinkRegex()
  let match: RegExpExecArray | null

  while ((match = regex.exec(text)) !== null) {
    const parsed = parseWikilink(match[0])
    if (parsed) {
      links.push(parsed)
    }
  }

  return links
}

export interface ResolvedWikilinkTarget {
  slug: FullSlug
  anchor?: string
}

function ensureFilePath(target: string): FilePath {
  if (target.length === 0) {
    return "index.md" as FilePath
  }

  if (target.endsWith("/")) {
    return `${target}index.md` as FilePath
  }

  const ext = getFileExtension(target as FilePath)
  if (ext) {
    return target as FilePath
  }

  return `${target}.md` as FilePath
}

export function resolveWikilinkTarget(
  link: WikilinkParsed,
  currentSlug: FullSlug,
): ResolvedWikilinkTarget | null {
  const baseSlug = stripSlashes(currentSlug)

  if (link.target && isAbsoluteUrl(link.target)) {
    return null
  }

  const normalizedTarget = link.target?.replace(/\\/g, "/") ?? ""

  let combined: string

  if (normalizedTarget.startsWith("/")) {
    combined = normalizedTarget.slice(1)
  } else if (normalizedTarget.length === 0) {
    combined = baseSlug
  } else {
    const dir = path.posix.dirname(baseSlug)
    const baseDir = dir === "." ? "" : dir
    combined = path.posix.join(baseDir, normalizedTarget)
  }

  let normalized = path.posix.normalize(combined)
  while (normalized.startsWith("../")) {
    normalized = normalized.slice(3)
  }
  if (normalized.startsWith("./")) {
    normalized = normalized.slice(2)
  }

  const filePath = ensureFilePath(normalized)
  const slug = slugifyFilePath(stripSlashes(filePath) as FilePath)

  return {
    slug,
    anchor: link.anchor,
  }
}

export function stripWikilinkFormatting(text: string): string {
  if (!text) {
    return text
  }

  return text.replace(createWikilinkRegex(), (value) => {
    const parsed = parseWikilink(value)
    if (!parsed) return value
    return parsed.alias ?? parsed.target
  })
}

/**
 * Normalize text for fuzzy matching.
 * Handles LaTeX, whitespace, case sensitivity.
 */
export function normalizeForMatching(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/\\/g, "") // remove backslashes
    .replace(/\$/g, "") // remove $ symbols
    .replace(/[^\w\s-]/g, "") // keep only word chars, spaces, and -
    .replace(/\s+/g, " ") // collapse whitespace (after other replacements)
    .trim() // trim again after collapsing
}

/**
 * Resolve wikilink anchor text to a heading slug.
 *
 * Matches Obsidian's behavior:
 * - LaTeX normalization: "architectural skeleton of $ mu$" → "architectural-skeleton-of-mu"
 * - Nested paths: "Parent#Child#Grandchild" uses only the last segment ("Grandchild")
 * - No validation of document structure or parent/child relationships
 *
 * The browser handles anchor navigation - we just normalize the text to match
 * the slugs that github-slugger produces for headings.
 *
 * @param anchorText - The anchor portion of a wikilink (e.g., "section" or "Parent#Child")
 * @returns The normalized slug (e.g., "section" or "child")
 *
 * @example
 * resolveAnchor("Section") → "section"
 * resolveAnchor("NVIDIA#cuda") → "cuda"
 * resolveAnchor("architectural skeleton of $ mu$") → "architectural-skeleton-of-mu"
 * resolveAnchor("Parent#Child#Grandchild") → "grandchild"
 */
export function resolveAnchor(anchorText: string): string {
  // if anchor contains #, take the last segment (Obsidian behavior)
  let text = anchorText.trim()
  if (text.includes("#")) {
    const segments = text.split("#")
    text = segments[segments.length - 1].trim()
  }
  // slugify using github-slugger (same as heading slugs)
  return slugAnchor(text)
}

/**
 * mdast node representing an Obsidian wikilink.
 * extends standard mdast Link but preserves wikilink-specific metadata.
 *
 * note: this is the legacy representation used by wikilinkToMdast().
 * the micromark extension in ./micromark-extension-wikilink/ creates
 * a different node type during parsing. this interface is used when
 * converting wikilink nodes to Link nodes in the OFM transformer.
 */
export interface WikilinkNode extends Link {
  data: Link["data"] & {
    wikilink: WikilinkParsed
  }
}

/**
 * mdast node representing an embedded wikilink (transclusion).
 * can be an image, video, audio, pdf, or block transclude.
 */
export interface EmbedNode {
  type: "html" | "image"
  value?: string
  url?: string
  data?: {
    hProperties?: Record<string, any>
    transclude?: boolean
  }
}

/**
 * result of converting a WikilinkParsed to mdast nodes.
 * can be a link, image, video, audio, pdf, or transclude block.
 */
export type WikilinkResult = WikilinkNode | Image | Html | null

/**
 * configuration for wikilink conversion.
 */
export interface WikilinkToMdastOptions {
  /**
   * all slugs in the vault, used for broken link detection.
   */
  allSlugs?: string[]
  /**
   * whether to mark broken wikilinks with a special class.
   */
  enableBrokenWikilinks?: boolean
  /**
   * base path for relative URLs (pathToRoot equivalent).
   */
  basePath?: string
}

/**
 * extension dimensions from image embed alias.
 * matches patterns like "100x200" or "100" in `![[image.png|100x200]]`.
 */
const wikilinkImageEmbedRegex = new RegExp(
  /^(?<alt>(?!^\d*x?\d*$).*?)?(\|?\s*?(?<width>\d+)(x(?<height>\d+))?)?$/,
)

/**
 * determine if a file extension is an image format.
 */
function isImageExtension(ext: string): boolean {
  return [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"].includes(ext)
}

/**
 * determine if a file extension is a video format.
 */
function isVideoExtension(ext: string): boolean {
  return [".mp4", ".webm", ".ogv", ".mov", ".mkv"].includes(ext)
}

/**
 * determine if a file extension is an audio format.
 */
function isAudioExtension(ext: string): boolean {
  return [".mp3", ".wav", ".m4a", ".ogg", ".3gp", ".flac"].includes(ext)
}

/**
 * convert a WikilinkParsed to an mdast node.
 * handles all Obsidian link types: basic links, embeds, images, videos, audio, pdfs, transclusions.
 *
 * @param wikilink - parsed wikilink from quartz/util/wikilinks.ts
 * @param options - configuration options
 * @returns mdast node or null if invalid
 */
export function wikilinkToMdast(
  wikilink: WikilinkParsed,
  options: WikilinkToMdastOptions = {},
): WikilinkResult {
  const { target, anchor, alias, embed } = wikilink
  const { allSlugs, enableBrokenWikilinks } = options

  const fp = target.trim()
  const displayAnchor = anchor?.trim() ?? ""

  // handle embeds (transclusions)
  if (embed) {
    const ext = path.extname(fp).toLowerCase()
    const url = slugifyFilePath(fp as FilePath)

    // image embeds
    if (isImageExtension(ext)) {
      const match = wikilinkImageEmbedRegex.exec(alias ?? "")
      const alt = match?.groups?.alt ?? ""
      const width = match?.groups?.width ?? "auto"
      const height = match?.groups?.height ?? "auto"

      return {
        type: "image",
        url,
        data: {
          hProperties: {
            width,
            height,
            alt,
          },
        },
      } as Image
    }

    // video embeds
    if (isVideoExtension(ext)) {
      return {
        type: "html",
        value: `<video src="${url}" controls loop></video>`,
      } as Html
    }

    // audio embeds
    if (isAudioExtension(ext)) {
      return {
        type: "html",
        value: `<audio src="${url}" controls></audio>`,
      } as Html
    }

    // pdf embeds
    if (ext === ".pdf") {
      return {
        type: "html",
        value: `<iframe src="${url}" class="pdf"></iframe>`,
      } as Html
    }

    // default: block transclude
    // note: we use hastscript in ofm.ts, but here we generate raw HTML
    // to avoid circular dependencies. the actual hastscript conversion
    // happens in ofm.ts when it calls this function.
    const block = displayAnchor
    return {
      type: "html",
      data: { hProperties: { transclude: true } },
      value: `<blockquote class="transclude" data-url="${url}" data-block="${block}" data-embed-alias="${alias ?? ""}"><a class="transclude-inner" href="${url}${displayAnchor}">Transclude of ${url} ${block}</a></blockquote>`,
    } as Html
  }

  // regular internal link
  const url = fp + displayAnchor
  const displayText = alias ?? fp

  // broken link detection
  if (enableBrokenWikilinks && allSlugs) {
    const slug = slugifyFilePath(fp as FilePath)
    const exists = allSlugs.includes(slug)
    if (!exists) {
      return {
        type: "link",
        url,
        data: {
          hProperties: {
            className: ["broken"],
          },
          wikilink,
        },
        children: [
          {
            type: "text",
            value: displayText,
          },
        ],
      } as WikilinkNode
    }
  }

  // standard internal link
  return {
    type: "link",
    url,
    data: {
      wikilink,
    },
    children: [
      {
        type: "text",
        value: displayText,
      },
    ],
  } as WikilinkNode
}

/**
 * escape wikilinks in table context.
 * obsidian requires escaping pipes and hashes inside table cells.
 *
 * @param wikilink - raw wikilink string
 * @returns escaped wikilink suitable for tables
 */
export function escapeWikilinkForTable(wikilink: string): string {
  let escaped = wikilink
  // escape hash for headers
  escaped = escaped.replace("#", "\\#")
  // escape pipe characters if not already escaped
  // regex: match pipe that's not preceded by backslash (or preceded by even number of backslashes)
  escaped = escaped.replace(/((^|[^\\])(\\\\)*)\|/g, "$1\\|")
  return escaped
}
