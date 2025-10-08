import { Image, Link, Html } from "mdast"
import { ParsedWikilink } from "./wikilinks"
import { FilePath, slugifyFilePath } from "./path"
import path from "path"

/**
 * mdast node representing an Obsidian wikilink.
 * extends standard mdast Link but preserves wikilink-specific metadata.
 */
export interface WikilinkNode extends Link {
  data: Link["data"] & {
    wikilink: ParsedWikilink
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
 * result of converting a ParsedWikilink to mdast nodes.
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
 * convert a ParsedWikilink to an mdast node.
 * handles all Obsidian link types: basic links, embeds, images, videos, audio, pdfs, transclusions.
 *
 * @param wikilink - parsed wikilink from quartz/util/wikilinks.ts
 * @param options - configuration options
 * @returns mdast node or null if invalid
 */
export function wikilinkToMdast(
  wikilink: ParsedWikilink,
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
 * check if a text node or string contains wikilinks.
 * useful for early optimization before running expensive regex operations.
 *
 * @param text - text to check
 * @returns true if text likely contains wikilinks
 */
export function containsWikilinks(text: string): boolean {
  return text.includes("[[") && text.includes("]]")
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

/**
 * represents a position in source text for error reporting.
 */
export interface SourcePosition {
  start: {
    line: number
    column: number
    offset: number
  }
  end: {
    line: number
    column: number
    offset: number
  }
}

/**
 * create a source position from regex match result.
 * useful for preserving source location info in AST nodes.
 *
 * @param match - regex match result
 * @param lineOffset - line number offset (0-based)
 * @returns position object for mdast nodes
 */
export function createPositionFromMatch(
  match: RegExpExecArray,
  lineOffset: number = 0,
): SourcePosition {
  const start = match.index
  const end = start + match[0].length

  return {
    start: {
      line: lineOffset + 1,
      column: start + 1,
      offset: start,
    },
    end: {
      line: lineOffset + 1,
      column: end + 1,
      offset: end,
    },
  }
}
