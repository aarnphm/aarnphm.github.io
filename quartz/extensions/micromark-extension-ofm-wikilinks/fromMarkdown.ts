/**
 * mdast-util-from-markdown extension for wikilinks.
 * converts micromark tokens to Wikilink AST nodes.
 */

import type { Element as HastElement, Text as HastText } from "hast"
import type { Extension, CompileContext, Token } from "mdast-util-from-markdown"
import { Literal } from "unist"
import {
  FilePath,
  FullSlug,
  getFileExtension,
  slugAnchor,
  stripSlashes,
  sluggify,
  endsWith,
} from "../../util/path"
import "./types"
import { buildYouTubeEmbed } from "../../util/youtube"

export interface WikilinkData {
  raw: string
  target: string
  anchor?: string
  anchorText?: string // original un-slugified anchor text for display
  metadata?: string
  metadataParsed?: Record<string, any>
  alias?: string
  embed: boolean
}

/**
 * mdast node for wikilinks.
 * extends standard Link with wikilink-specific metadata.
 * data.hName/hProperties/hChildren enable automatic hast conversion.
 */
export interface Wikilink extends Literal {
  type: "wikilink"
  value: string
  data?: {
    wikilink: WikilinkData
    hName?: string
    hProperties?: Record<string, any>
    hChildren?: (HastElement | HastText)[]
  }
  position?: {
    start: { line: number; column: number; offset: number }
    end: { line: number; column: number; offset: number }
  }
}

export interface FromMarkdownOptions {
  /**
   * enable Obsidian-style nested anchor handling and automatic hast conversion.
   * default: true
   *
   * when true: uses internal slugification and annotates nodes for automatic HTML conversion
   * when false: returns raw wikilink nodes without hName annotations
   *
   * see https://help.obsidian.md/links for more information on obsidian link behavior.
   */
  obsidian?: boolean

  /**
   * file extensions to strip before slugifying.
   * default: ['.md', '.base']
   * only applies when obsidian: true
   */
  stripExtensions?: string[]

  /**
   * function to check if a slug exists in the content index.
   * used to validate implicit alias splitting.
   */
  hasSlug?: (slug: string) => boolean
}

/**
 * file extension constants for media types
 */
const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]
const VIDEO_EXTS = [".mp4", ".webm", ".ogv", ".mov", ".mkv"]
const AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".ogg", ".3gp", ".flac"]

/**
 * determine if a file extension is an image format.
 */
function isImageExtension(ext: string): boolean {
  return IMAGE_EXTS.includes(ext.toLowerCase())
}

/**
 * determine if a file extension is a video format.
 */
function isVideoExtension(ext: string): boolean {
  return VIDEO_EXTS.includes(ext.toLowerCase())
}

/**
 * determine if a file extension is an audio format.
 */
function isAudioExtension(ext: string): boolean {
  return AUDIO_EXTS.includes(ext.toLowerCase())
}

/**
 * simple path slugifier using slugAnchor.
 * strips specified extensions before slugifying.
 * lowercases output for consistency with obsidian behavior.
 */
function slugifyFilePath(path: string, stripExts: string[]): string {
  let fp = stripSlashes(path) as FilePath
  let ext = getFileExtension(fp)
  const withoutFileExt = fp.replace(new RegExp(ext + "$"), "")
  if ([".md", ".base", ...stripExts, undefined].includes(ext)) {
    ext = ""
  }
  let slug = sluggify(withoutFileExt)

  // treat _index as index
  if (endsWith(slug, "_index")) {
    slug = slug.replace(/_index$/, "index")
  }
  return (slug + ext) as FullSlug
}

/**
 * create mdast-util-from-markdown extension for wikilinks.
 */
export function wikilinkFromMarkdown(options: FromMarkdownOptions = {}): Extension {
  const { obsidian = true } = options

  return {
    enter: {
      wikilink: enterWikilink,
      wikilinkEmbedMarker: enterEmbedMarker,
      wikilinkTarget: enterTarget,
      wikilinkAnchor: enterAnchor,
      wikilinkMetadata: enterMetadata,
      wikilinkAlias: enterAlias,
    },
    exit: {
      wikilink: function (this: CompileContext, token: Token): undefined {
        return exitWikilink.call(this, token, options)
      },
      wikilinkTarget: exitTarget,
      wikilinkAnchor: function (this: CompileContext, token: Token): undefined {
        return exitAnchor.call(this, token, obsidian)
      },
      wikilinkMetadata: exitMetadata,
      wikilinkAlias: exitAlias,
    },
  }
}

/**
 * enter wikilink container.
 * creates the Wikilink and pushes it onto the stack.
 */
function enterWikilink(this: CompileContext, token: Token): undefined {
  const node: Wikilink = {
    type: "wikilink",
    value: "",
    data: {
      wikilink: {
        raw: "",
        target: "",
        embed: false,
      },
    },
  }
  // @ts-expect-error: custom node type not in base mdast
  this.enter(node, token)
  return undefined
}

/**
 * enter embed marker `!`.
 * marks the current node as an embed.
 */
function enterEmbedMarker(this: CompileContext): undefined {
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink
  if (node.data?.wikilink) {
    node.data.wikilink.embed = true
  }
  return undefined
}

/**
 * enter target component.
 * initializes target buffer.
 */
function enterTarget(this: CompileContext): undefined {
  this.buffer()
  return undefined
}

/**
 * exit target component.
 * captures the target text from buffer.
 */
function exitTarget(this: CompileContext): undefined {
  const target = this.resume()
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink
  if (node.data?.wikilink) {
    node.data.wikilink.target = target
  }
  return undefined
}

/**
 * enter anchor component.
 * initializes anchor buffer.
 */
function enterAnchor(this: CompileContext): undefined {
  this.buffer()
  return undefined
}

/**
 * exit anchor component.
 * captures the anchor text from buffer and formats it.
 *
 * when obsidian mode is enabled, implements Obsidian's nested heading behavior:
 * - "Parent#Child#Grandchild" → uses only "Grandchild"
 * - "path/to/file heading text" → uses only "heading text" (implicit alias in anchors)
 * - applies slugification using github-slugger
 */
function exitAnchor(this: CompileContext, _token: Token, obsidian: boolean = false): undefined {
  let anchorText = this.resume()
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink

  if (node.data?.wikilink) {
    // preserve block reference marker ^
    const isBlockRef = anchorText.startsWith("^")
    const blockMarker = isBlockRef ? "^" : ""
    let cleanAnchor = isBlockRef ? anchorText.slice(1) : anchorText

    // handle Obsidian's implicit alias in anchors: "#path/to/file heading text"
    // similar to target implicit alias, extract heading text after space following path
    if (obsidian && cleanAnchor.includes("/") && cleanAnchor.includes(" ")) {
      const lastSlash = cleanAnchor.lastIndexOf("/")
      const afterSlash = cleanAnchor.substring(lastSlash + 1)
      const spaceIndex = afterSlash.indexOf(" ")

      if (spaceIndex >= 0) {
        // extract the heading text part after the space
        const headingText = afterSlash.substring(spaceIndex + 1).trim()
        if (headingText) {
          cleanAnchor = headingText
        }
      }
    }

    // obsidian-style nested heading resolution
    if (obsidian && cleanAnchor.includes("#")) {
      // take only the last segment after splitting by #
      const segments = cleanAnchor.split("#")
      cleanAnchor = segments[segments.length - 1].trim()
    }

    // normalize inline math delimiters: strip $ and trim content
    // "$ mu$" → "mu", "architectural skeleton of $ mu$" → "architectural skeleton of mu"
    cleanAnchor = cleanAnchor
      .replace(/\$([^$]+)\$/g, (_, content) => content.trim())
      .replace(/\s+/g, " ")
      .trim()

    // store original anchor text for display before slugification
    node.data.wikilink.anchorText = cleanAnchor

    // apply slugification for consistency with heading IDs
    const slugifiedAnchor = obsidian ? slugAnchor(cleanAnchor) : cleanAnchor

    // format as #anchor or #^block
    node.data.wikilink.anchor = `#${blockMarker}${slugifiedAnchor}`
  }
  return undefined
}

/**
 * enter alias component.
 * initializes alias buffer.
 */
function enterAlias(this: CompileContext): undefined {
  this.buffer()
  return undefined
}

/**
 * exit alias component.
 * captures the alias text from buffer.
 */
function exitAlias(this: CompileContext): undefined {
  const alias = this.resume()
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink
  if (node.data?.wikilink) {
    node.data.wikilink.alias = alias
  }
  return undefined
}

/**
 * enter metadata component.
 * initializes metadata buffer.
 */
function enterMetadata(this: CompileContext): undefined {
  this.buffer()
  return undefined
}

/**
 * parse JSON5-style metadata string.
 * handles unquoted keys, trailing commas, single quotes, and other relaxed JSON syntax.
 * @param raw raw metadata string without outer braces
 * @returns parsed object or undefined if parsing fails
 */
function parseMetadata(raw: string): Record<string, any> | undefined {
  try {
    // try parsing as standard JSON first
    return JSON.parse(`{${raw}}`)
  } catch {
    try {
      // fallback: basic JSON5 parsing for common cases
      // handle unquoted keys, single quotes, trailing commas
      let normalized = raw
        .replace(/(\w+):/g, '"$1":') // quote unquoted keys
        .replace(/'/g, '"') // convert single quotes to double quotes
        .replace(/,\s*([}\]])/g, "$1") // remove trailing commas
      return JSON.parse(`{${normalized}}`)
    } catch {
      // parsing failed, return undefined
      return undefined
    }
  }
}

/**
 * exit metadata component.
 * captures the raw metadata text from buffer.
 * parses as JSON5 and stores both raw and parsed versions.
 */
function exitMetadata(this: CompileContext): undefined {
  const metadataContent = this.resume()
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink
  if (node.data?.wikilink) {
    // store raw metadata including braces: "{key:value}"
    node.data.wikilink.metadata = `{${metadataContent}}`
    // parse and store parsed version
    node.data.wikilink.metadataParsed = parseMetadata(metadataContent)
  }
  return undefined
}

/**
 * annotate regular link with hast properties.
 * converts `[[target]]`, `[[target|alias]]`, `[[target#anchor]]` to <a> elements.
 */
function annotateRegularLink(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { alias, target, anchor, anchorText, metadataParsed, metadata } = wikilink

  const fp = target.trim()
  let displayText = alias ?? fp

  // if no explicit alias and there's an anchor, use the original anchor text as display
  if (!alias && anchor && anchorText) {
    displayText = anchorText
  }

  if (!node.data) node.data = { wikilink }

  node.data.hName = "a"
  node.data.hProperties = {
    href: url,
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }
  node.data.hChildren = [{ type: "text", value: displayText }]
}

/**
 * annotate image embed with hast properties.
 * converts `![[image.png]]`, `![[image.png|alt]]`, `![[image.png|100x200]]` to <img> elements.
 * supports figure/figcaption: `![[image.png|caption text|300x200]]`.
 */
function annotateImageEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { alias, metadataParsed, metadata } = wikilink

  const parts = (alias ?? "").split("|").map((s) => s.trim())
  let caption = ""
  let width = "auto"
  let height = "auto"

  if (parts.length > 0) {
    const lastPart = parts[parts.length - 1]
    const dimMatch = /^(\d+)(?:x(\d+))?$/.exec(lastPart)

    if (dimMatch) {
      width = dimMatch[1]
      height = dimMatch[2] ?? "auto"
      caption = parts.slice(0, -1).join("|").trim()
    } else {
      caption = parts.join("|").trim()
    }
  }

  if (!node.data) node.data = { wikilink }

  // backward compatible
  node.data.hName = "img"
  node.data.hProperties = {
    src: url,
    width,
    height,
    alt: caption,
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }
}

/**
 * annotate video embed with hast properties.
 * converts `![[video.mp4]]` to <video> elements with controls.
 */
function annotateVideoEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { metadataParsed, metadata } = wikilink

  if (!node.data) node.data = { wikilink }

  node.data.hName = "video"
  node.data.hProperties = {
    src: url,
    controls: true,
    loop: true,
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }
}

function annotateAudioEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { metadataParsed, metadata } = wikilink

  if (!node.data) node.data = { wikilink }

  node.data.hName = "audio"
  node.data.hProperties = {
    src: url,
    controls: true,
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }
}

/**
 * annotate PDF embed with hast properties.
 * converts `![[document.pdf]]` to <iframe> elements.
 */
function annotatePdfEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { metadataParsed, metadata } = wikilink

  if (!node.data) node.data = { wikilink }

  node.data.hName = "iframe"
  node.data.hProperties = {
    src: url,
    class: "pdf",
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }
}

/**
 * annotate block transclude with hast properties.
 * converts `![[page#heading]]`, `![[page#^block]]` to <blockquote> with nested <a>.
 */
function annotateTransclude(
  node: Wikilink,
  wikilink: WikilinkData,
  url: string,
  displayAnchor: string,
): void {
  const { alias, metadataParsed, metadata } = wikilink
  const block = displayAnchor

  if (!node.data) node.data = { wikilink }

  node.data.hName = "blockquote"
  node.data.hProperties = {
    class: "transclude",
    transclude: true,
    "data-url": url,
    "data-block": block,
    "data-embed-alias": alias ?? "",
    ...(metadataParsed
      ? { "data-metadata": JSON.stringify(metadataParsed) }
      : metadata
        ? { "data-metadata": metadata }
        : {}),
  }

  node.data.hChildren = [
    {
      type: "element",
      tagName: "a",
      properties: {
        href: url + displayAnchor,
        class: "transclude-inner",
      },
      children: [
        {
          type: "text",
          value: `Transclude of ${url} ${block}`,
        },
      ],
    },
  ]
}

/**
 * exit wikilink container.
 * finalizes the node, annotates with hName/hProperties/hChildren (if obsidian mode), and pops from stack.
 */
function exitWikilink(
  this: CompileContext,
  token: Token,
  options: FromMarkdownOptions = {},
): undefined {
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink

  if (node) {
    node.value = this.sliceSerialize(token)

    if (node.data?.wikilink) {
      node.data.wikilink.raw = node.value

      const wikilink = node.data.wikilink
      const { obsidian = true, stripExtensions = [], hasSlug } = options

      // handle implicit alias from space in target (Obsidian behavior)
      // only extract from target, not from anchors
      // if no explicit alias, has a target (not just anchor), and target contains a space after last /, split it
      if (!wikilink.alias && wikilink.target && !wikilink.anchor && wikilink.target.includes(" ")) {
        const lastSlash = wikilink.target.lastIndexOf("/")
        const afterSlash =
          lastSlash >= 0 ? wikilink.target.substring(lastSlash + 1) : wikilink.target
        const spaceIndex = afterSlash.indexOf(" ")

        if (spaceIndex >= 0) {
          // split: everything before space is path, everything after is display text
          const pathPart =
            lastSlash >= 0
              ? wikilink.target.substring(0, lastSlash + 1 + spaceIndex)
              : wikilink.target.substring(0, spaceIndex)
          const displayPart = afterSlash.substring(spaceIndex + 1).trim()

          // only apply implicit alias if we can verify:
          // - partial path (before space) exists as a file
          // - full path (with space) doesn't exist
          // this prevents breaking links like [[thoughts/Lebesgue measure]]
          if (hasSlug) {
            const fullPathSlug = slugifyFilePath(wikilink.target, stripExtensions)
            const partialPathSlug = slugifyFilePath(pathPart, stripExtensions)

            const fullPathExists = hasSlug(fullPathSlug)
            const partialPathExists = hasSlug(partialPathSlug)

            // only split if partial exists and full doesn't
            if (partialPathExists && !fullPathExists) {
              wikilink.target = pathPart
              wikilink.alias = displayPart
            }
          } else {
            // fallback: if no hasSlug function provided, apply the split
            // (maintains backward compatibility)
            wikilink.target = pathPart
            wikilink.alias = displayPart
          }
        }
      }

      // only annotate nodes when obsidian mode is enabled
      if (obsidian) {
        const targetPath = wikilink.target.trim()
        const ext = getFileExtension(targetPath) ?? ""
        const displayAnchor = wikilink.anchor?.trim() ?? ""

        let url: string
        // handle absolute paths like /tags/ml
        if (targetPath.startsWith("/")) {
          url = targetPath
        } else if (!targetPath) {
          // handle same-file anchors like #heading
          url = ""
        } else {
          url = slugifyFilePath(targetPath, stripExtensions)
        }
        if (wikilink.embed) {
          // check for youtube URLs first (before file extension checks)
          const youtubeEmbed = buildYouTubeEmbed(targetPath)
          if (youtubeEmbed) {
            // create img tag with youtube URL as src
            // downstream ofm.ts handler will convert to iframe
            if (!node.data) node.data = { wikilink }
            node.data.hName = "img"
            node.data.hProperties = {
              src: targetPath,
              ...(wikilink.metadataParsed
                ? { "data-metadata": JSON.stringify(wikilink.metadataParsed) }
                : wikilink.metadata
                  ? { "data-metadata": wikilink.metadata }
                  : {}),
            }
          } else if (ext && isImageExtension(ext)) {
            annotateImageEmbed(node, wikilink, url)
          } else if (ext && isVideoExtension(ext)) {
            annotateVideoEmbed(node, wikilink, url)
          } else if (ext && isAudioExtension(ext)) {
            annotateAudioEmbed(node, wikilink, url)
          } else if (ext === ".pdf") {
            annotatePdfEmbed(node, wikilink, url)
          } else {
            annotateTransclude(node, wikilink, url, displayAnchor)
          }
        } else {
          annotateRegularLink(node, wikilink, url + displayAnchor)
        }
      }
      // when obsidian: false, no annotations - raw wikilink nodes only
    }
  }

  this.exit(token)
  return undefined
}

/**
 * type guard for Wikilink.
 */
export function isWikilink(node: any): node is Wikilink {
  return node && node.type === "wikilink" && node.data?.wikilink
}

declare module "mdast" {
  interface StaticPhrasingContentMap {
    wikilink: Wikilink
  }
}
