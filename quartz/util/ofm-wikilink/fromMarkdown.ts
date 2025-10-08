/**
 * mdast-util-from-markdown extension for wikilinks.
 * converts micromark tokens to Wikilink AST nodes.
 */

import type { Extension, CompileContext, Token } from "mdast-util-from-markdown"
import type { Element as HastElement, Text as HastText } from "hast"
import { FilePath, FullSlug, getFileExtension, slugAnchor, stripSlashes, sluggify } from "../path"
import "./types"
import { Literal } from "unist"

export interface WikilinkParsed {
  raw: string
  target: string
  anchor?: string
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
    wikilink: WikilinkParsed
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
function sluggifyFilePath(path: string, stripExts: string[]): string {
  let fp = stripSlashes(path) as FilePath
  let ext = getFileExtension(fp)
  const withoutFileExt = fp.replace(new RegExp(ext + "$"), "")
  if ([".md", ".base", ...stripExts, undefined].includes(ext)) {
    ext = ""
  }
  const slug = sluggify(withoutFileExt).toLowerCase()
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

    // obsidian-style anchor resolution
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
 * annotate regular link with hast properties.
 * converts `[[target]]`, `[[target|alias]]`, `[[target#anchor]]` to <a> elements.
 */
function annotateRegularLink(node: Wikilink, wikilink: WikilinkParsed, url: string): void {
  const { alias, target } = wikilink

  const fp = target.trim()
  const displayText = alias ?? fp

  if (!node.data) node.data = { wikilink }

  node.data.hName = "a"
  node.data.hProperties = { href: url }
  node.data.hChildren = [{ type: "text", value: displayText }]
}

/**
 * annotate image embed with hast properties.
 * converts `![[image.png]]`, `![[image.png|alt]]`, `![[image.png|100x200]]` to <img> elements.
 * supports figure/figcaption: `![[image.png|caption text|300x200]]`.
 */
function annotateImageEmbed(node: Wikilink, wikilink: WikilinkParsed, url: string): void {
  const { alias } = wikilink

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
  }
}

/**
 * annotate video embed with hast properties.
 * converts `![[video.mp4]]` to <video> elements with controls.
 */
function annotateVideoEmbed(node: Wikilink, wikilink: WikilinkParsed, url: string): void {
  if (!node.data) node.data = { wikilink }

  node.data.hName = "video"
  node.data.hProperties = {
    src: url,
    controls: true,
    loop: true,
  }
}

/**
 * annotate audio embed with hast properties.
 * converts `![[audio.mp3]]` to <audio> elements with controls.
 */
function annotateAudioEmbed(node: Wikilink, wikilink: WikilinkParsed, url: string): void {
  if (!node.data) node.data = { wikilink }

  node.data.hName = "audio"
  node.data.hProperties = {
    src: url,
    controls: true,
  }
}

/**
 * annotate PDF embed with hast properties.
 * converts `![[document.pdf]]` to <iframe> elements.
 */
function annotatePdfEmbed(node: Wikilink, wikilink: WikilinkParsed, url: string): void {
  if (!node.data) node.data = { wikilink }

  node.data.hName = "iframe"
  node.data.hProperties = {
    src: url,
    class: "pdf",
  }
}

/**
 * annotate block transclude with hast properties.
 * converts `![[page#heading]]`, `![[page#^block]]` to <blockquote> with nested <a>.
 */
function annotateTransclude(
  node: Wikilink,
  wikilink: WikilinkParsed,
  url: string,
  displayAnchor: string,
): void {
  const { alias } = wikilink
  const block = displayAnchor

  if (!node.data) node.data = { wikilink }

  node.data.hName = "blockquote"
  node.data.hProperties = {
    class: "transclude",
    transclude: true,
    "data-url": url,
    "data-block": block,
    "data-embed-alias": alias ?? "",
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
      const { obsidian = true, stripExtensions = [] } = options

      // only annotate nodes when obsidian mode is enabled
      if (obsidian) {
        const targetPath = wikilink.target.trim()
        const url = sluggifyFilePath(targetPath, stripExtensions)
        const ext = getFileExtension(wikilink.target) ?? ""
        const displayAnchor = wikilink.anchor?.trim() ?? ""

        if (wikilink.embed) {
          if (ext && isImageExtension(ext)) {
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
  interface PhrasingContentMap {
    wikilink: Wikilink
  }
}
