/**
 * mdast-util-from-markdown extension for wikilinks.
 * converts micromark tokens to WikilinkNode AST nodes.
 */

import type { Extension, CompileContext, Token } from "mdast-util-from-markdown"
import { slugAnchor } from "../path"
import "./types"

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
 */
export interface Wikilink {
  type: "wikilink"
  value: string
  data?: {
    wikilink: WikilinkParsed
    hName?: string
    hProperties?: Record<string, any>
  }
  position?: {
    start: { line: number; column: number; offset: number }
    end: { line: number; column: number; offset: number }
  }
}

export interface FromMarkdownOptions {
  /**
   * enable Obsidian-style anchor handling.
   * when true, anchors with multiple # segments (e.g., "Parent#Child#Grandchild")
   * will use only the last segment ("Grandchild").
   * default: false
   */
  obsidian?: boolean
}

/**
 * create mdast-util-from-markdown extension for wikilinks.
 */
export function wikilinkFromMarkdown(options: FromMarkdownOptions = {}): Extension {
  const { obsidian = false } = options

  return {
    enter: {
      wikilink: enterWikilink,
      wikilinkEmbedMarker: enterEmbedMarker,
      wikilinkTarget: enterTarget,
      wikilinkAnchor: enterAnchor,
      wikilinkAlias: enterAlias,
    },
    exit: {
      wikilink: exitWikilink,
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
 * creates the WikilinkNode and pushes it onto the stack.
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
 * exit wikilink container.
 * finalizes the node and pops it from the stack.
 */
function exitWikilink(this: CompileContext, token: Token): undefined {
  // get the node from the stack before exiting
  const node = this.stack[this.stack.length - 1] as unknown as Wikilink

  // serialize the original wikilink text from token positions
  if (node) {
    node.value = this.sliceSerialize(token)

    // store raw value in wikilink data
    if (node.data?.wikilink) {
      node.data.wikilink.raw = node.value
    }
  }

  this.exit(token)
  return undefined
}

/**
 * type guard for WikilinkNode.
 */
export function isWikilinkNode(node: any): node is Wikilink {
  return node && node.type === "wikilink" && node.data?.wikilink
}

declare module "mdast" {
  interface StaticPhrasingContentMap {
    wikilink: Wikilink
  }
}
