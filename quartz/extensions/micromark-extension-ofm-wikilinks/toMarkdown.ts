/**
 * mdast-util-to-markdown extension for wikilinks.
 * serializes Wikilink back to Obsidian wikilink syntax.
 *
 * usage with unified:
 * ```typescript
 * import { unified } from 'unified'
 * import remarkParse from 'remark-parse'
 * import remarkStringify from 'remark-stringify'
 * import { wikilink, wikilinkFromMarkdown, wikilinkToMarkdown } from 'ofm-wikilink'
 *
 * const processor = unified()
 *   .use(remarkParse)
 *   .use(remarkStringify)
 *   .data('micromarkExtensions', [wikilink()])
 *   .data('fromMarkdownExtensions', [wikilinkFromMarkdown()])
 *   .data('toMarkdownExtensions', [wikilinkToMarkdown()])
 * ```
 */

import type { Options, Handle } from "mdast-util-to-markdown"
import type { Wikilink } from "./fromMarkdown"

/**
 * escape special characters in wikilink component.
 * prevents early termination of component during parsing.
 *
 * @param text - text to escape
 * @param chars - characters to escape (e.g., '\\|#]')
 * @returns escaped text with backslashes before special chars
 */
function escape(text: string, chars: string): string {
  let result = ""
  for (const char of text) {
    if (chars.includes(char)) {
      result += "\\" + char
    } else {
      result += char
    }
  }
  return result
}

/**
 * create mdast-util-to-markdown extension for wikilinks.
 */
export function wikilinkToMarkdown(): Options {
  const handlers: any = { wikilink: handleWikilink }

  return {
    handlers,
    unsafe: [
      { character: "[", inConstruct: ["phrasing", "label", "reference"] as any },
      { character: "]", inConstruct: ["phrasing", "label", "reference"] as any },
    ],
  }
}

/**
 * handler for Wikilink serialization.
 * converts Wikilink back to Obsidian wikilink syntax.
 *
 * examples:
 * - { target: "page" } → `[[page]]`
 * - { target: "page", alias: "text" } → `[[page|text]]`
 * - { target: "page", anchor: "#heading" } → `[[page#heading]]`
 * - { target: "page", anchor: "#heading", alias: "text" } → `[[page#heading|text]]`
 * - { target: "page", metadata: "{key:value}" } → `[[page#{key:value}]]`
 * - { target: "page", anchor: "#heading", metadata: "{key:value}" } → `[[page#heading#{key:value}]]`
 * - { target: "img.png", embed: true } → `![[img.png]]`
 * - { target: "file", anchor: "#^block" } → `[[file#^block]]`
 *
 * escaping:
 * - target: escapes `\`, `|`, `#`, `]`
 * - anchor: escapes `\`, `|`, `]`
 * - metadata: escapes `\`, `|`, `]`
 * - alias: escapes `\`, `]`
 */
const handleWikilink: Handle = (node: Wikilink): string => {
  const wikilink = node.data?.wikilink
  if (!wikilink) {
    // fallback: use node value if available
    return node.value ?? "[[]]"
  }

  // prefer raw original text for exact round-trip serialization
  // this preserves the original anchor text before slugification
  if (wikilink.raw) {
    return wikilink.raw
  }

  // reconstruct from components (may not match original exactly)
  const { target = "", anchor, metadata, alias, embed = false } = wikilink

  let result = ""

  // embed prefix
  if (embed) {
    result += "!"
  }

  // opening brackets
  result += "[["

  // target component
  // escape: \ | # ]
  result += escape(target, "\\|#]")

  // anchor component (includes leading # or #^)
  // escape: \ | ]
  if (anchor) {
    // anchor already has # prefix from fromMarkdown.ts
    result += escape(anchor, "\\|]")
  }

  // metadata component
  // metadata already includes braces from exitMetadata
  // escape: \ | ]
  if (metadata) {
    result += "#" + escape(metadata, "\\|]")
  }

  // alias component
  // escape: \ ]
  if (alias) {
    result += "|" + escape(alias, "\\]")
  }

  // closing brackets
  result += "]]"

  return result
}
