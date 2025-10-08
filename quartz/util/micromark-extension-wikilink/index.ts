/**
 * micromark extension for Obsidian wikilinks.
 *
 * provides:
 * - micromark syntax extension (tokenizer)
 * - mdast-util-from-markdown extension (token → AST conversion)
 * - mdast-util-to-markdown extension (AST → markdown serialization)
 *
 * usage with unified:
 * ```typescript
 * import { unified } from 'unified'
 * import remarkParse from 'remark-parse'
 * import remarkStringify from 'remark-stringify'
 * import { wikilink, wikilinkFromMarkdown, wikilinkToMarkdown } from './micromark-extension-wikilink'
 *
 * const processor = unified()
 *   .use(remarkParse)
 *   .use(remarkStringify)
 *   .data('micromarkExtensions', [wikilink()])
 *   .data('fromMarkdownExtensions', [wikilinkFromMarkdown()])
 *   .data('toMarkdownExtensions', [wikilinkToMarkdown()])
 * ```
 */
import { wikilink } from "./syntax"
import { wikilinkToMarkdown } from "./toMarkdown"
import { wikilinkFromMarkdown, isWikilinkNode } from "./fromMarkdown"

export { wikilink, wikilinkToMarkdown, wikilinkFromMarkdown, isWikilinkNode }
export type { Wikilink, FromMarkdownOptions } from "./fromMarkdown"
export type {
  WikilinkToken,
  WikilinkEmbedMarker,
  WikilinkOpenMarker,
  WikilinkCloseMarker,
  WikilinkTarget,
  WikilinkAnchorMarker,
  WikilinkAnchor,
  WikilinkAliasMarker,
  WikilinkAlias,
  WikilinkChunk,
  WikilinkTokenType,
} from "./types"

export interface RemarkWikilinkOptions {
  /**
   * enable Obsidian-style anchor handling.
   * when true, anchors with multiple # segments (e.g., "Parent#Child#Grandchild")
   * will use only the last segment ("Grandchild").
   * default: false
   */
  obsidian?: boolean
}

export function remarkWikilink(options: RemarkWikilinkOptions = {}) {
  // @ts-expect-error: TS is wrong about `this`.
  // eslint-disable-next-line unicorn/no-this-assignment
  const self = /** @type {Processor<Root>} */ this
  const data = self.data()
  const micromarkExtensions = data.micromarkExtensions || (data.micromarkExtensions = [])
  const fromMarkdownExtensions = data.fromMarkdownExtensions || (data.fromMarkdownExtensions = [])
  const toMarkdownExtensions = data.toMarkdownExtensions || (data.toMarkdownExtensions = [])

  micromarkExtensions.push(wikilink())
  fromMarkdownExtensions.push(wikilinkFromMarkdown(options))
  toMarkdownExtensions.push(wikilinkToMarkdown())
}
