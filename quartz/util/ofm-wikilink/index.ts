/**
 * ofm-wikilink - micromark extension for Obsidian-flavored wikilinks.
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
import { wikilink } from "./syntax"
import { wikilinkToMarkdown } from "./toMarkdown"
import { wikilinkFromMarkdown, isWikilink } from "./fromMarkdown"

export { wikilink, wikilinkToMarkdown, wikilinkFromMarkdown, isWikilink }
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
   * enable Obsidian-style nested anchor handling and automatic hast conversion.
   * default: true
   *
   * when true: uses internal slugification and annotates nodes for automatic HTML conversion
   * when false: returns raw wikilink nodes without hName annotations
   */
  obsidian?: boolean

  /**
   * file extensions to strip before slugifying.
   * default: ['.md', '.base']
   * only applies when obsidian: true
   */
  stripExtensions?: string[]
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
