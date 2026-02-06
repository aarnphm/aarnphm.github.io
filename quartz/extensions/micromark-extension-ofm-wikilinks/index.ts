import type { Root } from 'mdast'
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
import type { Processor } from 'unified'
import { wikilinkFromMarkdown, isWikilink } from './fromMarkdown'
import { wikilink } from './syntax'
import { wikilinkToMarkdown } from './toMarkdown'

export { wikilink, wikilinkToMarkdown, wikilinkFromMarkdown, isWikilink }
export type { Wikilink, WikilinkData, FromMarkdownOptions } from './fromMarkdown'

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

  /**
   * function to check if a slug exists in the content index.
   * used to validate implicit alias splitting: only split "[[file text]]"
   * if "file" exists but "file text" doesn't.
   */
  hasSlug?: (slug: string) => boolean
}

export function remarkWikilink(this: Processor<Root>, options: RemarkWikilinkOptions = {}): void {
  const data = this.data()

  data.micromarkExtensions ??= []
  data.fromMarkdownExtensions ??= []
  //@ts-ignore
  data.toMarkdownExtensions ??= []

  data.micromarkExtensions.push(wikilink())
  data.fromMarkdownExtensions.push(wikilinkFromMarkdown(options))
  //@ts-ignore
  data.toMarkdownExtensions.push(wikilinkToMarkdown())
}
