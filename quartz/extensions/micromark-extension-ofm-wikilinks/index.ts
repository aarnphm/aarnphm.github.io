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
export { editWikilink } from './edit'

export { wikilink, wikilinkToMarkdown, wikilinkFromMarkdown, isWikilink }
export type { Wikilink } from './fromMarkdown'

export function remarkWikilink(this: Processor): void {
  const data = this.data()

  data.micromarkExtensions ??= []
  data.fromMarkdownExtensions ??= []
  //@ts-ignore
  data.toMarkdownExtensions ??= []

  data.micromarkExtensions.push(wikilink())
  data.fromMarkdownExtensions.push(wikilinkFromMarkdown())
  //@ts-ignore
  data.toMarkdownExtensions.push(wikilinkToMarkdown())
}
