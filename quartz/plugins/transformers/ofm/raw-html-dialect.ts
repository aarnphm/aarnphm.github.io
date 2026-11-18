import type { Nodes } from 'mdast'
import type { Data } from 'vfile'
import { toHtml } from 'hast-util-to-html'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toHast } from 'mdast-util-to-hast'
import type { Options } from '../ofm'
import { dialectFromMarkdown } from './dialect'
import { dialectSyntax } from './dialect-syntax'
import { lowerTags } from './tags'

/** Render opt-in inline syntax while retaining the surrounding raw HTML source. */
export function rawHtmlDialect(
  source: string,
  options: Pick<Options, 'highlight' | 'parseTags'>,
  data: Data,
): string {
  if (
    !((options.highlight && source.includes('==')) || (options.parseTags && source.includes('#')))
  )
    return source
  const tree = fromMarkdown(source, {
    extensions: [
      { disable: { null: ['htmlFlow'] } },
      dialectSyntax({
        comments: false,
        enableInlineFootnotes: false,
        highlight: options.highlight,
        parseTags: options.parseTags,
      }),
    ],
    mdastExtensions: [dialectFromMarkdown()],
  })
  const protectedRanges: [number, number][] = []
  const openTags: { tag: string; start: number }[] = []
  function protect(node: Nodes) {
    if (node.type === 'html') {
      const tag = /^<(\/?)(code|pre|script|style|textarea)(?=[\s/>])/i.exec(node.value)
      const start = node.position?.start.offset
      const end = node.position?.end.offset
      if (tag && start !== undefined && end !== undefined) {
        const name = tag[2].toLowerCase()
        if (tag[1]) {
          const index = openTags.findLastIndex(open => open.tag === name)
          if (index >= 0) {
            protectedRanges.push([openTags[index].start, end])
            openTags.splice(index)
          }
        } else if (!node.value.endsWith('/>')) openTags.push({ tag: name, start })
      }
    }
    if ('children' in node) node.children.forEach(protect)
  }
  protect(tree)
  for (const open of openTags) protectedRanges.push([open.start, source.length])
  const isProtected = (node: Nodes) => {
    const start = node.position?.start.offset
    const end = node.position?.end.offset
    return (
      start === undefined ||
      end === undefined ||
      protectedRanges.some(([a, b]) => start < b && end > a)
    )
  }
  if (options.parseTags) lowerTags(tree, data, isProtected)
  const replacements: { start: number; end: number; html: string }[] = []
  function collect(node: Nodes) {
    if (node.type === 'ofmHighlight' || (node.type === 'link' && node.data?.tagIdentity)) {
      const start = node.position?.start.offset
      const end = node.position?.end.offset
      if (start === undefined || end === undefined || isProtected(node)) return
      const html = toHast(node, { allowDangerousHtml: true })
      if (html) replacements.push({ start, end, html: toHtml(html, { allowDangerousHtml: true }) })
      return
    }
    if ('children' in node) node.children.forEach(collect)
  }
  collect(tree)
  let result = source
  for (const replacement of replacements.sort((a, b) => b.start - a.start))
    result = result.slice(0, replacement.start) + replacement.html + result.slice(replacement.end)
  return result
}
