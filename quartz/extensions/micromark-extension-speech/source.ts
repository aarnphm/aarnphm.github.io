import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { math } from 'micromark-extension-math'
import { visit } from 'unist-util-visit'
import { speech, speechFromMarkdown } from './index'

/** Speech annotations affect playback, so their delimiters are excluded from scheduling identity. */
export function stripSpeechMarkup(source: string): string {
  if (!source.includes('{{')) return source
  const tree = fromMarkdown(source, {
    extensions: [speech(), math()],
    mdastExtensions: [speechFromMarkdown(), mathFromMarkdown()],
  })
  const delimiters: { start: number; end: number }[] = []
  visit(tree, 'speechPhrase', node => {
    const start = node.position?.start.offset
    const end = node.position?.end.offset
    if (start === undefined || end === undefined) return
    delimiters.push({ start, end: start + 2 }, { start: end - 2, end })
  })
  let result = ''
  let offset = 0
  for (const delimiter of delimiters.sort((a, b) => a.start - b.start)) {
    result += source.slice(offset, delimiter.start)
    offset = delimiter.end
  }
  return result + source.slice(offset)
}
