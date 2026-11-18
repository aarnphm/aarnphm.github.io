import type { Options, Handle } from 'mdast-util-to-markdown'
import { serializeReference } from '../../util/references/syntax'
import { isWikilink } from './node'

export function wikilinkToMarkdown(): Options {
  return {
    handlers: { wikilink: handleWikilink },
    unsafe: [
      { character: '[', inConstruct: ['phrasing', 'label', 'reference'] },
      { character: ']', inConstruct: ['phrasing', 'label', 'reference'] },
    ],
  }
}

const handleWikilink: Handle = node => {
  if (isWikilink(node) && node.data?.reference) return serializeReference(node.data.reference)
  return 'value' in node && typeof node.value === 'string' ? node.value : '[[]]'
}
