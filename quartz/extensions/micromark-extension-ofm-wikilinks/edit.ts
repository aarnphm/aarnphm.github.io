import { fromMarkdown } from 'mdast-util-from-markdown'
import type { ReferenceDiagnostic, ReferenceParseResult } from '../../util/references/model'
import { editReference, parseReference, type ReferenceEdit } from '../../util/references/syntax'
import { wikilinkFromMarkdown } from './fromMarkdown'
import { isWikilink, type Wikilink } from './node'
import { wikilink } from './syntax'

export function editWikilink(
  node: Wikilink,
  patch: ReferenceEdit,
):
  | { kind: 'complete'; node: Wikilink }
  | { kind: 'invalid'; diagnostics: readonly ReferenceDiagnostic[] } {
  const current = node.data?.reference
  const parsed: ReferenceParseResult = current
    ? { kind: 'complete', reference: current }
    : parseReference(node.value, { context: 'markdown' })
  if (parsed.kind !== 'complete') {
    return {
      kind: 'invalid',
      diagnostics: [
        { code: 'invalid-edit', message: 'The node has no complete authored reference to edit.' },
      ],
    }
  }
  const edited = editReference(parsed.reference, patch)
  if (edited.kind === 'complete') {
    const tree = fromMarkdown(edited.reference.raw, {
      extensions: [wikilink()],
      mdastExtensions: [wikilinkFromMarkdown()],
    })
    const paragraph = tree.children[0]
    const replacement = paragraph?.type === 'paragraph' ? paragraph.children[0] : undefined
    if (!isWikilink(replacement) || !replacement.data)
      throw new Error('Edited reference failed Markdown conversion')
    replacement.data.reference = edited.reference
    return { kind: 'complete', node: replacement }
  }
  return edited
}
