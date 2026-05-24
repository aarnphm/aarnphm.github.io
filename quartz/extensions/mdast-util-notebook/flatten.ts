import type { BlockContent, Html, Root as MdRoot } from 'mdast'
import type { NotebookDoc, Output } from '../../util/notebook/types'
import type { NotebookCell, NotebookOutput } from './types'
import { renderOutputHtml } from '../../util/notebook/render/output-to-hast'
import { notebookToMdast } from './index'

function outputToHtmlNode(notebookOutput: NotebookOutput): Html {
  const data = notebookOutput.data.notebookOutput
  const typed = reconstructOutput(data)
  return { type: 'html', value: typed ? renderOutputHtml(typed) : '' }
}

function reconstructOutput(data: NotebookOutput['data']['notebookOutput']): Output | undefined {
  if (data.kind === 'stream' && data.name && data.text !== undefined) {
    return { kind: 'stream', name: data.name, text: data.text }
  }
  if (
    data.kind === 'error' &&
    data.ename !== undefined &&
    data.evalue !== undefined &&
    data.traceback !== undefined
  ) {
    return { kind: 'error', ename: data.ename, evalue: data.evalue, traceback: data.traceback }
  }
  if (data.kind === 'display_data' && data.data) {
    return { kind: 'display_data', data: data.data, metadata: data.metadata ?? {} }
  }
  if (data.kind === 'execute_result' && data.data) {
    return {
      kind: 'execute_result',
      data: data.data,
      metadata: data.metadata ?? {},
      executionCount: data.executionCount ?? null,
    }
  }
  return undefined
}

function flattenCell(cell: NotebookCell): BlockContent[] {
  const meta = cell.data.notebookCell
  const openTag: Html = {
    type: 'html',
    value: `<div class="notebook-flat-cell notebook-flat-cell-${meta.cellType}" data-notebook-cell-frame="${meta.cellId}" id="${meta.cellId}" data-notebook-language="${meta.language}">`,
  }
  const closeTag: Html = { type: 'html', value: '</div>' }
  const children = cell.children.flatMap(child => {
    if (child.type === 'notebookOutput') return [outputToHtmlNode(child as NotebookOutput)]
    return [child]
  })
  return [openTag, ...children, closeTag] as BlockContent[]
}

export function notebookDocToFlatMdast(doc: NotebookDoc): MdRoot {
  const tree = notebookToMdast(doc)
  const rawChildren = tree.children as unknown as (BlockContent | NotebookCell)[]
  const children = rawChildren.flatMap<BlockContent>(child => {
    if ((child as { type: string }).type === 'notebookCell')
      return flattenCell(child as NotebookCell)
    return [child as BlockContent]
  })
  return { type: 'root', children }
}
