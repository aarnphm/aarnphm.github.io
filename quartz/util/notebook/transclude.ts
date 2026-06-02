import { Element, ElementContent, Root, Text } from 'hast'
import { h } from 'hastscript'
import { EXIT, visit } from 'unist-util-visit'
import type { NotebookRuntimeCell, NotebookRuntimeData } from '../../runtime/notebook/types'
import { FullSlug } from '../path'
import { isRecord } from '../type-guards'
import { notebookRuntimeJson } from './cell-html'
import { notebookId } from './identity'

function propertyString(node: Element, key: string): string | undefined {
  const value = node.properties?.[key]
  return typeof value === 'string' ? value : undefined
}

function propertyPresent(node: Element, key: string): boolean {
  return node.properties !== undefined && key in node.properties
}

export function notebookCellRef(blockRef: string | undefined): string | undefined {
  if (!blockRef?.startsWith('#')) return undefined
  const ref = blockRef.slice(1)
  return ref.startsWith('^') ? ref.slice(1) : ref
}

function notebookCellFrameId(node: Element): string | undefined {
  return (
    propertyString(node, 'dataNotebookCellFrame') ??
    propertyString(node, 'data-notebook-cell-frame')
  )
}

export function findNotebookCellFrame(root: Root, cellId: string): Element | undefined {
  let cell: Element | undefined
  visit(root, 'element', node => {
    if (notebookCellFrameId(node) !== cellId) return
    cell = node
    return EXIT
  })
  return cell
}

export function resolveNotebookCell(
  root: Root,
  ref: string,
): { id: string; frame: Element } | undefined {
  const direct = findNotebookCellFrame(root, ref)
  if (direct) return { id: ref, frame: direct }
  const codeRef = `code-${ref}`
  const fallback = findNotebookCellFrame(root, codeRef)
  if (fallback) return { id: codeRef, frame: fallback }
  return undefined
}

function notebookRuntimeScriptText(root: Root): string | undefined {
  let text: string | undefined
  visit(root, { tagName: 'script' }, node => {
    if (
      propertyString(node, 'type') !== 'application/json' ||
      (!propertyPresent(node, 'dataNotebookRuntimeData') &&
        !propertyPresent(node, 'data-notebook-runtime-data'))
    ) {
      return
    }
    text = (node.children ?? [])
      .filter((child): child is Text => child.type === 'text')
      .map(child => child.value)
      .join('')
    return EXIT
  })
  return text
}

function readNotebookRuntimeCell(value: unknown): NotebookRuntimeCell | undefined {
  if (!isRecord(value)) return undefined
  const { id, source, language, displayLanguage, executionIndex } = value
  if (typeof id !== 'string' || typeof source !== 'string' || typeof language !== 'string') {
    return undefined
  }
  if (displayLanguage !== undefined && typeof displayLanguage !== 'string') return undefined
  if (typeof executionIndex !== 'number' && executionIndex !== null) return undefined
  return displayLanguage === undefined
    ? { id, source, language, executionIndex }
    : { id, source, language, displayLanguage, executionIndex }
}

export function notebookRuntimeData(root: Root): NotebookRuntimeData | undefined {
  const text = notebookRuntimeScriptText(root)
  if (!text) return undefined
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return undefined
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.cells)) return undefined
  const { id, sourcePath, language, indexUrl } = parsed
  if (
    typeof id !== 'string' ||
    typeof sourcePath !== 'string' ||
    typeof language !== 'string' ||
    (indexUrl !== undefined && typeof indexUrl !== 'string')
  ) {
    return undefined
  }
  const cells = parsed.cells.map(readNotebookRuntimeCell)
  if (cells.some(cell => cell === undefined)) return undefined
  const data: NotebookRuntimeData = {
    id,
    sourcePath,
    language,
    cells: cells.filter((cell): cell is NotebookRuntimeCell => cell !== undefined),
  }
  if (typeof indexUrl === 'string') data.indexUrl = indexUrl
  if (typeof parsed.toolbar === 'boolean') data.toolbar = parsed.toolbar
  if (typeof parsed.debug === 'boolean') data.debug = parsed.debug
  if (typeof parsed.vimMode === 'boolean') data.vimMode = parsed.vimMode
  if (Array.isArray(parsed.importableModules)) {
    const modules = parsed.importableModules.filter(
      (item): item is string => typeof item === 'string',
    )
    if (modules.length !== parsed.importableModules.length) return undefined
    data.importableModules = modules
  }
  return data
}

export function notebookCellRuntimeNodes(
  pageTree: Root,
  transcludeSource: { slug: FullSlug; transcludeTarget: FullSlug; cellId: string; count: number },
): ElementContent[] {
  const data = notebookRuntimeData(pageTree)
  const cell = data?.cells.find(cell => cell.id === transcludeSource.cellId)
  if (!data || !cell) return []
  const runtimeData: NotebookRuntimeData = {
    ...data,
    id: notebookId(
      transcludeSource.count === 0
        ? `notebook-cell-transclude:${transcludeSource.slug}:${transcludeSource.transcludeTarget}:${transcludeSource.cellId}`
        : `notebook-cell-transclude:${transcludeSource.slug}:${transcludeSource.transcludeTarget}:${transcludeSource.cellId}:${transcludeSource.count}`,
    ),
    toolbar: false,
    cells: [cell],
  }
  return [
    h('.notebook-runtime', { dataNotebookRuntime: runtimeData.id }),
    h(
      'script',
      { type: 'application/json', dataNotebookRuntimeData: '' },
      { type: 'text', value: notebookRuntimeJson(runtimeData) },
    ),
  ]
}
