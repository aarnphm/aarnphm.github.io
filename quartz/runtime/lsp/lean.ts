import type { CompletionResult, CompletionSource } from '@codemirror/autocomplete'
import type { Extension } from '@codemirror/state'
import { hoverTooltip, type Tooltip } from '@codemirror/view'
import type { LspBridge, LspConfig } from './bridge'
import { getSharedLeanWebSocketSession } from '../lean/kernel'
import { getLeanAbbreviationIndex, leanUnicodeCompletionSource } from '../lean/unicode-input'
import {
  collectLocalIndexedForCompletion,
  matchIndexedCatalogCandidates,
  mergeRemoteCompletions,
} from './lean/completion'
import { isRemoteBrowser, leanLspConfigFacet } from './lean/editor-context'
import { renderLeanHoverHtml, resolveLeanHoverAt } from './lean/hover'
import { extractLeanIdentAtColumn, scanLeanPrefixBeforeColumn } from './lean/lexical'
export { extractLeanDocumentSymbols } from './lean/symbol-index'
export { extractLeanIdentAtColumn } from './lean/lexical'
export { leanLspConfigFacet } from './lean/editor-context'
export { renderLeanHoverHtml, resolveLeanHoverAt } from './lean/hover'
export type { LeanHoverSummary } from './lean/hover'
export type { LeanCatalogCategory, LeanCatalogEntry } from './lean/catalog-entry'

export const leanCompletionSource: CompletionSource = context => {
  const line = context.state.doc.lineAt(context.pos)
  const col = context.pos - line.from
  const token = scanLeanPrefixBeforeColumn(line.text, col)
  if (token?.kind === 'backslash') {
    return leanUnicodeCompletionSource(context)
  }

  if (!token && !context.explicit) return null

  const from = token ? line.from + token.fromCol : context.pos
  const prefix = token ? token.prefix : ''
  const { indexed, cacheMap } = collectLocalIndexedForCompletion(context, prefix)
  const localOptions = matchIndexedCatalogCandidates(prefix, indexed, cacheMap)

  const session = getSharedLeanWebSocketSession()
  if (!session.isWarm()) {
    if (isRemoteBrowser()) {
      void session.warm()
    }
    if (localOptions.length === 0 && !context.explicit) return null
    return { from, options: localOptions } satisfies CompletionResult
  }

  const docText = context.state.doc.toString()
  const position = { line: line.number - 1, character: col }

  return session.complete(docText, position).then(remoteItems => {
    const merged = mergeRemoteCompletions(prefix, localOptions, remoteItems)
    if (merged.length === 0 && !context.explicit) return null
    return { from, options: merged } satisfies CompletionResult
  })
}

export function leanHoverTooltipExtension(hoverTime = 220): Extension {
  return hoverTooltip(
    async (view, pos): Promise<Tooltip | null> => {
      const line = view.state.doc.lineAt(pos)
      const col = pos - line.from
      const lineText = line.text
      if (
        !getLeanAbbreviationIndex().hoverInfoForTextAt(lineText, col) &&
        !extractLeanIdentAtColumn(lineText, col)
      ) {
        return null
      }

      const docText = view.state.doc.toString()
      const summary = await resolveLeanHoverAt(docText, line.number - 1, lineText, col)
      if (!summary) return null

      return {
        pos: line.from + summary.startCol,
        end: line.from + summary.endCol,
        above: true,
        create() {
          const element = document.createElement('div')
          element.className = 'cm-lsp-hover-tooltip cm-lsp-documentation'
          element.innerHTML = renderLeanHoverHtml(summary)
          return { dom: element }
        },
      }
    },
    { hideOn: transaction => transaction.docChanged, hoverTime },
  )
}

export const leanLspBridge: LspBridge = {
  extensions(config: LspConfig): Promise<readonly Extension[]> {
    if (isRemoteBrowser()) {
      void getSharedLeanWebSocketSession().warm()
    }
    return Promise.resolve(config.enabled ? [leanLspConfigFacet.of(config)] : [])
  },
}
