import { escapeHTML } from '../../../util/escape'
import { getSharedLeanWebSocketSession } from '../../lean/kernel'
import { getLeanAbbreviationIndex } from '../../lean/unicode-input'
import { isRemoteBrowser } from './editor-context'
import { extractLeanIdentAtColumn } from './lexical'
import { getDocumentSymbolsForSource, LEAN_CATALOG_BY_LABEL } from './symbol-index'

export type LeanHoverSummary = {
  readonly startCol: number
  readonly endCol: number
  readonly unicodeHint?: string
  readonly signature?: string
  readonly documentation?: string
  readonly goals?: readonly string[]
}

const offlineLineHoverCache = new Map<string, LeanHoverSummary | null>()
const MAX_OFFLINE_HOVER_CACHE = 128

export async function resolveLeanHoverAt(
  docText: string,
  lineNumber0: number,
  lineText: string,
  col: number,
): Promise<LeanHoverSummary | undefined> {
  const session = getSharedLeanWebSocketSession()
  const warm = session.isWarm()
  const lineKey = !warm ? `${lineText}\0${col}` : ''
  if (!warm) {
    const cached = offlineLineHoverCache.get(lineKey)
    if (cached !== undefined) {
      if (isRemoteBrowser()) void session.warm()
      return cached ?? undefined
    }
  }

  const symbolHover = getLeanAbbreviationIndex().hoverInfoForTextAt(lineText, col)
  const ident = extractLeanIdentAtColumn(lineText, col)
  if (!symbolHover && !ident) {
    if (!warm) {
      if (offlineLineHoverCache.size >= MAX_OFFLINE_HOVER_CACHE) {
        const oldest = offlineLineHoverCache.keys().next().value
        if (oldest !== undefined) offlineLineHoverCache.delete(oldest)
      }
      offlineLineHoverCache.set(lineKey, null)
    }
    return undefined
  }

  const unicodeHint = symbolHover ? symbolHover.markdown.replaceAll('`', '') : undefined
  const staticCatalog = ident ? LEAN_CATALOG_BY_LABEL.get(ident.token) : undefined
  const localCatalog =
    staticCatalog ??
    (ident ? getDocumentSymbolsForSource(docText).byLabel.get(ident.token) : undefined)

  const signature = localCatalog ? `${localCatalog.label} : ${localCatalog.detail}` : undefined
  let documentation = localCatalog?.documentation
  let goals: readonly string[] | undefined

  if (warm) {
    const liveHover = await session.hover(docText, { line: lineNumber0, character: col })
    if (liveHover?.contents) {
      documentation = liveHover.contents
    }
    if (liveHover?.goals && liveHover.goals.length > 0) {
      goals = liveHover.goals
    }
  } else if (isRemoteBrowser()) {
    void session.warm()
  }

  if (!unicodeHint && !signature && !documentation && (!goals || goals.length === 0)) {
    return undefined
  }

  const startCol = ident
    ? ident.startCol
    : symbolHover
      ? symbolHover.fromCol
      : Math.min(col, lineText.length)
  const endCol = ident
    ? ident.endCol
    : symbolHover
      ? symbolHover.toCol
      : Math.min(col + 1, lineText.length)

  const summary: LeanHoverSummary = {
    startCol,
    endCol,
    ...(unicodeHint ? { unicodeHint } : {}),
    ...(signature ? { signature } : {}),
    ...(documentation ? { documentation } : {}),
    ...(goals ? { goals } : {}),
  }

  if (!warm && (staticCatalog !== undefined || !ident)) {
    if (offlineLineHoverCache.size >= MAX_OFFLINE_HOVER_CACHE) {
      const oldest = offlineLineHoverCache.keys().next().value
      if (oldest !== undefined) offlineLineHoverCache.delete(oldest)
    }
    offlineLineHoverCache.set(lineKey, summary)
  }

  return summary
}

export function renderLeanHoverHtml(summary: LeanHoverSummary): string {
  const parts: string[] = []
  if (summary.unicodeHint) {
    parts.push(
      `<div class="cm-lean-hover-unicode"><code>${escapeHTML(summary.unicodeHint)}</code></div>`,
    )
  }
  if (summary.signature) {
    parts.push(
      `<pre class="cm-lean-hover-signature"><code>${escapeHTML(summary.signature)}</code></pre>`,
    )
  }
  if (summary.documentation) {
    parts.push(`<div class="cm-lean-hover-doc">${escapeHTML(summary.documentation)}</div>`)
  }
  if (summary.goals && summary.goals.length > 0) {
    parts.push(
      `<pre class="cm-lean-hover-goals"><code>${escapeHTML(summary.goals.join('\n\n'))}</code></pre>`,
    )
  }
  return parts.join('')
}
