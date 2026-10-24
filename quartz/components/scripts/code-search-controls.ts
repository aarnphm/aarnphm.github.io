import type { CodeMatch, CodeSearchProgress, CodeSearchResult } from '../../util/code-search-types'
import { isFullSlug, resolveRelative, type FullSlug } from '../../util/path'

export interface CodeSearchRow extends CodeMatch {
  selectedRange: number
}

/** Keep owners together while retaining every occurrence and range. */
export function codeSearchRows(matches: CodeMatch[]): CodeSearchRow[] {
  const owners = new Map<string, CodeSearchRow[]>()
  for (const match of matches) {
    const group = owners.get(match.unit.ownerSlug) ?? []
    group.push(
      ...(match.ranges.length
        ? match.ranges.map((_, index) => ({
            ...match,
            selectedRange: index,
            excerpt: index === 0 ? match.excerpt : 'select to preview this range',
          }))
        : [{ ...match, selectedRange: 0 }]),
    )
    owners.set(match.unit.ownerSlug, group)
  }
  return [...owners.values()].flat()
}

export function codeMatchLocation(match: CodeSearchRow): string {
  const range = match.ranges[match.selectedRange]
  const start = range?.start.line ?? 1
  const end = range?.end.line ?? start
  const frame =
    match.unit.kind === 'attachment' ? 'L' : match.unit.kind === 'cell' ? 'cell L' : 'block L'
  return `${frame}${start}${end === start ? '' : `–${end}`}`
}

export function codeMatchLabel(match: CodeSearchRow): string {
  const count = match.ranges.length
  return `${match.unit.language} · ${codeMatchLocation(match)}${count > 1 ? ` · ${count} matches` : ''}`
}

export function codeMatchUrl(current: FullSlug, match: CodeMatch): URL {
  const slug = match.unit.ownerSlug
  if (!isFullSlug(slug)) throw new Error('Invalid code source location')
  const url = new URL(resolveRelative(current, slug), location.href)
  if (match.unit.anchor) url.hash = match.unit.anchor
  return url
}

export function codeSearchStatus(value: CodeSearchProgress): HTMLParagraphElement {
  const status = document.createElement('p')
  status.className = 'code-search-status'
  status.setAttribute('role', 'status')
  status.textContent = value.phase === 'loading' ? 'loading code' : 'searching code'
  return status
}

const lastCodeSearchNotice = new WeakMap<HTMLElement, string>()

export function codeSearchSummary(
  result: CodeSearchResult,
  count: number,
  host: HTMLElement,
): HTMLParagraphElement {
  const unsupported = result.unsupportedLanguages
  const notice = unsupported.length
    ? `structural search skipped ${unsupported.length === 1 ? unsupported[0] : `${unsupported.length} languages`}. use a language filter or remove ast:.`
    : result.complete
      ? ''
      : 'results are incomplete. narrow the query and retry.'
  if (lastCodeSearchNotice.get(host) !== notice) {
    lastCodeSearchNotice.set(host, notice)
    if (notice) {
      const event: CustomEventMap['toast'] = new CustomEvent('toast', {
        detail: { message: notice, durationMs: 6000, containerHost: host },
      })
      document.dispatchEvent(event)
    }
  }
  const status = document.createElement('p')
  status.className = 'code-search-result-count'
  status.setAttribute('role', 'status')
  status.textContent = `${count} ${count === 1 ? 'result' : 'results'}`
  return status
}

export function codeSearchError(error: unknown): HTMLParagraphElement {
  const status = document.createElement('p')
  status.className = 'code-search-status'
  status.setAttribute('role', 'status')
  status.textContent = `code search error${error instanceof Error ? `: ${error.message}` : ''}. edit the query to retry.`
  return status
}
