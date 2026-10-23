import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'

export type NotebookCacheReason =
  | 'force'
  | 'entry-missing'
  | 'implementation-changed'
  | 'notebook-changed'
  | 'runtime-changed'
  | 'modules-changed'
  | 'key-changed'
  | 'metadata-match'
  | 'blob-restored'
  | 'spill-restored'
  | 'blob-unavailable'
  | 'blob-invalid'
  | 'assets-unavailable'

export function notebookCacheMiss(
  expected: string | undefined,
  cached: string | undefined,
): NotebookCacheReason | undefined {
  if (!expected) return 'force'
  if (!cached) return 'entry-missing'
  if (cached === expected || cached.endsWith(`|${expected}`)) return undefined
  const previous = cached.split('\u0001')
  const next = expected.split('\u0001')
  if (previous.length !== 4 || next.length !== 4) return 'key-changed'
  if (previous[0] !== next[0]) return 'implementation-changed'
  if (previous[1] !== next[1]) return 'notebook-changed'
  if (previous[2] !== next[2]) return 'runtime-changed'
  if (previous[3] !== next[3]) return 'modules-changed'
  return 'key-changed'
}

export function notebookCacheReceipt(
  ctx: BuildCtx,
  file: FilePath,
  phase: 'prepare' | 'restore',
  status: 'candidate' | 'hit' | 'miss' | 'reload',
  reason: NotebookCacheReason,
): void {
  console.log(
    `__QUARTZ_NOTEBOOK_CACHE__${JSON.stringify({ epoch: ctx.buildId, file, phase, status, reason })}`,
  )
}
