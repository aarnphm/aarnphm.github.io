import {
  cellSourceLegacyPrefix,
  cellSourceStoragePrefix,
} from '../../../runtime/notebook/persistence'
import {
  notebookKernelCommandEvent,
  notebookKernelRequestEvent,
  notebookKernelRunAllEvent,
  type NotebookKernelCommand,
  type NotebookKernelSnapshot,
} from '../../../util/notebook-kernel-events'
import { notebookLanguageIconSvg } from '../../../util/notebook/render/icons'

export function notebookKernelSnapshots(): NotebookKernelSnapshot[] {
  const snapshots: NotebookKernelSnapshot[] = []
  const event: CustomEventMap['notebookkernelrequest'] = new CustomEvent(
    notebookKernelRequestEvent,
    { detail: { respond: snapshot => snapshots.push(snapshot) } },
  )
  document.dispatchEvent(event)
  return snapshots
}

export function notebookKernelSourceLabel(sourcePath: string): string {
  const name = sourcePath.split('/').filter(Boolean).at(-1)
  return name ?? sourcePath
}

export function notebookKernelStatusLabel(snapshot: NotebookKernelSnapshot): string | undefined {
  if (snapshot.status === 'ready') return undefined
  const detail = snapshot.status === 'running' ? snapshot.runningCellId : snapshot.statusDetail
  if (detail) {
    return `${snapshot.status} ${detail}`
  }
  return snapshot.status
}

export function notebookKernelLanguageToken(language: string): string {
  const normalized = language.toLowerCase()
  if (normalized === 'python') return 'py'
  if (normalized === 'javascript') return 'js'
  if (normalized === 'typescript') return 'ts'
  if (normalized === 'haskell') return 'hs'
  if (normalized === 'rust') return 'rs'
  if (normalized === 'ocaml') return 'ml'
  if (normalized === 'bash') return 'sh'
  return normalized
}

export function notebookKernelLanguageIcon(language: string): string {
  const normalized = language.toLowerCase()
  if (Object.hasOwn(notebookLanguageIconSvg, normalized)) return notebookLanguageIconSvg[normalized]
  return notebookLanguageIconSvg.text
}

export function dispatchNotebookKernelCommand(
  snapshot: NotebookKernelSnapshot,
  command: NotebookKernelCommand,
) {
  const event: CustomEventMap['notebookkernelcommand'] = new CustomEvent(
    notebookKernelCommandEvent,
    { detail: { runtimeId: snapshot.runtimeId, language: snapshot.language, command } },
  )
  document.dispatchEvent(event)
}

export function dispatchNotebookKernelRunAll() {
  const event: CustomEventMap['notebookkernelrunall'] = new CustomEvent(notebookKernelRunAllEvent, {
    detail: {},
  })
  document.dispatchEvent(event)
}

export function clearNotebookLocalSources(storage: Storage = localStorage): number {
  const keys: string[] = []
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (
      key &&
      (key.startsWith(cellSourceStoragePrefix) || key.startsWith(cellSourceLegacyPrefix))
    ) {
      keys.push(key)
    }
  }
  for (const key of keys) storage.removeItem(key)
  return keys.length
}
