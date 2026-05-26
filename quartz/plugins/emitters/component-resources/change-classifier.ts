import type { ChangeEvent } from '../../../types/plugin'
import { isWorkerEntryPath } from '../../../util/workers'
import {
  collaborativeCommentsAssetEntries,
  collaborativeCommentsAssetPrefixes,
  emojiAssetSourceDir,
  notebookRuntimeAssetEntries,
  notebookRuntimeAssetPrefixes,
  notebookRuntimeInlineEntry,
  semanticWorkerAssetEntries,
  semanticWorkerEntry,
} from './asset-paths'

export type ComponentResourceChanges = {
  indexStylesheet: boolean
  notebookRuntime: boolean
  notebookRuntimePageScript: boolean
  pageScripts: boolean
  collaborativeComments: boolean
  semanticWorker: boolean
  semanticWorkerDeleted: boolean
  emoji: boolean
  genericWorkerChanges: ChangeEvent[]
}

export function isNotebookRuntimeAssetChange(changePath: string): boolean {
  if (notebookRuntimeAssetEntries.has(changePath)) return true
  return notebookRuntimeAssetPrefixes.some(prefix => changePath.startsWith(prefix))
}

export function isNotebookRuntimePageScriptChange(changePath: string): boolean {
  return changePath === notebookRuntimeInlineEntry
}

export function isPageScriptChange(changePath: string): boolean {
  if (changePath.startsWith('quartz/components/scripts/')) return true
  return (
    changePath === 'quartz/util/mime.ts' ||
    changePath === 'quartz/util/path.ts' ||
    changePath === 'quartz/util/stacked-notes.ts' ||
    changePath === 'quartz/util/type-guards.ts' ||
    changePath === 'quartz/util/wikipedia.ts'
  )
}

export function isCollaborativeCommentsAssetChange(changePath: string): boolean {
  if (collaborativeCommentsAssetEntries.has(changePath)) return true
  return collaborativeCommentsAssetPrefixes.some(prefix => changePath.startsWith(prefix))
}

export function isSemanticWorkerAssetChange(changePath: string): boolean {
  return semanticWorkerAssetEntries.has(changePath)
}

export function isEmojiAssetChange(changePath: string): boolean {
  return changePath.startsWith(`${emojiAssetSourceDir}/`) && changePath.endsWith('.json')
}

export function isIndexStylesheetChange(changePath: string): boolean {
  return changePath.startsWith('quartz/styles/') && changePath.endsWith('.scss')
}

export function classifyResourceChanges(
  changeEvents: readonly ChangeEvent[],
): ComponentResourceChanges {
  const notebookRuntimePageScript = changeEvents.some(changeEvent =>
    isNotebookRuntimePageScriptChange(changeEvent.path),
  )
  const pageScripts =
    notebookRuntimePageScript ||
    changeEvents.some(changeEvent => isPageScriptChange(changeEvent.path))

  return {
    indexStylesheet: changeEvents.some(changeEvent => isIndexStylesheetChange(changeEvent.path)),
    notebookRuntime: changeEvents.some(changeEvent =>
      isNotebookRuntimeAssetChange(changeEvent.path),
    ),
    notebookRuntimePageScript,
    pageScripts,
    collaborativeComments: changeEvents.some(changeEvent =>
      isCollaborativeCommentsAssetChange(changeEvent.path),
    ),
    semanticWorker: changeEvents.some(changeEvent => isSemanticWorkerAssetChange(changeEvent.path)),
    semanticWorkerDeleted: changeEvents.some(
      changeEvent => changeEvent.path === semanticWorkerEntry && changeEvent.type === 'delete',
    ),
    emoji: changeEvents.some(changeEvent => isEmojiAssetChange(changeEvent.path)),
    genericWorkerChanges: changeEvents.filter(
      changeEvent =>
        isWorkerEntryPath(changeEvent.path) && changeEvent.path !== semanticWorkerEntry,
    ),
  }
}
