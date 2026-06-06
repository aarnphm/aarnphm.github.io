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

const indexStylesheetComponentStyles = new Set([
  'quartz/components/styles/audio.scss',
  'quartz/components/styles/clipboard.scss',
  'quartz/components/styles/popover.scss',
  'quartz/components/styles/pseudocode.scss',
])

const staticStylesheetEntries = new Set([
  'quartz/components/styles/collapseHeader.inline.scss',
  'quartz/components/styles/mermaid.inline.scss',
  'quartz/components/styles/protected.scss',
  'quartz/components/styles/sidenotes.inline.scss',
  'quartz/components/styles/signatures.scss',
  'quartz/components/styles/telescopic.inline.scss',
])

const staticScriptEntries = new Set([
  'quartz/components/scripts/collapse-header.inline.ts',
  'quartz/components/scripts/pdf.inline.ts',
  'quartz/components/scripts/transclude.inline.ts',
])

export type ComponentResourceChanges = {
  componentStyles: boolean
  staticStyles: boolean
  staticScripts: boolean
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
  if (isStaticScriptChange(changePath)) return false
  if (changePath.startsWith('quartz/components/scripts/')) return true
  if (changePath.startsWith('quartz/components/') && /\.(tsx|ts|jsx|js)$/.test(changePath)) {
    return true
  }
  return (
    changePath === 'quartz/util/mime.ts' ||
    changePath === 'quartz/util/path.ts' ||
    changePath === 'quartz/util/lesswrong.ts' ||
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
  if (changePath.startsWith('quartz/styles/') && changePath.endsWith('.scss')) return true
  return indexStylesheetComponentStyles.has(changePath)
}

export function isComponentStylesheetChange(changePath: string): boolean {
  return (
    changePath.startsWith('quartz/components/styles/') &&
    changePath.endsWith('.scss') &&
    !isStaticStylesheetChange(changePath) &&
    !isIndexStylesheetChange(changePath)
  )
}

export function isStaticStylesheetChange(changePath: string): boolean {
  return staticStylesheetEntries.has(changePath)
}

export function isStaticScriptChange(changePath: string): boolean {
  return staticScriptEntries.has(changePath)
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
    componentStyles: changeEvents.some(changeEvent =>
      isComponentStylesheetChange(changeEvent.path),
    ),
    staticStyles: changeEvents.some(changeEvent => isStaticStylesheetChange(changeEvent.path)),
    staticScripts: changeEvents.some(changeEvent => isStaticScriptChange(changeEvent.path)),
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

export function hasComponentResourceChanges(changes: ComponentResourceChanges): boolean {
  return (
    changes.componentStyles ||
    changes.staticStyles ||
    changes.staticScripts ||
    changes.indexStylesheet ||
    changes.notebookRuntime ||
    changes.notebookRuntimePageScript ||
    changes.pageScripts ||
    changes.collaborativeComments ||
    changes.semanticWorker ||
    changes.semanticWorkerDeleted ||
    changes.emoji ||
    changes.genericWorkerChanges.length > 0
  )
}
