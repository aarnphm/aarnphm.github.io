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
  xsltPolyfillAssetEntries,
} from './asset-paths'

const indexStylesheetComponentStyles = new Set([
  'quartz/components/styles/audio.scss',
  'quartz/components/styles/clipboard.scss',
  'quartz/components/styles/popover.scss',
  'quartz/components/styles/pseudocode.scss',
  'quartz/components/styles/twitter.scss',
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
  xsltPolyfill: boolean
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
    changePath.startsWith('quartz/util/') &&
    /\.[jt]sx?$/.test(changePath) &&
    !/\.(?:test|spec)\.[jt]sx?$/.test(changePath)
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
    componentStyles: changeEvents.some(
      changeEvent =>
        isComponentStylesheetChange(changeEvent.path) ||
        isSharedStylePartialChange(changeEvent.path),
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
    xsltPolyfill: changeEvents.some(changeEvent => xsltPolyfillAssetEntries.has(changeEvent.path)),
    genericWorkerChanges: changeEvents.filter(
      changeEvent =>
        isWorkerEntryPath(changeEvent.path) && changeEvent.path !== semanticWorkerEntry,
    ),
  }
}

// Component stylesheets `@use` these partials, so their edits recompile component.css as well.
const sharedStylePartials = new Set(['quartz/styles/variables.scss', 'quartz/styles/mixin.scss'])

function isSharedStylePartialChange(changePath: string): boolean {
  return sharedStylePartials.has(changePath)
}

export type SourceRebuildScope =
  | { kind: 'full' }
  | { kind: 'partial'; staticFiles: boolean; styles: boolean; scripts: boolean }

const clientScriptRoots = [
  'quartz/components/',
  'quartz/util/',
  'quartz/workers/',
  'quartz/runtime/',
]
const inlineScriptPattern = /\.inline\.[jt]s$/
const scriptPattern = /\.[jt]sx?$/

function isClientScriptChange(changePath: string, serverInputs: ReadonlySet<string>): boolean {
  // The build bundle loads an inline script as text, so its edits only change page chunks.
  if (inlineScriptPattern.test(changePath)) return true
  return (
    !serverInputs.has(changePath) &&
    scriptPattern.test(changePath) &&
    clientScriptRoots.some(root => changePath.startsWith(root))
  )
}

/**
 * Decides how much of the site a source edit invalidates. `serverInputs` is the esbuild metafile
 * closure of the build bundle (before and after the edit). A module outside it cannot change rendered
 * HTML, so it only needs the client bundles (ComponentResources, LazyScripts). Anything unrecognised
 * is a full rebuild.
 */
export function sourceRebuildScope(
  changedPaths: readonly string[],
  serverInputs: ReadonlySet<string>,
): SourceRebuildScope {
  if (changedPaths.length === 0) return { kind: 'full' }
  const scope = { kind: 'partial' as const, staticFiles: false, styles: false, scripts: false }
  for (const changePath of changedPaths) {
    if (changePath.startsWith('quartz/static/')) {
      scope.staticFiles = true
    } else if (
      isComponentStylesheetChange(changePath) ||
      isIndexStylesheetChange(changePath) ||
      isStaticStylesheetChange(changePath)
    ) {
      scope.styles = true
    } else if (isClientScriptChange(changePath, serverInputs)) {
      scope.scripts = true
    } else if (
      serverInputs.has(changePath) ||
      !(changePath.startsWith('quartz/scripts/') || changePath.endsWith('.py'))
    ) {
      return { kind: 'full' }
    }
    // Remaining paths are CLI scripts and Python helpers that no bundle imports.
  }
  return scope
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
    changes.xsltPolyfill ||
    changes.genericWorkerChanges.length > 0
  )
}
