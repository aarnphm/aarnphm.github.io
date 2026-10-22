import { backendFor } from '../notebook/registry'
import { warmLanguageExtension } from './code-editor/highlighting'
import { configureNotebookVimBindings } from './code-editor/vim-navigation'
export {
  notebookCodeEditorLanguageExtension,
  renderNotebookHighlightedLines,
} from './code-editor/highlighting'
export {
  notebookSurroundKeyPlan,
  notebookSurroundPair,
  notebookSurroundingPairRange,
  notebookSurroundingPairRangeInText,
  notebookWordRangeAt,
} from './code-editor/surround'
export type {
  NotebookSurroundKeyPlan,
  NotebookSurroundingPairRange,
  NotebookTextRange,
} from './code-editor/surround'
export {
  notebookCursorCoordinate,
  notebookCursorCoordinateInText,
  notebookEditorStatusText,
  notebookEditorStatusTextInText,
  notebookLeapMotionForKey,
  notebookLeapTargets,
  notebookLeapTargetsInText,
  notebookVimNoremaps,
  registerNotebookSurroundBindings,
} from './code-editor/vim-navigation'
export type {
  NotebookCursorCoordinate,
  NotebookLeapMotion,
  NotebookLeapSearchRange,
  NotebookLeapStatus,
  NotebookLeapTarget,
  NotebookVimBindingsApi,
} from './code-editor/vim-navigation'
export { notebookWhitespaceGlyphs } from './code-editor/view-extensions'
export type {
  NotebookCodeEditor,
  NotebookWhitespaceGlyph,
  NotebookWhitespaceKind,
} from './code-editor/view-extensions'
export { createNotebookCodeEditor } from './code-editor/editor-session'

type NotebookCodeEditorWarmupConfig = {
  languages?: readonly string[]
  lsp?: boolean
  vimMode?: boolean
}

let notebookCodeEditorCoreWarmup: Promise<void> | undefined
let notebookCodeEditorLspWarmup: Promise<void> | undefined
let notebookCodeEditorVimWarmup: Promise<void> | undefined

export async function warmNotebookCodeEditorAssets(
  config: NotebookCodeEditorWarmupConfig = {},
): Promise<void> {
  notebookCodeEditorCoreWarmup ??= Promise.all([
    import('@codemirror/commands'),
    import('@codemirror/autocomplete'),
    import('@codemirror/language'),
    import('@codemirror/state'),
    import('@codemirror/view'),
  ]).then(() => {})

  const languages = config.languages && config.languages.length > 0 ? config.languages : ['python']
  const warmups: Promise<unknown>[] = [
    notebookCodeEditorCoreWarmup,
    ...languages.map(language => warmLanguageExtension(language)),
  ]

  if (config.lsp) {
    notebookCodeEditorLspWarmup ??= Promise.all(
      languages.map(async language => {
        const bridge = await backendFor(language)?.editor?.lspBridge?.()
        if (bridge) await import('../lsp/pyright').then(module => module.warmNotebookLspAssets())
      }),
    ).then(() => {})
    warmups.push(notebookCodeEditorLspWarmup)
  }

  if (config.vimMode) {
    notebookCodeEditorVimWarmup ??= import('@replit/codemirror-vim').then(vimApi => {
      configureNotebookVimBindings(vimApi)
    })
    warmups.push(notebookCodeEditorVimWarmup)
  }

  await Promise.all(warmups)
}
