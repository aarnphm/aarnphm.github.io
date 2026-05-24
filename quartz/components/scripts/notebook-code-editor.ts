import type { Extension } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView } from '@codemirror/view'
import type { CodeMirror } from '@replit/codemirror-vim'
import { forceParsing } from '@codemirror/language'
import type { NotebookLspConfig } from './notebook-lsp'

export type NotebookCodeEditor = {
  getValue(): string
  setValue(content: string): void
  highlightedLines(): HTMLElement[]
  setVimMode(enabled: boolean): Promise<void>
  focus(): void
  destroy(): void
}

type NotebookCodeEditorConfig = {
  parent: HTMLElement
  initialContent?: string
  language?: string
  vimMode?: boolean
  onChange?: (content: string) => void
  onSubmit?: () => void
  onSave?: () => void
  onCancel?: () => void
  lsp?: NotebookLspConfig
}

type VimApi = typeof import('@replit/codemirror-vim')
type NotebookCodeMirror = NonNullable<ReturnType<VimApi['getCM']>>
type NotebookVimModule = Pick<VimApi, 'Vim'>

let notebookVimBindingsConfigured = false
const notebookRunKeys = ['Mod-Enter', 'Ctrl-Enter', 'Shift-Enter', 'Alt-Enter'] as const

async function languageExtension(language: string | undefined): Promise<Extension> {
  const name = (language ?? '').trim().toLowerCase()
  if (name === 'javascript' || name === 'js') {
    const { javascript } = await import('@codemirror/lang-javascript')
    return javascript()
  }
  if (name === 'jsx') {
    const { javascript } = await import('@codemirror/lang-javascript')
    return javascript({ jsx: true })
  }
  if (name === 'typescript' || name === 'ts') {
    const { javascript } = await import('@codemirror/lang-javascript')
    return javascript({ typescript: true })
  }
  if (name === 'tsx') {
    const { javascript } = await import('@codemirror/lang-javascript')
    return javascript({ typescript: true, jsx: true })
  }
  if (name === 'go' || name === 'golang') {
    const { go } = await import('@codemirror/lang-go')
    return go()
  }
  if (name === 'rust' || name === 'rs') {
    const { rust } = await import('@codemirror/lang-rust')
    return rust()
  }
  if (name === 'zig') {
    const { zig } = await import('codemirror-lang-zig')
    return zig()
  }
  const { python } = await import('@codemirror/lang-python')
  return python()
}

function notebookPythonLanguage(language: string | undefined) {
  const name = (language ?? '').trim().toLowerCase()
  return name === 'python' || name === 'py' || name === 'ipython'
}

async function lspExtensions(config: NotebookLspConfig | undefined): Promise<readonly Extension[]> {
  if (!config?.enabled || !notebookPythonLanguage(config.language)) return []
  try {
    const { notebookLspExtensions } = await import('./notebook-lsp')
    return await notebookLspExtensions(config)
  } catch (error) {
    console.warn('notebook lsp extension failed to load', error)
    return []
  }
}

function offsetForLineStart(lines: string[], lineNumber: number): number {
  let offset = 0
  for (let index = 0; index < lineNumber - 1; index++) {
    offset += (lines[index]?.length ?? 0) + 1
  }
  return offset
}

function offsetForLineEnd(lines: string[], lineNumber: number): number {
  return offsetForLineStart(lines, lineNumber) + (lines[lineNumber - 1]?.length ?? 0)
}

function moveSelectedLines(cm: NotebookCodeMirror, direction: -1 | 1) {
  const view = cm.cm6
  const doc = view.state.doc
  const range = view.state.selection.main
  const from = Math.min(range.anchor, range.head)
  const to = Math.max(range.anchor, range.head)
  const startLineNumber = doc.lineAt(from).number
  const endLineNumber = doc.lineAt(to > from ? to - 1 : to).number
  if (direction < 0 && startLineNumber === 1) return
  if (direction > 0 && endLineNumber === doc.lines) return

  const lines = doc.toString().split('\n')
  const startIndex = startLineNumber - 1
  const selected = lines.splice(startIndex, endLineNumber - startLineNumber + 1)
  lines.splice(direction < 0 ? startIndex - 1 : startIndex + 1, 0, ...selected)

  const nextStartLineNumber = startLineNumber + direction
  const nextEndLineNumber = endLineNumber + direction
  view.dispatch({
    changes: { from: 0, to: doc.length, insert: lines.join('\n') },
    selection: {
      anchor: offsetForLineStart(lines, nextStartLineNumber),
      head: offsetForLineEnd(lines, nextEndLineNumber),
    },
    scrollIntoView: true,
  })
}

function configureNotebookVimBindings(vimApi: NotebookVimModule) {
  if (notebookVimBindingsConfigured) return
  notebookVimBindingsConfigured = true
  const { Vim } = vimApi
  const maps = [
    ['insert', 'jj', '<Esc>'],
    ['insert', 'jk', '<Esc>'],
    ['insert', '<A-BS>', '<C-w>'],
    ['insert', '<M-BS>', '<C-w>'],
    ['normal', 'Y', 'y$'],
    ['normal', 'D', 'd$'],
    ['normal', ';', ':'],
    ['normal', '\\', ':noh<CR>'],
    ['visual', '<', '<gv'],
    ['visual', '>', '>gv'],
  ] as const
  for (const [context, lhs, rhs] of maps) {
    Vim.noremap(lhs, rhs, context)
  }
  Vim.defineAction('notebookNoop', () => {})
  Vim.defineAction('notebookMoveSelectedLinesDown', cm => moveSelectedLines(cm, 1))
  Vim.defineAction('notebookMoveSelectedLinesUp', cm => moveSelectedLines(cm, -1))
  Vim.mapCommand('<Space>', 'action', 'notebookNoop', {}, { context: 'normal' })
  Vim.mapCommand('<Space>', 'action', 'notebookNoop', {}, { context: 'visual' })
  Vim.mapCommand(
    'J',
    'action',
    'notebookMoveSelectedLinesDown',
    {},
    { context: 'visual', isEdit: true },
  )
  Vim.mapCommand(
    'K',
    'action',
    'notebookMoveSelectedLinesUp',
    {},
    { context: 'visual', isEdit: true },
  )
}

function inlineHighlightStyle(source: HTMLElement, target: HTMLElement) {
  const style = getComputedStyle(source)
  if (style.color) {
    target.style.color = style.color
    target.style.setProperty('--shiki-light', style.color)
    target.style.setProperty('--shiki-dark', style.color)
  }
  if (style.fontStyle && style.fontStyle !== 'normal') target.style.fontStyle = style.fontStyle
  if (style.fontWeight && style.fontWeight !== '400') target.style.fontWeight = style.fontWeight
  if (style.textDecorationLine && style.textDecorationLine !== 'none') {
    target.style.textDecorationLine = style.textDecorationLine
  }
  const sourceChildren = Array.from(source.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement,
  )
  const targetChildren = Array.from(target.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement,
  )
  sourceChildren.forEach((child, index) => {
    const targetChild = targetChildren[index]
    if (targetChild) inlineHighlightStyle(child, targetChild)
  })
}

function cloneHighlightedNode(node: ChildNode): Node {
  const cloned = node.cloneNode(true)
  if (node instanceof HTMLElement && cloned instanceof HTMLElement) {
    inlineHighlightStyle(node, cloned)
  }
  return cloned
}

function highlightedLineSpans(view: CodeMirrorEditorView): HTMLElement[] {
  forceParsing(view, view.state.doc.length, 500)
  return Array.from(view.dom.querySelectorAll<HTMLElement>('.cm-content > .cm-line')).map(line => {
    const span = document.createElement('span')
    span.dataset.line = ''
    for (const child of Array.from(line.childNodes)) {
      span.append(cloneHighlightedNode(child))
    }
    return span
  })
}

export async function renderNotebookHighlightedLines(
  source: string,
  languageName: string | undefined,
): Promise<HTMLElement[]> {
  const [{ syntaxHighlighting, defaultHighlightStyle }, { EditorState }, { EditorView }, language] =
    await Promise.all([
      import('@codemirror/language'),
      import('@codemirror/state'),
      import('@codemirror/view'),
      languageExtension(languageName),
    ])
  const lineCount = source.split(/\r?\n/).length
  const host = document.createElement('div')
  host.style.position = 'fixed'
  host.style.left = '-10000px'
  host.style.top = '0'
  host.style.width = '1000px'
  host.style.height = `${Math.max(1, lineCount) * 24}px`
  host.style.visibility = 'hidden'
  host.style.pointerEvents = 'none'
  document.body.append(host)
  const state = EditorState.create({
    doc: source,
    extensions: [
      language,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      EditorView.theme({
        '&': { fontFamily: 'monospace', fontSize: '16px', lineHeight: '24px' },
        '.cm-scroller': { overflow: 'visible' },
      }),
    ],
  })
  const view = new EditorView({ state, parent: host })
  forceParsing(view, view.state.doc.length, 500)
  await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
  const lines = highlightedLineSpans(view)
  view.destroy()
  host.remove()
  return lines
}

export async function createNotebookCodeEditor(
  config: NotebookCodeEditorConfig,
): Promise<NotebookCodeEditor> {
  const [
    { defaultKeymap, historyKeymap, history, indentWithTab },
    { syntaxHighlighting, defaultHighlightStyle },
    { Compartment, EditorState, Prec },
    { EditorView, keymap, lineNumbers },
    language,
    lsp,
  ] = await Promise.all([
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/state'),
    import('@codemirror/view'),
    languageExtension(config.language),
    lspExtensions(config.lsp),
  ])

  const notebookKeymap = (): Extension =>
    Prec.highest(
      keymap.of([
        ...notebookRunKeys.map(key => ({
          key,
          run: () => {
            config.onSubmit?.()
            return true
          },
        })),
        {
          key: 'Mod-s',
          run: () => {
            config.onSave?.()
            return true
          },
        },
        {
          key: 'Ctrl-s',
          run: () => {
            config.onSave?.()
            return true
          },
        },
        {
          key: 'Escape',
          run: () => {
            config.onCancel?.()
            return true
          },
        },
      ]),
    )

  let vimApi: Pick<VimApi, 'getCM' | 'vim' | 'Vim'> | undefined
  const loadVimApi = async (): Promise<Pick<VimApi, 'getCM' | 'vim' | 'Vim'>> => {
    vimApi ??= await import('@replit/codemirror-vim')
    configureNotebookVimBindings(vimApi)
    return vimApi
  }

  const vimExtension = async (enabled: boolean): Promise<Extension> => {
    if (!enabled) return []
    const [{ vim }, { drawSelection }] = await Promise.all([
      loadVimApi(),
      import('@codemirror/view'),
    ])
    return [vim(), drawSelection()]
  }

  const initialVimMode = config.vimMode === true
  const vimCompartment = new Compartment()
  const keymapCompartment = new Compartment()
  const initialVimExtension = await vimExtension(initialVimMode)

  const updateListener = EditorView.updateListener.of(update => {
    if (update.docChanged) config.onChange?.(update.state.doc.toString())
  })

  const transparentTheme = EditorView.theme({
    '&': {
      backgroundColor: 'transparent !important',
      border: 'none !important',
      outline: 'none !important',
      fontSize: 'inherit',
      fontFamily: 'inherit !important',
      lineHeight: 'inherit',
      color: 'inherit',
      height: 'auto',
      padding: '0',
    },
    '&.cm-focused': {
      outline: 'none !important',
      border: 'none !important',
      boxShadow: 'none !important',
    },
    '.cm-content': { padding: '0 !important', minHeight: '20px', caretColor: 'inherit' },
    '.cm-gutter': { minHeight: '20px' },
    '.cm-scroller': { overflow: 'visible', fontFamily: 'inherit !important', fontSize: 'inherit' },
    '.cm-line': { padding: '0' },
    '.cm-cursor': { borderLeftColor: 'inherit' },
  })

  const state = EditorState.create({
    doc: config.initialContent ?? '',
    extensions: [
      lineNumbers(),
      language,
      lsp,
      history(),
      vimCompartment.of(initialVimExtension),
      keymapCompartment.of(notebookKeymap()),
      keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
      updateListener,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      transparentTheme,
    ],
  })

  const view = new EditorView({ state, parent: config.parent })
  const bindVimSave = async () => {
    if (!config.onSave) return
    const { getCM } = await loadVimApi()
    const cm: CodeMirror | null = getCM(view)
    if (cm) cm.save = config.onSave
  }
  if (initialVimMode) await bindVimSave()

  return {
    getValue: () => view.state.doc.toString(),
    setValue(content: string) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: content } })
    },
    highlightedLines: () => highlightedLineSpans(view),
    async setVimMode(enabled: boolean) {
      view.dispatch({
        effects: [
          vimCompartment.reconfigure(await vimExtension(enabled)),
          keymapCompartment.reconfigure(notebookKeymap()),
        ],
      })
      if (enabled) await bindVimSave()
    },
    focus: () => view.focus(),
    destroy: () => view.destroy(),
  }
}
