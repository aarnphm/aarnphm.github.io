import type { Extension } from '@codemirror/state'
import type { CodeMirror } from '@replit/codemirror-vim'

export type NotebookCodeEditor = {
  getValue(): string
  setValue(content: string): void
  setVimMode(enabled: boolean): Promise<void>
  focus(): void
  destroy(): void
}

export type NotebookCodeEditorLspConfig = { runtimeId: string; cellId: string; language: string }

type NotebookLspBridge = {
  createEditorExtension(config: NotebookCodeEditorLspConfig): Promise<Extension[]>
}

type NotebookCodeEditorConfig = {
  parent: HTMLElement
  initialContent?: string
  language?: string
  lsp?: NotebookCodeEditorLspConfig
  vimMode?: boolean
  onChange?: (content: string) => void
  onSubmit?: () => void
  onSave?: () => void
  onCancel?: () => void
}

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

function isNotebookLspBridge(value: unknown): value is NotebookLspBridge {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    typeof value.createEditorExtension === 'function'
  )
}

function readNotebookLspBridge(): NotebookLspBridge | undefined {
  const value = Reflect.get(globalThis, '__quartzNotebookLsp')
  return isNotebookLspBridge(value) ? value : undefined
}

async function notebookLspExtensions(
  config: NotebookCodeEditorLspConfig | undefined,
): Promise<Extension[]> {
  if (!config || !config.language.toLowerCase().startsWith('python')) return []
  let bridge = readNotebookLspBridge()
  if (!bridge) {
    try {
      await import(new URL('notebook-lsp.client.js', import.meta.url).href)
      bridge = readNotebookLspBridge()
    } catch {
      return []
    }
  }
  try {
    return bridge?.createEditorExtension(config) ?? []
  } catch (error) {
    console.warn('failed to create notebook lsp extension', error)
    return []
  }
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
    lspExtensions,
  ] = await Promise.all([
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/state'),
    import('@codemirror/view'),
    languageExtension(config.language),
    notebookLspExtensions(config.lsp),
  ])

  const notebookKeymap = (vimMode: boolean): Extension =>
    Prec.highest(
      keymap.of([
        {
          key: 'Mod-Enter',
          run: () => {
            config.onSubmit?.()
            return true
          },
        },
        ...(vimMode
          ? []
          : [
              {
                key: 'Escape',
                run: () => {
                  config.onCancel?.()
                  return true
                },
              },
            ]),
      ]),
    )

  type VimApi = typeof import('@replit/codemirror-vim')
  let vimApi: Pick<VimApi, 'getCM' | 'vim'> | undefined
  const loadVimApi = async (): Promise<Pick<VimApi, 'getCM' | 'vim'>> => {
    vimApi ??= await import('@replit/codemirror-vim')
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
      lspExtensions,
      history(),
      vimCompartment.of(initialVimExtension),
      keymapCompartment.of(notebookKeymap(initialVimMode)),
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
    async setVimMode(enabled: boolean) {
      view.dispatch({
        effects: [
          vimCompartment.reconfigure(await vimExtension(enabled)),
          keymapCompartment.reconfigure(notebookKeymap(enabled)),
        ],
      })
      if (enabled) await bindVimSave()
    },
    focus: () => view.focus(),
    destroy: () => view.destroy(),
  }
}
