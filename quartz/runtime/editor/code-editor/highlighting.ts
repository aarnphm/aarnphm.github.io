import type { Extension } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView } from '@codemirror/view'
import { ensureSyntaxTree, forceParsing, syntaxTreeAvailable } from '@codemirror/language'
import type { NotebookLspConfig } from '../../lsp/pyright'
import { codemirrorCodeLanguage } from '../../../util/codemirror-language'
import { backendFor } from '../../notebook/registry'

const notebookLanguageWarmups = new Map<string, Promise<Extension>>()

const notebookHighlightParseTimeoutMs = 1500

function languageWarmupKey(language: string | undefined): string {
  return (language ?? '').trim().toLowerCase() || 'python'
}

export async function notebookCodeEditorLanguageExtension(
  language: string | undefined,
): Promise<Extension> {
  const name = (language ?? '').trim().toLowerCase()
  const backendExtension = await backendFor(name)?.editor?.languageExtension?.()
  if (backendExtension) return backendExtension
  return codemirrorCodeLanguage(language)
}

export function warmLanguageExtension(language: string | undefined): Promise<Extension> {
  const key = languageWarmupKey(language)
  let warmup = notebookLanguageWarmups.get(key)
  if (!warmup) {
    warmup = notebookCodeEditorLanguageExtension(language)
    notebookLanguageWarmups.set(key, warmup)
  }
  return warmup
}

// Kernel names that @pierre/highlights does not list among its aliases.
const kernelLanguages: Record<string, string> = {
  python3: 'python',
  ipython: 'python',
  lhs: 'haskell',
  ghc: 'haskell',
  runghc: 'haskell',
  utop: 'ocaml',
  sqlite: 'sql',
  postgres: 'sql',
  postgresql: 'sql',
  webassembly: 'wasm',
}

/** Highlights lines for languages @pierre/highlights knows; others return undefined for CodeMirror. */
async function highlightsLines(
  source: string,
  language: string | undefined,
): Promise<HTMLElement[] | undefined> {
  const key = languageWarmupKey(language)
  const label = kernelLanguages[key] ?? key
  const { highlightCode, syntaxLanguage } = await import('../../../util/highlight-source')
  if (syntaxLanguage(label) === 'plain') return undefined
  const template = document.createElement('template')
  template.innerHTML = highlightCode(source, label).html
  return Array.from(template.content.children).filter(
    (line): line is HTMLElement => line instanceof HTMLElement && line.hasAttribute('data-line'),
  )
}

export async function lspExtensions(
  config: NotebookLspConfig | undefined,
): Promise<readonly Extension[]> {
  if (!config?.enabled) return []
  const bridgeFactory = backendFor(config.language)?.editor?.lspBridge
  if (!bridgeFactory) return []
  try {
    const bridge = await bridgeFactory()
    return await bridge.extensions(config)
  } catch (error) {
    console.warn('notebook lsp extension failed to load', error)
    return []
  }
}

function inlineHighlightStyle(source: HTMLElement, target: HTMLElement) {
  const style = getComputedStyle(source)
  if (style.color) {
    target.style.color = style.color
    target.style.setProperty('--syntax-light', style.color)
    target.style.setProperty('--syntax-dark', style.color)
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
  const upto = view.state.doc.length
  if (!syntaxTreeAvailable(view.state, upto)) {
    ensureSyntaxTree(view.state, upto, notebookHighlightParseTimeoutMs)
    forceParsing(view, upto, notebookHighlightParseTimeoutMs)
  }
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
  const highlighted = await highlightsLines(source, languageName)
  if (highlighted) return highlighted

  const [{ syntaxHighlighting, defaultHighlightStyle }, { EditorState }, { EditorView }, language] =
    await Promise.all([
      import('@codemirror/language'),
      import('@codemirror/state'),
      import('@codemirror/view'),
      notebookCodeEditorLanguageExtension(languageName),
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
  highlightedLineSpans(view)
  await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
  const lines = highlightedLineSpans(view)
  view.destroy()
  host.remove()
  return lines
}
