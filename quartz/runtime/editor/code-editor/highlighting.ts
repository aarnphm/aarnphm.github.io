import type { Extension } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView } from '@codemirror/view'
import type { LanguageFn } from 'highlight.js'
import { ensureSyntaxTree, forceParsing, syntaxTreeAvailable } from '@codemirror/language'
import type { NotebookLspConfig } from '../../lsp/pyright'
import { codemirrorCodeLanguage } from '../../../util/codemirror-language'
import { backendFor } from '../../notebook/registry'

const notebookLanguageWarmups = new Map<string, Promise<Extension>>()

const notebookHighlightJsLanguages = new Set<string>()

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

const defineLeanHighlightJs: LanguageFn = hljs => ({
  name: 'Lean',
  keywords: {
    keyword:
      'theorem lemma def abbrev example instance class structure inductive mutual where deriving extends namespace section end open import variable variables universe axiom opaque attribute macro macro_rules syntax notation infixl infixr infix prefix postfix set_option noncomputable partial unsafe private protected local scoped fun do let have show suffices match with if then else from by calc return try catch finally for in while continue break Type Prop Sort sorry admit',
    built_in:
      'intro intros apply exact refine rw rwa simp simp_all omega decide rfl constructor cases rcases obtain induction exists use left right split contradiction assumption trivial ring linarith nlinarith norm_num field_simp ext funext congr subst unfold change specialize generalize gcongr positivity aesop tauto Nat Int Float String Bool List Array Option IO Unit',
    literal: 'true false',
  },
  contains: [
    hljs.COMMENT('--', '$'),
    hljs.COMMENT('/-', '-/', { contains: ['self'] }),
    hljs.QUOTE_STRING_MODE,
    hljs.C_NUMBER_MODE,
    { className: 'meta', begin: /@\[[^\]]*\]/ },
    { className: 'keyword', begin: /#[A-Za-z_][A-Za-z0-9_']*/ },
  ],
})

const defineZigHighlightJs: LanguageFn = hljs => ({
  name: 'Zig',
  keywords: {
    keyword:
      'addrspace align allowzero and anyframe anytype asm async await break callconv catch comptime const continue defer else enum errdefer error export extern fn for if inline linksection noalias noinline nosuspend opaque or orelse packed pub resume return struct suspend switch test threadlocal try union unreachable usingnamespace var volatile while',
    literal: 'true false null undefined',
    type: 'bool f16 f32 f64 f80 f128 i8 u8 i16 u16 i32 u32 i64 u64 i128 u128 isize usize c_char c_short c_ushort c_int c_uint c_long c_ulong c_longlong c_ulonglong c_longdouble anyerror void noreturn type comptime_int comptime_float',
  },
  contains: [
    hljs.C_LINE_COMMENT_MODE,
    hljs.QUOTE_STRING_MODE,
    hljs.APOS_STRING_MODE,
    hljs.C_NUMBER_MODE,
    { className: 'built_in', begin: /@[A-Za-z_][A-Za-z0-9_]*/ },
  ],
})

async function notebookHighlightJsLanguage(
  language: string | undefined,
): Promise<{ name: string; define: LanguageFn } | undefined> {
  const name = languageWarmupKey(language)
  if (
    name === 'python' ||
    name === 'py' ||
    name === 'python3' ||
    name === 'ipython' ||
    name === 'mojo'
  ) {
    return { name: 'python', define: (await import('highlight.js/lib/languages/python')).default }
  }
  if (name === 'javascript' || name === 'js' || name === 'jsx') {
    return {
      name: 'javascript',
      define: (await import('highlight.js/lib/languages/javascript')).default,
    }
  }
  if (name === 'typescript' || name === 'ts' || name === 'tsx') {
    return {
      name: 'typescript',
      define: (await import('highlight.js/lib/languages/typescript')).default,
    }
  }
  if (name === 'go' || name === 'golang') {
    return { name: 'go', define: (await import('highlight.js/lib/languages/go')).default }
  }
  if (name === 'rust' || name === 'rs') {
    return { name: 'rust', define: (await import('highlight.js/lib/languages/rust')).default }
  }
  if (name === 'c') {
    return { name: 'c', define: (await import('highlight.js/lib/languages/c')).default }
  }
  if (name === 'cpp' || name === 'c++' || name === 'cxx') {
    return { name: 'cpp', define: (await import('highlight.js/lib/languages/cpp')).default }
  }
  if (name === 'wasm' || name === 'wat' || name === 'webassembly') {
    return { name: 'wasm', define: (await import('highlight.js/lib/languages/wasm')).default }
  }
  if (
    name === 'haskell' ||
    name === 'hs' ||
    name === 'lhs' ||
    name === 'ghc' ||
    name === 'runghc'
  ) {
    return { name: 'haskell', define: (await import('highlight.js/lib/languages/haskell')).default }
  }
  if (name === 'ocaml' || name === 'ml' || name === 'mli' || name === 'utop') {
    return { name: 'ocaml', define: (await import('highlight.js/lib/languages/ocaml')).default }
  }
  if (name === 'bash' || name === 'sh' || name === 'shell' || name === 'zsh') {
    return { name: 'bash', define: (await import('highlight.js/lib/languages/bash')).default }
  }
  if (name === 'json' || name === 'jsonc') {
    return { name: 'json', define: (await import('highlight.js/lib/languages/json')).default }
  }
  if (name === 'css' || name === 'scss') {
    return { name: 'css', define: (await import('highlight.js/lib/languages/css')).default }
  }
  if (name === 'html' || name === 'xml' || name === 'svg') {
    return { name: 'xml', define: (await import('highlight.js/lib/languages/xml')).default }
  }
  if (name === 'sql' || name === 'sqlite' || name === 'postgres' || name === 'postgresql') {
    return { name: 'sql', define: (await import('highlight.js/lib/languages/sql')).default }
  }
  if (name === 'yaml' || name === 'yml') {
    return { name: 'yaml', define: (await import('highlight.js/lib/languages/yaml')).default }
  }
  if (name === 'lean' || name === 'lean4') {
    return { name: 'lean', define: defineLeanHighlightJs }
  }
  if (name === 'zig') {
    return { name: 'zig', define: defineZigHighlightJs }
  }
}

async function highlightNotebookSource(source: string, language: string | undefined) {
  const languageDefinition = await notebookHighlightJsLanguage(language)
  if (!languageDefinition) return undefined
  const hljs = (await import('highlight.js/lib/core')).default
  if (!notebookHighlightJsLanguages.has(languageDefinition.name)) {
    hljs.registerLanguage(languageDefinition.name, languageDefinition.define)
    notebookHighlightJsLanguages.add(languageDefinition.name)
  }
  return hljs.highlight(source, { language: languageDefinition.name, ignoreIllegals: true }).value
}

function createHighlightedLine(): HTMLElement {
  const line = document.createElement('span')
  line.dataset.line = ''
  return line
}

function appendWrappedText(line: HTMLElement, wrappers: readonly HTMLElement[], text: string) {
  if (text.length === 0) return
  let node: Node = document.createTextNode(text)
  for (let index = wrappers.length - 1; index >= 0; index -= 1) {
    const wrapper = wrappers[index]?.cloneNode(false)
    if (!(wrapper instanceof HTMLElement)) continue
    wrapper.append(node)
    node = wrapper
  }
  line.append(node)
}

function highlightedHtmlLineSpans(html: string): HTMLElement[] {
  const template = document.createElement('template')
  template.innerHTML = html
  const lines = [createHighlightedLine()]
  const wrappers: HTMLElement[] = []
  const currentLine = () => lines[lines.length - 1] ?? lines[0]
  const startLine = () => {
    lines.push(createHighlightedLine())
  }
  const visit = (node: ChildNode) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const parts = (node.textContent ?? '').split('\n')
      parts.forEach((part, index) => {
        appendWrappedText(currentLine(), wrappers, part)
        if (index < parts.length - 1) startLine()
      })
      return
    }
    if (node instanceof HTMLElement) {
      wrappers.push(node)
      for (const child of Array.from(node.childNodes)) visit(child)
      wrappers.pop()
      return
    }
    for (const child of Array.from(node.childNodes)) visit(child)
  }
  for (const child of Array.from(template.content.childNodes)) visit(child)
  return lines
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
  const highlightedSource = await highlightNotebookSource(source, languageName)
  if (highlightedSource !== undefined) return highlightedHtmlLineSpans(highlightedSource)

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
