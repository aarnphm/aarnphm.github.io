import { Element, Root as HastRoot } from 'hast'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toHtml } from 'hast-util-to-html'
import { toString } from 'hast-util-to-string'
import { s, h } from 'hastscript'
import { Root as MdRoot } from 'mdast'
import { createElement } from 'preact'
import render from 'preact-render-to-string'
// @ts-ignore
import Lexer from 'pseudocode/src/Lexer.js'
// @ts-ignore
import Parser from 'pseudocode/src/Parser.js'
// @ts-ignore
import Renderer from 'pseudocode/src/Renderer.js'
import { SKIP, visit } from 'unist-util-visit'
import {
  DisabledPluginNotice,
  disabledPluginNoticeStyle,
} from '../../components/DisabledPluginNotice'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { extractInlineMacros } from '../../util/latex'

export interface Options {
  code: string
  css: string
  renderer?: RendererOptions
}

/**
 * Options of the renderer itself. These are a subset of the options that can be passed to the Quartz plugin.
 * See the PseudocodeOptions type for the full list of options.
 */
interface RendererOptions {
  /**
   * The indent size of inside a control block, e.g. if, for, etc. The unit must be in 'em'. Default value: '1.2em'.
   */
  indentSize?: string
  /**
   * The delimiters used to start and end a comment region. Note that only line comments are supported. Default value: '//'.
   */
  commentDelimiter?: string
  /**
   * The punctuation that follows line number. Default value: ':'.
   */
  lineNumberPunc?: string
  /**
   * Whether line numbering is enabled. Default value: false.
   */
  lineNumber?: boolean
  /**
   * Whether block ending, like `end if`, end `procedure`, etc., are showned. Default value: false.
   */
  noEnd?: boolean
  /**
   * Set the caption counter to this new value.
   */
  captionCount?: number
  /**
   * Whether to set scope lines
   */
  scopeLines?: boolean
  /**
   * The prefix in the title of the algorithm. Default value: 'Algorithm'.
   */
  titlePrefix?: string

  mathEngine?: 'katex'
  mathRenderer?: (input: string) => string
}

const defaultOptions: Options = {
  code: 'pseudo',
  css: 'latex-pseudo',
  renderer: {
    indentSize: '0.6em',
    commentDelimiter: '  ▷',
    lineNumberPunc: ':',
    lineNumber: true,
    noEnd: false,
    scopeLines: false,
    captionCount: undefined,
    titlePrefix: 'Algorithm',
    mathEngine: 'katex',
    mathRenderer: undefined,
  },
}

export const PseudocodeFallback: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'PseudocodeFallback',
    htmlPlugins(ctx) {
      return [
        () => (tree: HastRoot) => {
          visit(tree, 'element', (node, index, parent) => {
            if (node.tagName !== 'pre' || index === undefined || !parent) return
            const code = node.children.find(
              child => child.type === 'element' && child.tagName === 'code',
            )
            if (!code || code.type !== 'element') return
            const classes = code.properties.className
            const language = `language-${opts.code}`
            if (!(Array.isArray(classes) ? classes.includes(language) : classes === language))
              return
            const notice = fromHtmlIsomorphic(
              render(
                createElement(DisabledPluginNotice, {
                  label: 'Pseudocode',
                  locale: ctx.cfg.configuration.locale,
                  source: toString(code).replace(/\n$/, ''),
                }),
              ),
              { fragment: true },
            ).children[0]
            if (!notice || notice.type !== 'element') return
            parent.children[index] = notice
            return SKIP
          })
        },
      ]
    },
    externalResources() {
      return { css: [{ content: disabledPluginNoticeStyle, inline: true }] }
    },
  }
}

function renderToString(input: string, options?: RendererOptions) {
  if (input === null || input === undefined) throw new ReferenceError('Input cannot be empty')

  const lexer = new Lexer(input)
  const parser = new Parser(lexer)
  const renderer = new Renderer(parser, options)
  if (options?.mathEngine || options?.mathRenderer) {
    renderer.backend ??= {}
    renderer.backend.name ??= options?.mathEngine
    renderer.backend.driver ??= {}
    renderer.backend.driver.renderToString ??= options?.mathRenderer
  }
  return renderer.toMarkup()
}

function parseMeta(meta: string | null, opts: Options) {
  if (!meta) meta = ''

  const lineNumberMatch = meta.match(/lineNumber=(false|true|0|1)/i)
  const lnum = lineNumberMatch?.[1] ?? null
  let enableLineNumber: boolean
  if (lnum) {
    enableLineNumber = lnum === 'true' || lnum === '1'
  } else {
    enableLineNumber = opts.renderer?.lineNumber as boolean
  }
  meta = meta.replace(lineNumberMatch?.[0] ?? '', '')

  return { enableLineNumber, meta }
}

export const Pseudocode: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  /**
   * Used to store the LaTeX raw string content in order as they are found in the markdown file.
   * They will be processed in the same order later on to be converted to HTML.
   */

  return {
    name: 'Pseudocode',
    markdownPlugins() {
      return [
        () => (tree: MdRoot, _file) => {
          // The library keeps its caption counter on Renderer; each document owns a new sequence.
          let captionCount: number | undefined = opts.renderer?.captionCount ?? 0
          visit(tree, 'code', node => {
            let { lang, meta, value } = node
            if (lang === opts.code) {
              const { enableLineNumber: lineNumber } = parseMeta(meta!, opts)

              // PERF: we are currently doing one round trip from text -> html -> hast
              // pseudocode (katex backend) --|renderToString|--> html string --|fromHtml|--> hast
              // ideally, we should cut this down to render directly to hast
              const [inlineMacros, algo] = extractInlineMacros(value ?? '')
              // TODO: Might be able to optimize.
              // find all $ enclosements in source, and add the preamble.
              const mathRegex = /\$(.*?)\$/g
              const algoWithPreamble = algo.replace(mathRegex, (_, p1) => {
                return `$${inlineMacros}${p1}$`
              })

              const rendered = fromHtmlIsomorphic(
                renderToString(algoWithPreamble!, { ...opts?.renderer, lineNumber, captionCount }),
                { fragment: true },
              ).children[0] as Element
              captionCount = undefined

              rendered.children = [
                h(
                  'span',
                  {
                    type: 'button',
                    class: 'clipboard-button ps-clipboard',
                    ariaLabel: 'Copy pseudocode to clipboard',
                  },
                  [
                    s('svg', { width: 16, height: 16, viewbox: '0 0 16 16', class: 'copy-icon' }, [
                      s('use', { href: '#github-copy' }),
                    ]),
                    s('svg', { width: 16, height: 16, viewbox: '0 0 16 16', class: 'check-icon' }, [
                      s('use', {
                        href: '#github-check',
                        fillRule: 'evenodd',
                        fill: 'rgb(63, 185, 80)',
                      }),
                    ]),
                  ],
                ),
                h('span', { class: 'ps-mathml' }, [
                  h('math', { xmlns: 'http://www.w3.org/1998/Math/MathML' }, [
                    h('semantics', [
                      h('annotation', { encoding: 'application/x-tex' }, [
                        { type: 'text', value: JSON.stringify(algoWithPreamble) },
                      ]),
                    ]),
                  ]),
                ]),
                ...rendered.children,
              ]
              rendered.properties['data-inline-macros'] = inlineMacros ?? ''

              node.type = 'html' as 'code'
              node.value = toHtml(rendered, { allowDangerousHtml: true })
            }
          })
        },
      ]
    },
  }
}
