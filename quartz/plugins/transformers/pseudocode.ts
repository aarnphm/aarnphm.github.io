import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toHtml } from 'hast-util-to-html'
import { s, h } from 'hastscript'
import { Root as MdRoot } from 'mdast'
// @ts-ignore
import Lexer from 'pseudocode/src/Lexer.js'
// @ts-ignore
import Parser from 'pseudocode/src/Parser.js'
// @ts-ignore
import Renderer from 'pseudocode/src/Renderer.js'
import { SKIP, visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { extractInlineMacros } from '../../util/latex'
import { onModuleRelease } from '../../util/module-caches'

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

interface CachedBlock {
  html: string
  captionCount: number
}

// Bound retained strings per parser worker, including source keys, across edits.
const maxCacheBytes = 8 * 1024 * 1024
const maxCacheEntries = 512
const blockCache = new Map<string, CachedBlock>()
let cacheBytes = 0
onModuleRelease(() => {
  blockCache.clear()
  cacheBytes = 0
})

function cacheBlock(key: string, block: CachedBlock): void {
  const bytes = 2 * (key.length + block.html.length)
  if (bytes > maxCacheBytes) return
  while (cacheBytes + bytes > maxCacheBytes || blockCache.size >= maxCacheEntries) {
    const oldest = blockCache.entries().next().value
    if (!oldest) break
    blockCache.delete(oldest[0])
    cacheBytes -= 2 * (oldest[0].length + oldest[1].html.length)
  }
  blockCache.set(key, block)
  cacheBytes += bytes
}

function renderToString(input: string, options?: RendererOptions) {
  if (input === null || input === undefined) throw new ReferenceError('Input cannot be empty')

  const lexer = new Lexer(input)
  const parser = new Parser(lexer)
  const renderer = new Renderer(parser, options)
  if (options?.mathRenderer) {
    renderer.backend = { name: 'katex', driver: { renderToString: options.mathRenderer } }
  }
  return renderer.toMarkup()
}

function parseMeta(meta: string | null | undefined, opts: Options) {
  if (!meta) meta = ''

  const lineNumberMatch = meta.match(/lineNumber=(false|true|0|1)/i)
  const lnum = lineNumberMatch?.[1] ?? null
  let enableLineNumber: boolean
  if (lnum) {
    enableLineNumber = lnum.toLowerCase() === 'true' || lnum === '1'
  } else {
    enableLineNumber = opts.renderer?.lineNumber ?? false
  }
  meta = meta.replace(lineNumberMatch?.[0] ?? '', '')

  return { enableLineNumber, meta }
}

export const Pseudocode: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'Pseudocode',
    markdownPlugins() {
      return [
        () => (tree: MdRoot, file) => {
          // The library keeps its caption counter on Renderer; each document owns a new sequence.
          let captionCount = opts.renderer?.captionCount ?? 0
          visit(tree, 'code', (node, index, parent) => {
            let { lang, meta, value } = node
            if (lang === opts.code && index !== undefined && parent) {
              const { enableLineNumber: lineNumber } = parseMeta(meta, opts)

              file.data.hasMath = true
              // Callbacks may read external state; only declarative renderer settings are cacheable.
              const key = opts.renderer?.mathRenderer
                ? undefined
                : JSON.stringify([value, lineNumber, captionCount, opts.renderer])
              const cached = key === undefined ? undefined : blockCache.get(key)
              if (cached && key !== undefined) {
                blockCache.delete(key)
                blockCache.set(key, cached)
                captionCount = cached.captionCount
                parent.children[index] = {
                  type: 'html',
                  value: cached.html,
                  position: node.position,
                }
                return SKIP
              }

              const [inlineMacros, algo] = extractInlineMacros(value ?? '')
              const mathRegex = /\$(.*?)\$/g
              const algoWithPreamble = algo.replace(mathRegex, (_, p1) => {
                return `$${inlineMacros}${p1}$`
              })

              const rendered = fromHtmlIsomorphic(
                renderToString(algoWithPreamble, { ...opts?.renderer, lineNumber, captionCount }),
                { fragment: true },
              ).children[0]
              if (!rendered || rendered.type !== 'element') {
                throw new Error('Pseudocode renderer returned no root element')
              }
              captionCount = Renderer.captionCount

              rendered.children = [
                h(
                  'button',
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

              const html = toHtml(rendered, { allowDangerousHtml: true })
              if (key !== undefined) cacheBlock(key, { html, captionCount })
              parent.children[index] = { type: 'html', value: html, position: node.position }
              return SKIP
            }
          })
        },
      ]
    },
  }
}
