import { Element, Properties } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import { Code, Root as MdRoot } from 'mdast'
import { load, tex, dvi2svg } from 'node-tikzjax'
import { visit } from 'unist-util-visit'
import { svgOptions } from '../../components/svg'
import { QuartzTransformerPlugin } from '../../types/plugin'

const TIKZ_TIMEOUT = 30_000

async function tex2svg(
  input: string,
  opts: { showConsole: boolean; disableSanitize: boolean; disableOptimize: boolean },
) {
  await load()
  const dvi = await tex(input, {
    texPackages: { pgfplots: '', amsmath: 'intlimits' },
    tikzLibraries: 'arrows.meta,calc,positioning',
    addToPreamble: '% comment',
    showConsole: opts.showConsole,
  })
  return dvi2svg(dvi, {
    disableSanitize: opts.disableSanitize,
    disableOptimize: opts.disableOptimize,
  })
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`tikz: ${label} timed out after ${ms}ms`)), ms)
      timer.unref()
    }),
  ])
}

interface TikzNode {
  index: number
  value: string
  parent: MdRoot
  base64?: string
  disableSanitize: boolean
}

function parseStyle(meta: string | null | undefined): string {
  if (!meta) return ''
  const styleMatch = meta.match(/style\s*=\s*["']([^"']+)["']/)
  return styleMatch ? styleMatch[1] : ''
}

const docs = (node: Code): string => JSON.stringify(node.value)

function makeTikzGraph(node: Code, svg: string, style?: string): Element {
  const mathMl = h(
    'span.tikz-mathml',
    h(
      'math',
      { xmlns: 'http://www.w3.org/1998/Math/MathML' },
      h(
        'semantics',
        h('annotation', { encoding: 'application/x-tex' }, { type: 'text', value: docs(node) }),
      ),
    ),
  )

  const sourceCodeCopy = h(
    'figcaption',
    h('em', [{ type: 'text', value: 'source code' }]),
    h(
      'button.source-code-button',
      {
        ariaLabel: 'copy source code for this tikz graph',
        title: 'copy source code for this tikz graph',
      },
      s(
        'svg.source-icon',
        {
          ...svgOptions,
          width: 12,
          height: 16,
          viewbox: '0 -4 24 24',
          fill: 'none',
          stroke: 'currentColor',
          strokewidth: 2,
        },
        s('use', { href: '#code-icon' }),
      ),
      s(
        'svg.check-icon',
        { ...svgOptions, width: 12, height: 16, viewbox: '0 -4 16 16' },
        s('use', { href: '#github-check' }),
      ),
    ),
  )

  const properties: Properties = { 'data-remark-tikz': true, style: '' }
  if (style) properties.style = style

  const encoded = Buffer.from(svg).toString('base64')
  const imgNode = h('img', {
    src: `data:image/svg+xml;base64,${encoded}`,
    alt: 'tikz diagram',
    loading: 'lazy',
    decoding: 'async',
  })

  return h('figure.tikz', properties, mathMl, imgNode, sourceCodeCopy)
}

interface Options {
  showConsole: boolean
  disableOptimize: boolean
}

const defaultOpts: Options = { showConsole: false, disableOptimize: true }

export const TikzJax: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const o = { ...defaultOpts, ...opts }
  return {
    name: 'TikzJax',
    markdownPlugins() {
      return [
        () => async tree => {
          const nodes: TikzNode[] = []
          visit(tree, 'code', (node: Code, index, parent) => {
            let { lang, meta, value } = node
            if (lang === 'tikz') {
              const base64Match = meta?.match(/alt\s*=\s*"data:image\/svg\+xml;base64,([^"]+)"/)
              let base64String = undefined
              if (base64Match) {
                base64String = Buffer.from(base64Match[1], 'base64').toString()
              }
              nodes.push({
                index: index as number,
                parent: parent as MdRoot,
                value,
                base64: base64String,
                disableSanitize: !!meta?.match(/disableSanitize\s*=\s*true/),
              })
            }
          })

          for (let i = 0; i < nodes.length; i++) {
            const { index, parent, value, base64, disableSanitize } = nodes[i]
            let svg
            if (base64 !== undefined) svg = base64
            else {
              try {
                svg = await withTimeout(
                  tex2svg(value, { disableSanitize, ...o }),
                  TIKZ_TIMEOUT,
                  `node ${i + 1}/${nodes.length}`,
                )
              } catch (e) {
                console.warn(`[tikz] skipping node ${i + 1}: ${e}`)
                continue
              }
            }
            const node = parent.children[index] as Code

            parent.children.splice(index, 1, {
              type: 'html',
              value: toHtml(makeTikzGraph(node, svg, parseStyle(node?.meta)), {
                allowDangerousHtml: true,
              }),
            })
          }
        },
      ]
    },
    externalResources() {
      return {
        css: [{ content: 'https://cdn.jsdelivr.net/npm/node-tikzjax@latest/css/fonts.css' }],
      }
    },
  }
}
