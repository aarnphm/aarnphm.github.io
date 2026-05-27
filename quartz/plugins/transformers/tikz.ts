import { Element, Properties } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import { Code, Root as MdRoot } from 'mdast'
import { load, tex, dvi2svg } from 'node-tikzjax'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { QUARTZ } from '../../util/path'
import { svgOptions } from '../../util/svg-options'

const TIKZ_TIMEOUT = 30_000
const TIKZ_CACHE_VERSION = 1
const TIKZ_CACHE_DIR = path.join(QUARTZ, '.quartz-cache', 'tikz')
const texPackages = { pgfplots: '', amsmath: 'intlimits' }
const tikzLibraries = 'arrows.meta,calc,positioning'
const addToPreamble = '% comment'

const cmMathItalicGlyphs = new Map([
  ['£', 'Γ'],
  ['¤', 'Δ'],
  ['¥', 'Θ'],
  ['¦', 'Λ'],
  ['§', 'Ξ'],
  ['¨', 'Π'],
  ['©', 'Σ'],
  ['ª', 'Υ'],
  ['«', 'Φ'],
  ['¬', 'Ψ'],
  ['®', 'α'],
  ['¯', 'β'],
  ['°', 'γ'],
  ['±', 'δ'],
  ['²', 'ε'],
  ['³', 'ζ'],
  ['´', 'η'],
  ['µ', 'θ'],
  ['¶', 'ι'],
  ['·', 'κ'],
  ['¸', 'λ'],
  ['¹', 'μ'],
  ['º', 'ν'],
  ['»', 'ξ'],
  ['¼', 'π'],
  ['½', 'ρ'],
  ['¾', 'σ'],
  ['¿', 'τ'],
  ['À', 'υ'],
  ['Á', 'φ'],
  ['Â', 'χ'],
  ['Ã', 'ψ'],
  ['Ä', 'ω'],
  ["'", 'φ'],
])

const cmSymbolGlyphs = new Map([
  ['¡', '−'],
  ['¢', '·'],
  ['£', '×'],
  ['¤', '∗'],
  ['¥', '÷'],
  ['¦', '⋄'],
  ['§', '±'],
  ['¨', '∓'],
  ['µ', '≤'],
  ['¶', '≥'],
  ['¹', '∼'],
  ['º', '≈'],
  ['¼', '⊃'],
])

const cmRomanGlyphs = new Map([
  ['£', 'Γ'],
  ['¤', 'Δ'],
  ['¥', 'Θ'],
  ['¦', 'Λ'],
  ['§', 'Ξ'],
  ['¨', 'Π'],
  ['©', 'Σ'],
  ['ª', 'Υ'],
  ['«', 'Φ'],
  ['¬', 'Ψ'],
  ['®', 'ff'],
  ['¯', 'fi'],
  ['°', 'fl'],
  ['±', 'ffi'],
  ['²', 'ffl'],
])

export interface TikzRenderOptions {
  showConsole: boolean
  disableSanitize: boolean
  disableOptimize: boolean
}

type TikzRenderer = (input: string, opts: TikzRenderOptions) => Promise<string>

function tikzCacheKey(input: string, opts: TikzRenderOptions): string {
  return createHash('sha256')
    .update(
      JSON.stringify({
        version: TIKZ_CACHE_VERSION,
        input,
        texPackages,
        tikzLibraries,
        addToPreamble,
        opts,
      }),
    )
    .digest('hex')
}

function isMissingFile(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT'
}

async function readTikzCache(cacheDir: string, key: string): Promise<string | undefined> {
  try {
    const cached = await readFile(path.join(cacheDir, `${key}.svg`), 'utf8')
    return cached.length > 0 ? cached : undefined
  } catch (error) {
    if (isMissingFile(error)) return undefined
    throw error
  }
}

async function writeTikzCache(cacheDir: string, key: string, svg: string): Promise<void> {
  await mkdir(cacheDir, { recursive: true })
  const target = path.join(cacheDir, `${key}.svg`)
  const temp = path.join(cacheDir, `${key}.${randomUUID()}.tmp`)
  await writeFile(temp, svg)
  try {
    await rename(temp, target)
  } catch (error) {
    await rm(temp, { force: true })
    throw error
  }
}

export async function cachedTikzSvg(
  input: string,
  opts: TikzRenderOptions,
  render: TikzRenderer = tex2svg,
  cacheDir: string = TIKZ_CACHE_DIR,
): Promise<string> {
  const key = tikzCacheKey(input, opts)
  const cached = await readTikzCache(cacheDir, key)
  if (cached !== undefined) return cached

  const svg = await render(input, opts)
  await writeTikzCache(cacheDir, key, svg)
  return svg
}

async function tex2svg(input: string, opts: TikzRenderOptions) {
  await load()
  const dvi = await tex(input, {
    texPackages,
    tikzLibraries,
    addToPreamble,
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

function decodeNumericEntities(value: string): string {
  return value.replace(/&#(x[\da-f]+|\d+);/gi, (_, code: string) => {
    const radix = code.startsWith('x') || code.startsWith('X') ? 16 : 10
    return String.fromCodePoint(Number.parseInt(code.replace(/^x/i, ''), radix))
  })
}

function escapeSvgText(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function normalizeComputerModernText(svg: string): string {
  return svg.replace(/<text\b([^>]*)>([\s\S]*?)<\/text>/g, (match, attrs: string, text: string) => {
    const font = attrs.match(/\bfont-family="([^"]+)"/)?.[1]
    if (!font) return match

    const decoded = decodeNumericEntities(text)
    let glyphs: Map<string, string> | undefined
    if (font.startsWith('cmmi') || font.startsWith('cmmib')) glyphs = cmMathItalicGlyphs
    else if (font.startsWith('cmsy') || font.startsWith('cmbsy')) glyphs = cmSymbolGlyphs
    else if (/^cm(?:r|bx|ti|sl|csc|ss)\d/.test(font)) glyphs = cmRomanGlyphs

    const normalized = [...decoded].map(char => glyphs?.get(char) ?? char).join('')
    let nextAttrs = attrs
    if (/^(?:cm|eu|ms)[a-z]+\d+$/.test(font)) {
      nextAttrs = nextAttrs.replace(/\bfont-family="[^"]+"/, 'font-family="serif"')
    }
    if ((font.startsWith('cmmi') || font.startsWith('cmmib')) && !/\bfont-style=/.test(nextAttrs)) {
      nextAttrs += ' font-style="italic"'
    }

    return `<text${nextAttrs}>${escapeSvgText(normalized)}</text>`
  })
}

function closeDanglingSvgTags(svg: string): string {
  const stack: string[] = []
  const tags = svg.matchAll(/<\/?([A-Za-z][\w:.-]*)(?:\s[^<>]*)?>/g)

  for (const tag of tags) {
    const source = tag[0]
    const name = tag[1]
    if (source.startsWith('</')) {
      for (let i = stack.length - 1; i >= 0; i--) {
        const open = stack.pop()
        if (open === name) break
      }
    } else if (!source.endsWith('/>')) {
      stack.push(name)
    }
  }

  return `${svg}${stack
    .reverse()
    .map(name => `</${name}>`)
    .join('')}`
}

function prepareSvgForImage(svg: string): string {
  return normalizeComputerModernText(closeDanglingSvgTags(svg))
}

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

  const encoded = Buffer.from(prepareSvgForImage(svg)).toString('base64')
  const imgNode = h('img', {
    src: `data:image/svg+xml;base64,${encoded}`,
    alt: 'tikz diagram',
    loading: 'lazy',
    decoding: 'async',
  })

  return h('figure.tikz', properties, mathMl, imgNode, sourceCodeCopy)
}

interface Options {
  showConsole?: boolean
  disableOptimize?: boolean
}

const defaultOpts: Required<Options> = { showConsole: false, disableOptimize: true }

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
                  cachedTikzSvg(value, { disableSanitize, ...o }),
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
