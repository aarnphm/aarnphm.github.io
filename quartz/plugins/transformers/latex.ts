import { KatexOptions } from 'katex'
import { Root } from 'mdast'
import { Fragment, h } from 'preact'
import remarkMath from 'remark-math'
import { visit, EXIT } from 'unist-util-visit'
import { VFile } from 'vfile'
import { QuartzTransformerPlugin } from '../../types/plugin'
import cachedKatex from '../../util/cached-katex'
import { QuartzPluginData } from '../vfile'

const katexDist = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist'
// The faces behind most formulas: text, math italic, bold vectors (\mathbf) and
// blackboard sets (\mathbb). The stylesheet is cross-origin, so its font requests
// otherwise start only after it is fetched and parsed, and math paints late.
const katexPreloadFonts = [
  'KaTeX_Main-Regular',
  'KaTeX_Math-Italic',
  'KaTeX_Main-Bold',
  'KaTeX_AMS-Regular',
]

// mark pages with math, so only they preload KaTeX's fonts
const markMath = () => (tree: Root, file: VFile) => {
  visit(tree, ['math', 'inlineMath'], () => {
    file.data.hasMath = true
    return EXIT
  })
}

interface Options {
  renderEngine: 'katex'
  customMacros: MacroType
  katexOptions: Omit<KatexOptions, 'macros' | 'output'>
}

interface MacroType {
  [key: string]: string
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = opts => {
  const engine = opts?.renderEngine ?? 'katex'
  const macros = opts?.customMacros ?? {}
  return {
    name: 'Latex',
    markdownPlugins: () => [remarkMath, markMath],
    htmlPlugins() {
      switch (engine) {
        default: {
          return [[cachedKatex, { output: 'htmlAndMathml', macros, ...opts?.katexOptions }]]
        }
      }
    },
    externalResources() {
      switch (engine) {
        case 'katex':
          return {
            css: [{ content: `${katexDist}/katex.min.css` }],
            additionalHead: katexPreloadFonts.map(
              name => (fileData: QuartzPluginData) =>
                h(
                  Fragment,
                  null,
                  fileData.hasMath
                    ? h('link', {
                        rel: 'preload',
                        as: 'font',
                        type: 'font/woff2',
                        href: `${katexDist}/fonts/${name}.woff2`,
                        crossOrigin: 'anonymous',
                      })
                    : null,
                ),
            ),
            js: [
              {
                // fix copy behaviour: https://github.com/KaTeX/KaTeX/blob/main/contrib/copy-tex/README.md
                src: `${katexDist}/contrib/copy-tex.min.js`,
                loadTime: 'afterDOMReady',
                contentType: 'external',
              },
            ],
          }
      }
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    hasMath: boolean
  }
}
