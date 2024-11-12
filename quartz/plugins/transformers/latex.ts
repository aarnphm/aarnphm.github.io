import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import { QuartzTransformerPlugin } from "../types"
import { KatexOptions as KOptions } from "katex"

type KatexOptions = Omit<KOptions, "macros" | "output" | "displayMode" | "throwOnError">

interface Options {
  renderEngine: "katex" | "mathjax"
  customMacros: MacroType
  katexOptions: KatexOptions
}

interface MacroType {
  [key: string]: string
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = (opts) => {
  const macros = opts?.customMacros ?? {}
  return {
    name: "Latex",
    markdownPlugins() {
      return [remarkMath]
    },
    htmlPlugins() {
      return [[rehypeKatex, { output: "htmlAndMathml", macros, ...(opts?.katexOptions ?? {}) }]]
    },
    externalResources() {
      return {
        css: [{ content: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" }],
        js: [
          {
            // fix copy behaviour: https://github.com/KaTeX/KaTeX/blob/main/contrib/copy-tex/README.md
            src: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js",
            loadTime: "afterDOMReady",
            contentType: "external",
          },
        ],
      }
    },
  }
}
