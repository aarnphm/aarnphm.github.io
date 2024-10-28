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
  const engine = opts?.renderEngine ?? "katex"
  const macros = opts?.customMacros ?? {}
  const katexOptions = opts?.katexOptions ?? {}
  return {
    name: "Latex",
    markdownPlugins() {
      return [remarkMath]
    },
    htmlPlugins() {
      if (engine === "katex") {
        return [[rehypeKatex, { output: "html", macros, ...katexOptions }]]
      }
      return []
    },
    externalResources() {
      return {
        css: [
          // base css
          "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css",
        ],
        js: [
          {
            // fix copy behaviour: https://github.com/KaTeX/KaTeX/blob/main/contrib/copy-tex/README.md
            src: "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/copy-tex.min.js",
            loadTime: "afterDOMReady",
            contentType: "external",
          },
        ],
      }
    },
  }
}
