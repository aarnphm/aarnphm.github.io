import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
//@ts-ignore
import rehypeTypst from "@myriaddreamin/rehype-typst"
import { QuartzTransformerPlugin } from "../../types/plugin"
import { KatexOptions } from "katex"
//@ts-ignore
import { Options as TypstOptions } from "@myriaddreamin/rehype-typst"

interface Options {
  renderEngine: "katex" | "typst"
  customMacros: MacroType
  katexOptions: Omit<KatexOptions, "macros" | "output">
  typstOptions: TypstOptions
}

interface MacroType {
  [key: string]: string
}

export const Latex: QuartzTransformerPlugin<Partial<Options>> = (opts) => {
  const engine = opts?.renderEngine ?? "katex"
  const macros = opts?.customMacros ?? {}
  return {
    name: "Latex",
    markdownPlugins: () => [remarkMath],
    htmlPlugins() {
      switch (engine) {
        case "typst": {
          return [[rehypeTypst, opts?.typstOptions ?? {}]]
        }
        default: {
          return [[rehypeKatex, { output: "htmlAndMathml", macros, ...opts?.katexOptions }]]
        }
      }
    },
    externalResources() {
      switch (engine) {
        case "katex":
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
      }
    },
  }
}
