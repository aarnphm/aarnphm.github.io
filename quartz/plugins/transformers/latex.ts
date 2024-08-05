import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import { QuartzTransformerPlugin } from "../types"

export const Latex: QuartzTransformerPlugin = () => {
  return {
    name: "Latex",
    markdownPlugins() {
      return [remarkMath]
    },
    htmlPlugins() {
      return [[rehypeKatex, { output: "html", strict: "error" }]]
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
