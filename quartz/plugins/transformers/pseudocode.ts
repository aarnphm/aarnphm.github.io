import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { JSResource } from "../../util/resources"

export const Pseudocode: QuartzTransformerPlugin = () => ({
  name: "Pseudocode",
  markdownPlugins() {
    return [
      () => (tree: Root, _file) => {
        visit(tree, "code", (node) => {
          if (node.lang === "pseudo") {
            node.type = "html" as "code"
            node.value = `<pre class="pseudocode latex-pseudo" data-line-number=true>${node.value}</pre>`
          }
        })
      },
    ]
  },
  externalResources() {
    return {
      css: ["https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min.css"],
      js: [
        {
          src: "https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min.js",
          loadTime: "afterDOMReady",
          contentType: "external",
        },
        {
          src: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",
          loadTime: "afterDOMReady",
          contentType: "external",
        },
        {
          script: `
          document.addEventListener('nav', async () => {
            const algos = document.querySelectorAll('.latex-pseudo')
            if (algos.length > 0) {
              pseudocode.renderClass('latex-pseudo', {lineNumbers: true});
            }
          })
          `,
          loadTime: "afterDOMReady",
          contentType: "inline",
          moduleType: "module",
        },
      ],
    }
  },
})
