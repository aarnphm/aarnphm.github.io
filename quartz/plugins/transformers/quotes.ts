import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const Quotes: QuartzTransformerPlugin = () => ({
  name: "Quotes",
  markdownPlugins() {
    return [
      () => (tree: Root, _file) => {
        visit(tree, "code", (node) => {
          if (node.lang === "quotes") {
            node.type = "html" as "code"
            node.value = `<p class="quotes">${node.value}</p>`
          }
        })
      },
    ]
  },
})
