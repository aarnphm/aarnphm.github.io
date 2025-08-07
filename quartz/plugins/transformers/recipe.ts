import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const Recipe: QuartzTransformerPlugin = () => ({
  name: "Recipe",
  markdownPlugins() {
    return [
      () => (tree: Root, _file) => {
        visit(tree, "code", (node) => {
          if (node.lang === "recipe") {
            node.type = "html" as "code"
            node.value = `<pre class="" data-line-number=true>${node.value}</pre>`
          }
        })
      },
    ]
  },
})
