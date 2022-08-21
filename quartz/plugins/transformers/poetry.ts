import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const Poetry: QuartzTransformerPlugin = () => ({
  name: "Poetry",
  markdownPlugins(ctx) {
    return [
      () => (tree: Root, file) => {
        const cfg = ctx.cfg.configuration
        const lang = file.data.frontmatter?.lang ?? cfg.locale.split("-")[0]
        visit(tree, "code", (node) => {
          if (node.lang === "poetry") {
            node.type = "html" as "code"
            node.value = `<pre class="poetry" data-language="${lang}">${node.value}</pre>`
          }
        })
      },
    ]
  },
})
