import { Code, Root as MdRoot } from "mdast"
import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { load, tex, dvi2svg } from "node-tikzjax"
import { Argv } from "../../util/ctx"

async function tex2svg(input: string, argv: Argv) {
  await load()
  const dvi = await tex(input, { showConsole: argv.verbose })
  const svg = await dvi2svg(dvi)
  return svg
}

interface TikzNode {
  index: number
  value: string
  parent: MdRoot
}

export const TikzJax: QuartzTransformerPlugin = () => {
  return {
    name: "TikzJax",
    markdownPlugins({ argv }) {
      return [
        () => async (tree: MdRoot, _file) => {
          const nodes: TikzNode[] = []
          visit(tree, "code", (node: Code, index, parent) => {
            if (node.lang === "tikz") {
              nodes.push({ index: index as number, parent: parent as MdRoot, value: node.value })
            }
          })

          for (let i = 0; i < nodes.length; i++) {
            const { index, parent, value } = nodes[i]
            const svg = await tex2svg(value, argv)

            parent.children.splice(index, 1, {
              type: "html",
              value: `<div class="tikz">${svg}</div>`,
            })
          }
        },
      ]
    },
    externalResources() {
      return {
        css: [
          {
            content: "https://cdn.jsdelivr.net/npm/node-tikzjax@latest/css/fonts.css",
          },
        ],
      }
    },
  }
}
