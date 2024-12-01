import { Code, Root as MdRoot } from "mdast"
import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { load, tex, dvi2svg } from "node-tikzjax"
import { Argv } from "../../util/ctx"

async function tex2svg(input: string, argv: Argv) {
  await load()
  const dvi = await tex(input, {
    showConsole: argv.verbose,
    texPackages: { pgfplots: "", amsmath: "intlimits" },
    tikzLibraries: "arrows.meta,calc",
    addToPreamble: "% comment",
  })
  const svg = await dvi2svg(dvi)
  return svg
}

interface TikzNode {
  index: number
  value: string
  parent: MdRoot
}

function parseStyle(meta: string | null | undefined): string {
  if (!meta) return ""
  const styleMatch = meta.match(/style\s*=\s*["']([^"']+)["']/)
  return styleMatch ? styleMatch[1] : ""
}

const docs = (node: Code): string => JSON.stringify(node.value)

// mainly for reparse from HTML back to MD
const mathMl = (code: Code): string => {
  return `<span class="tikz-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><annotation encoding="application/x-tex">${docs(code)}</annotation></semantics></math></span>`
}

const sourceCodeCopy = (): string => {
  return `<figcaption>
  <em>source code</em>
  <button class="source-code-button" aria-label="copy source code for this tikz graph">
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" class="source-icon" width="12" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 16 4-4-4-4"></path><path d="m6 8-4 4 4 4"></path><path d="m14.5 4-5 16"></path></svg>
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" class="check-icon" width="12" height="16" viewBox="0 0 16 16"  fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path fill-rule="evenodd" fill="rgb(63, 185, 80)" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
  </button>
</figcaption>`
}

export const TikzJax: QuartzTransformerPlugin = () => {
  return {
    name: "TikzJax",
    markdownPlugins({ argv }) {
      return [
        () => async (tree: MdRoot, _file) => {
          const nodes: TikzNode[] = []
          visit(tree, "code", (node: Code, index, parent) => {
            const { lang, meta, value } = node
            if (lang === "tikz") {
              const base64Match = meta?.match(/alt\s*=\s*"data:image\/svg\+xml;base64,([^"]+)"/)
              if (base64Match) {
                const svgContent = Buffer.from(base64Match[1], "base64").toString()
                const style = parseStyle(meta)
                parent!.children.splice(index!, 1, {
                  type: "html",
                  value: `<figure class="tikz"${style ? ` style="${style}"` : ""} data-remark-tikz>${mathMl(node)}${svgContent}${sourceCodeCopy()}</figure>`,
                })
                return
              } else {
                nodes.push({ index: index as number, parent: parent as MdRoot, value })
              }
            }
          })

          for (let i = 0; i < nodes.length; i++) {
            const { index, parent, value } = nodes[i]
            const svg = await tex2svg(value, argv)
            const node = parent.children[index] as Code
            const style = parseStyle(node?.meta)

            parent.children.splice(index, 1, {
              type: "html",
              value: `<figure class="tikz"${style ? ` style="${style}"` : ""}>${mathMl(node)}${svg}${sourceCodeCopy()}</figure>`,
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
