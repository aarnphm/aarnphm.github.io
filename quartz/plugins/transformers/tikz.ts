import { Code, Root as MdRoot } from "mdast"
import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { load, tex, dvi2svg } from "node-tikzjax"

async function tex2svg(input: string) {
  await load()
  const dvi = await tex(input, {
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
  <button class="source-code-button" aria-label="copy source code for this tikz graph" title="copy source code for this tikz graph">
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" class="source-icon" width="12" height="16" viewBox="0 -4 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><use href="#code-icon"></use></svg>
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" class="check-icon" width="12" height="16" viewBox="0 -4 16 16"><use href="#github-check"></use></svg>
  </button>
</figcaption>`
}

export const TikzJax: QuartzTransformerPlugin = () => {
  return {
    name: "TikzJax",
    markdownPlugins(ctx) {
      // We skip tikz transpilation for now during process (takes too long for a file with a lot of tikz graph)
      // TODO: maybe we should render client-side instead of server-side? (build-time would increase)
      if (ctx.argv.serve) {
        return []
      }

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
              } else {
                nodes.push({ index: index as number, parent: parent as MdRoot, value })
              }
            }
          })

          for (let i = 0; i < nodes.length; i++) {
            const { index, parent, value } = nodes[i]
            const svg = await tex2svg(value)
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
