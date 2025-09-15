import { Code, Root as MdRoot } from "mdast"
import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { load, tex, dvi2svg } from "node-tikzjax"
import { h, s } from "hastscript"
import { Element, Properties } from "hast"
import { svgOptions } from "../../components/svg"
import { toHtml } from "hast-util-to-html"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"

async function tex2svg(input: string) {
  await load()
  const dvi = await tex(input, {
    texPackages: { pgfplots: "", amsmath: "intlimits" },
    tikzLibraries: "arrows.meta,calc",
    addToPreamble: "% comment",
    showConsole: true,
  })
  const svg = await dvi2svg(dvi)
  return svg
}

interface TikzNode {
  index: number
  value: string
  parent: MdRoot
  base64?: string
}

function parseStyle(meta: string | null | undefined): string {
  if (!meta) return ""
  const styleMatch = meta.match(/style\s*=\s*["']([^"']+)["']/)
  return styleMatch ? styleMatch[1] : ""
}

const docs = (node: Code): string => JSON.stringify(node.value)

// mainly for reparse from HTML back to MD
function makeTikzGraph(node: Code, svg: string, style?: string): Element {
  const mathMl = h(
    "span.tikz-mathml",
    h(
      "math",
      { xmlns: "http://www.w3.org/1998/Math/MathML" },
      h(
        "semantics",
        h("annotation", { encoding: "application/x-tex" }, { type: "text", value: docs(node) }),
      ),
    ),
  )

  const sourceCodeCopy = h(
    "figcaption",
    h("em", [{ type: "text", value: "source code" }]),
    h(
      "button.source-code-button",
      {
        ariaLabel: "copy source code for this tikz graph",
        title: "copy source code for this tikz graph",
      },
      s(
        "svg.source-icon",
        {
          ...svgOptions,
          width: 12,
          height: 16,
          viewbox: "0 -4 24 24",
          fill: "none",
          stroke: "currentColor",
          strokewidth: 2,
        },
        s("use", { href: "#code-icon" }),
      ),
      s(
        "svg.check-icon",
        {
          ...svgOptions,
          width: 12,
          height: 16,
          viewbox: "0 -4 16 16",
        },
        s("use", { href: "#github-check" }),
      ),
    ),
  )

  const properties: Properties = { "data-remark-tikz": true, style: "" }
  if (style) properties.style = style

  return h(
    "figure.tikz",
    properties,
    mathMl,
    fromHtmlIsomorphic(svg, { fragment: true }),
    sourceCodeCopy,
  )
}

export const TikzJax: QuartzTransformerPlugin = () => {
  return {
    name: "TikzJax",
    // TODO: maybe we should render client-side instead of server-side? (build-time would increase).
    // We skip tikz transpilation for now during process (takes too long for a file with a lot of tikz graph)
    markdownPlugins() {
      return [
        () => async (tree) => {
          const nodes: TikzNode[] = []
          visit(tree, "code", (node: Code, index, parent) => {
            let { lang, meta, value } = node
            if (lang === "tikz") {
              const base64Match = meta?.match(/alt\s*=\s*"data:image\/svg\+xml;base64,([^"]+)"/)
              let base64String = undefined
              if (base64Match) {
                base64String = Buffer.from(base64Match[1], "base64").toString()
              }
              nodes.push({
                index: index as number,
                parent: parent as MdRoot,
                value,
                base64: base64String,
              })
            }
          })

          for (let i = 0; i < nodes.length; i++) {
            const { index, parent, value, base64 } = nodes[i]
            let svg
            if (base64 !== undefined) svg = base64
            else svg = await tex2svg(value)
            const node = parent.children[index] as Code

            parent.children.splice(index, 1, {
              type: "html",
              value: toHtml(makeTikzGraph(node, svg, parseStyle(node?.meta)), {
                allowDangerousHtml: true,
              }),
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
