import remarkGfm from "remark-gfm"
import smartypants from "remark-smartypants"
import { QuartzTransformerPlugin } from "../types"
import rehypeSlug from "rehype-slug"
import rehypeAutolinkHeadings from "rehype-autolink-headings"
import { visit } from "unist-util-visit"
import { headingRank } from "hast-util-heading-rank"
import { h, s } from "hastscript"
import { PluggableList } from "unified"
import { Element } from "hast"
import { svgOptions } from "../../components/svg"

export interface Options {
  linkHeadings: boolean
}

const defaultOptions: Options = {
  linkHeadings: true,
}

export const GitHubFlavoredMarkdown: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "GitHubFlavoredMarkdown",
    markdownPlugins: () => [remarkGfm, smartypants],
    htmlPlugins() {
      const plugins: PluggableList = []

      if (opts.linkHeadings) {
        plugins.push(rehypeSlug, () => {
          const checkHeading = (node: Element) => headingRank(node) !== undefined
          return (tree, _) => {
            visit(
              tree,
              (node) => checkHeading(node as Element),
              (node) => {
                if (node.properties.id === "footnote-label") {
                  node.children = [{ type: "text", value: "Remarque" }]
                }
                node.children = [h("span.highlight-span", node.children)]
              },
            )
            visit(tree, { tagName: "section" }, (node) => {
              if (node.properties.dataFootnotes == "") {
                const className = Array.isArray(node.properties.className)
                  ? node.properties.className
                  : (node.properties.className = [])
                className.push("popover-hint")
              }
            })
          }
        }, [
          rehypeAutolinkHeadings,
          {
            behavior: "append",
            properties: {
              "data-role": "anchor",
              "data-no-popover": true,
            },
            content: s(
              "svg",
              { ...svgOptions, fill: "none", stroke: "currentColor", strokewidth: "2" },
              [s("use", { href: "#github-anchor" })],
            ),
          },
        ])
      }

      return plugins
    },
  }
}
