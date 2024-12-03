import remarkGfm from "remark-gfm"
import smartypants from "remark-smartypants"
import { QuartzTransformerPlugin } from "../types"
import rehypeSlug from "rehype-slug"
import rehypeAutolinkHeadings from "rehype-autolink-headings"
import { visit } from "unist-util-visit"
import { headingRank } from "hast-util-heading-rank"
import { h, s } from "hastscript"

export interface Options {
  enableSmartyPants: boolean
  linkHeadings: boolean
}

const defaultOptions: Options = {
  enableSmartyPants: true,
  linkHeadings: true,
}

export const GitHubFlavoredMarkdown: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "GitHubFlavoredMarkdown",
    markdownPlugins() {
      return opts.enableSmartyPants ? [remarkGfm, smartypants] : [remarkGfm]
    },
    htmlPlugins() {
      if (opts.linkHeadings) {
        return [
          rehypeSlug,
          () => {
            return (tree, _file) => {
              visit(tree, "element", function (node) {
                if (headingRank(node)) {
                  if (node.properties.id === "footnote-label") {
                    node.children = [{ type: "text", value: "Remarque" }]
                  }
                  node.children = [h("span.highlight-span", node.children)]
                }
              })
            }
          },
          [
            rehypeAutolinkHeadings,
            {
              behavior: "append",
              properties: {
                role: "anchor",
                ariaHidden: true,
                tabIndex: -1,
                "data-no-popover": true,
              },
              content: s(
                "svg",
                {
                  width: 16,
                  height: 16,
                  viewbox: "0 0 24 24",
                  fill: "none",
                  stroke: "currentColor",
                  strokewidth: "2",
                  strokelinecap: "round",
                  strokelinejoin: "round",
                },
                [
                  s("path", { d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" }),
                  s("path", { d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" }),
                ],
              ),
            },
          ],
        ]
      } else {
        return []
      }
    },
  }
}
