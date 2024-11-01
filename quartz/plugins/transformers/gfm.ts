import remarkGfm from "remark-gfm"
import smartypants from "remark-smartypants"
import { QuartzTransformerPlugin } from "../types"
import { Element, Root as HtmlRoot } from "hast"
import rehypeSlug from "rehype-slug"
import rehypeAutolinkHeadings from "rehype-autolink-headings"
import { visit } from "unist-util-visit"
import { PluggableList } from "unified"
import { CSSResource, JSResource } from "../../util/resources"
// @ts-ignore
import collapseHeaderScript from "../../components/scripts/collapse-header.inline.ts"
import collapseHeaderStyle from "../../components/styles/collapseHeader.inline.scss"

export interface Options {
  enableSmartyPants: boolean
  enableCollapseHeaders: boolean
  linkHeadings: boolean
}

const defaultOptions: Options = {
  enableSmartyPants: true,
  enableCollapseHeaders: true,
  linkHeadings: true,
}

function headerElement(node: Element, content: Element[], idx: number): Element {
  const buttonId = `collapsible-header-${node.properties?.id ?? idx}`
  return {
    type: "element",
    tagName: "div",
    properties: {
      className: ["collapsible-header"],
      "data-level": node.tagName[1],
    },
    children: [
      {
        type: "element",
        tagName: "button",
        properties: {
          id: buttonId,
          ariaLabel: "Toggle content visibility",
          ariaExpanded: true,
          className: ["header-button"],
        },
        children: [
          {
            type: "element",
            tagName: "svg",
            properties: {
              xmlns: "http://www.w3.org/2000/svg",
              width: 18,
              height: 18,
              viewBox: "0 0 24 24",
              fill: "none",
              stroke: "currentColor",
              "stroke-width": "2",
              "stroke-linecap": "round",
              "stroke-linejoin": "round",
              className: ["fold"],
            },
            children: [
              {
                type: "element",
                tagName: "polyline",
                properties: {
                  points: "6 9 12 15 18 9",
                },
                children: [],
              },
            ],
          },
          node,
        ],
      },
      {
        type: "element",
        tagName: "div",
        properties: {
          className: ["collapsible-header-content-outer"],
        },
        children: [
          {
            type: "element",
            tagName: "div",
            properties: {
              className: ["collapsible-header-content"],
              ["data-references"]: buttonId,
            },
            children: content,
          },
        ],
      },
    ],
  }
}

function processHeaders(node: Element, idx: number | undefined, parent: Element) {
  idx = idx ?? parent.children.indexOf(node)
  const currentLevel = parseInt(node.tagName[1])
  const contentNodes: Element[] = []
  let i = idx + 1

  // Collect all content until next header of same or higher level
  while (i < parent.children.length) {
    const nextNode = parent.children[i] as Element
    if (
      (["div"].includes(nextNode.tagName) && nextNode.properties.id == "refs") ||
      (nextNode?.type === "element" &&
        nextNode.tagName?.match(/^h[1-6]$/) &&
        nextNode.properties["data-footnotes"])
    ) {
      break
    }

    if (nextNode?.type === "element" && nextNode.tagName?.match(/^h[1-6]$/)) {
      const nextLevel = parseInt(nextNode.tagName[1])
      if (nextLevel <= currentLevel) {
        break
      }
      // Process nested header recursively
      processHeaders(nextNode, i, parent)

      // After processing, the next node at index i will be the wrapper
      contentNodes.push(parent.children[i] as Element)
      parent.children.splice(i, 1)
    } else {
      contentNodes.push(nextNode)
      parent.children.splice(i, 1)
    }
  }

  parent.children.splice(idx, 1, headerElement(node, contentNodes, idx))
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
        const plugins: PluggableList = [
          rehypeSlug,
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
              content: {
                type: "element",
                tagName: "svg",
                properties: {
                  width: 18,
                  height: 18,
                  viewBox: "0 0 24 24",
                  fill: "none",
                  stroke: "currentColor",
                  "stroke-width": "2",
                  "stroke-linecap": "round",
                  "stroke-linejoin": "round",
                },
                children: [
                  {
                    type: "element",
                    tagName: "path",
                    properties: {
                      d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",
                    },
                    children: [],
                  },
                  {
                    type: "element",
                    tagName: "path",
                    properties: {
                      d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
                    },
                    children: [],
                  },
                ],
              },
            },
          ],
        ]

        if (opts.enableCollapseHeaders) {
          const headingTagTypes = new Set(["h1", "h2", "h3", "h4", "h5", "h6"])
          plugins.push(() => {
            return (tree: HtmlRoot, file) => {
              visit(tree, "element", (node: Element, idx, parent) => {
                if (
                  headingTagTypes.has(node.tagName) &&
                  parent &&
                  node.properties.id !== "footnote-label" &&
                  file.data.slug !== "index"
                ) {
                  // then do the process headers and its children here
                  processHeaders(node, idx, parent as Element)
                }
              })
            }
          })
        }

        return plugins
      } else {
        return []
      }
    },
    externalResources() {
      const js: JSResource[] = []
      const css: CSSResource[] = []

      if (opts.enableCollapseHeaders) {
        js.push({
          script: collapseHeaderScript,
          loadTime: "afterDOMReady",
          contentType: "inline",
        })
        css.push({
          content: collapseHeaderStyle,
          inline: true,
        })
      }

      return { js, css }
    },
  }
}
