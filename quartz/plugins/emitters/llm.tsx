import { visit } from "unist-util-visit"
import { Root, Element, Node, Text } from "hast"
import { Blockquote, Code } from "mdast"
import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import BodyConstructor from "../../components/Body"
import { Landing as LandingConstructor } from "../../components"
import { pageResources, transcludeFinal } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { clone, FilePath, pathToRoot } from "../../util/path"
import { write } from "./helpers"
import { toMdast, defaultHandlers as hastToMdastHandlers } from "hast-util-to-mdast"
import { toMarkdown, defaultHandlers as mdastToTextHandlers } from "mdast-util-to-markdown"
import { gfmToMarkdown } from "mdast-util-gfm"
import { InlineMath, Math, mathToMarkdown } from "mdast-util-math"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import DepGraph from "../../depgraph"

export const LLMText: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, left, right, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()
  const Landing = LandingConstructor()

  return {
    name: "LLMText",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        Landing,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async getDependencyGraph() {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const cfg = ctx.cfg.configuration
      const fps: Promise<FilePath>[] = []
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!

        const externalResources = pageResources(pathToRoot(slug), resources)
        const componentData: QuartzComponentProps = {
          ctx,
          fileData: file.data,
          externalResources,
          cfg,
          children: [],
          tree,
          allFiles,
        }

        const root = transcludeFinal(clone(tree) as Root, componentData, { dynalist: false })
        const mdast = toMdast(root, {
          handlers: {
            // handle ast parsed by rehype-pretty-code
            figure(h, node) {
              if (node.properties?.dataRehypePrettyCodeFigure !== "")
                return hastToMdastHandlers.figure(h, node)

              let pre: Element | undefined
              let code: Element | undefined
              let figcaption: Element | undefined

              visit(node, "element", (el: Element) => {
                if (
                  el.tagName === "figcaption" &&
                  el.properties?.dataRehypePrettyCodeTitle === ""
                ) {
                  figcaption = el
                  return false
                }
              })
              visit(node, "element", (el: Element) => {
                if (el.tagName === "pre") {
                  pre = el
                  return false
                }
              })
              // Find pre, code, and figcaption elements
              visit(pre as Node, "element", (el: Element) => {
                if (el.tagName === "code") {
                  code = el
                  return false
                }
              })

              if (!code || !pre) return hastToMdastHandlers.figure(h, node)

              // Get language
              const lang = pre.properties?.dataLanguage

              // Get title from figcaption
              let title = ""
              if (figcaption) {
                title = (figcaption.children[0] as Text)?.value
              }

              // Get highlighted lines
              // FIX: CORRECT THE CHAIN, not work very well for now
              const highlightedLines: number[] = []
              // Get highlighted words
              const highlightedWords: string[] = []
              for (const [i, span] of code.children.entries()) {
                if ((span as Element).properties?.dataHighlightedLine == "") {
                  highlightedLines.push(i)
                }

                // FIX: THIS ALSO DOESN'T WORK YET
                visit(span, "element", (el: Element) => {
                  if (el.tagName === "mark" && el.properties?.dataHighlightedCharsMark) {
                    let word = ""
                    el.children.map((span) => {
                      word += ((span as Element).children[0] as Text)?.value
                    })
                    highlightedWords.push(word)
                  }
                })
              }

              // Build code content from spans
              let codeContent = ""
              visit(code, "element", (span: Element) => {
                if (span.properties?.dataLine !== undefined) {
                  visit(span, "text", (text: Text) => {
                    codeContent += text.value
                  })
                  codeContent += "\n"
                }
              })

              // Build meta string
              const meta = [
                title ? `title="${title}"` : "",
                highlightedLines.length ? `{${highlightedLines.join(",")}}` : "",
                highlightedWords.length ? `/${highlightedWords.join("/")}/` : "",
              ]
                .filter(Boolean)
                .join(" ")

              const result: Code = {
                type: "code",
                lang: (lang as string | null) ?? null,
                meta: meta || null,
                value: codeContent.trimEnd(),
              }

              h.patch(node, result)
              return result
            },
            // handle math node correctly
            span(h, node) {
              const classNames = (node.properties.className ?? []) as string[]
              // katex: inline-math, katex-display: block-math
              if (classNames.includes("katex") || classNames.includes("katex-display")) {
                const inline = !classNames.includes("katex-display")
                let source: string | null = null

                visit(node, "element", (node) => {
                  if (
                    node.tagName === "annotation" &&
                    node.properties?.encoding === "application/x-tex"
                  ) {
                    if (node.children?.[0]?.type === "text") {
                      source = node.children[0].value
                      return false // stop traversal
                    }
                  }
                })
                if (!source) {
                  console.warn(
                    `[emit:ContentPage] Could not extract LaTeX source from KaTeX node (slug: ${slug})`,
                  )
                  return hastToMdastHandlers.span(h, node)
                }

                const results: Math | InlineMath = {
                  type: inline ? "inlineMath" : "math",
                  value: source,
                }
                h.patch(node, results)
                return results
              } else {
                return hastToMdastHandlers.span(h, node)
              }
            },
            // handle mermaid
            pre(h, node) {
              let codeEl: Element | undefined
              visit(node, "element", (el) => {
                if (
                  el.tagName === "code" &&
                  ((el.properties?.className ?? []) as string[]).includes("mermaid")
                ) {
                  codeEl = el
                  return false
                }
              })
              if (!codeEl) return hastToMdastHandlers.pre(h, node)
              const results: Code = {
                type: "code",
                lang: "mermaid",
                value: JSON.parse(codeEl.properties?.dataClipboard as string),
              }
              h.patch(node, results)
              return results
            },
            // handle callout correctly
            blockquote(h, node) {
              const classNames = (node.properties?.className ?? []) as string[]
              if (!classNames.includes("callout")) {
                return hastToMdastHandlers.blockquote(h, node)
              }

              // Get callout type
              const type = node.properties?.dataCallout as string

              // Get title from callout-title-inner
              let title = ""
              let titleNode: Element | undefined
              visit(node, "element", (el: Element) => {
                if ((el.properties?.className as string[])?.includes("callout-title-inner")) {
                  titleNode = el
                  return false
                }
              })
              if (titleNode) {
                title = ((titleNode.children[0] as Element)?.children[0] as Text)?.value
              }

              // Check collapse state
              const isCollapsible = classNames.includes("is-collapsible")
              const isCollapsed = classNames.includes("is-collapsed")
              const collapseChar = isCollapsible ? (isCollapsed ? "-" : "+") : ""

              // Get remaining content
              let content: any[] = []
              visit(node, "element", (el: Element) => {
                if ((el.properties?.className as string[])?.includes("callout-content")) {
                  // Convert children using default blockquote handler to maintain parsing
                  content = h.all(el)
                  return false
                }
              })

              const result: Blockquote = {
                type: "blockquote",
                children: [
                  {
                    type: "paragraph",
                    children: [
                      {
                        type: "text",
                        value: `[!${type}]${collapseChar}${title ? ` ${title.trim()}` : ""}`,
                        data: { unescaped: true },
                      },
                    ],
                  },
                  ...content,
                ],
              }

              h.patch(node, result)
              return result
            },
          },
        })
        const fp = write({
          ctx,
          content: toMarkdown(mdast, {
            extensions: [
              {
                handlers: {
                  code(node, _parent, _context, _info) {
                    const { lang, meta, value } = node
                    const info = [lang, meta].filter(Boolean).join(" ")
                    return "```" + (info ? info + "\n" : "\n") + value + "\n```"
                  },
                  text(node, parent, context, info) {
                    if (node.data?.unescaped) {
                      return node.value
                    }
                    return mdastToTextHandlers.text(node, parent, context, info)
                  },
                },
              },
              mathToMarkdown(),
              gfmToMarkdown(),
            ],
          }),
          slug,
          ext: ".html.md",
        })
        fps.push(fp)
      }

      return await Promise.all(fps)
    },
  }
}
