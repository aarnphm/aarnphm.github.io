import { visit } from "unist-util-visit"
import { Element, Node, Text, Root as hastRoot } from "hast"
import { Blockquote, Code, DefinitionContent, Paragraph, Root } from "mdast"
import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import BodyConstructor from "../../components/Body"
import { Landing as LandingConstructor } from "../../components"
import { pageResources, transcludeFinal, headerRegex } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { clone, FilePath, pathToRoot } from "../../util/path"
import { write } from "./helpers"
import { toMdast, defaultHandlers as hastToMdastHandlers, State } from "hast-util-to-mdast"
import { toMarkdown, defaultHandlers as mdastToTextHandlers } from "mdast-util-to-markdown"
import { gfmToMarkdown } from "mdast-util-gfm"
import { InlineMath, Math, mathToMarkdown } from "mdast-util-math"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import DepGraph from "../../depgraph"
import { Heading } from "mdast"
import { FootnoteDefinition } from "mdast"

const heading = (h: State, node: Element): Heading => {
  // NOTE: for all heading, we append the links in hast syntax tree. For markdown, we don't need to do this.
  node.children.pop()
  switch (node.tagName.match(headerRegex)![0]) {
    case "h1":
      return hastToMdastHandlers.h1(h, node)
    case "h2":
      return hastToMdastHandlers.h2(h, node)
    case "h3":
      return hastToMdastHandlers.h3(h, node)
    case "h4":
      return hastToMdastHandlers.h4(h, node)
    case "h5":
      return hastToMdastHandlers.h5(h, node)
    case "h6":
      return hastToMdastHandlers.h6(h, node)
    default:
      throw new Error("Failed to parse correct headers")
  }
}

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

        const root = transcludeFinal(clone(tree) as hastRoot, componentData, { dynalist: false })
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
            h2: heading,
            h3: heading,
            h4: heading,
            h5: heading,
            h6: heading,
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
              if (classNames.includes("callout")) {
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
              } else if (classNames.includes("transclude")) {
                // we will also flatten transclude
                const unfold: Element = {
                  type: "element",
                  tagName: "div",
                  properties: node.properties,
                  children: node.children,
                }
                const result = hastToMdastHandlers.div(h, unfold)
                h.patch(node, result)
                return result
              }
              return hastToMdastHandlers.blockquote(h, node)
            },
            a(h, node) {
              if (node.properties.dataFootnoteRef === "") {
                const identifier = (node.properties?.id as string).replace(
                  "user-content-fnref-",
                  "",
                )
                const result: Paragraph = {
                  type: "paragraph",
                  children: [{ type: "footnoteReference", identifier, label: identifier }],
                }
                h.patch(node, result)
                return result
              } else if (node.properties.dataFootnoteBackref === "") {
                // FIXME: Right now, we patch the backref with a empty string, probably not good, should just remove it.
                const result: Paragraph = {
                  type: "paragraph",
                  children: [{ type: "text", value: "" }],
                }
                h.patch(node, result)
                return result
              }
              return hastToMdastHandlers.a(h, node)
            },
            // handle footnotes correctly
            section(h, node) {
              if (node.properties.dataFootnotes == "") {
                const ol = node.children.pop() as Element
                const defs: FootnoteDefinition[] = []
                for (const li of ol.children as Element[]) {
                  const identifier = (li.properties?.id as string).replace("user-content-fn-", "")
                  const children = h.all(li) as DefinitionContent[]
                  defs.push({
                    type: "footnoteDefinition",
                    identifier,
                    label: identifier,
                    children,
                  })
                }
                const results: Root = {
                  type: "root",
                  children: defs,
                }
                h.patch(node, results)
                return results
              }
              return hastToMdastHandlers.section(h, node)
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
