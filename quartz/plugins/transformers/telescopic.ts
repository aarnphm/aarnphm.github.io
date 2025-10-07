// A small subsets of https://github.com/jackyzha0/telescopic-text
// turned into remark plugin for Quartz
//
// Usage:
// ````
// ```telescopic id="unique-block"
// * reading
//   * reading a lot of Nietzsche,
//   * hosting functions,
//     * go on longs walks,
//     * building [[thoughts/work|open-source project]],
// ```
// ````
//
// NOTE: For this, we only a small subset of wikilinks, and it should always be absolute
//
// Some deviation: separator=" ", shouldExpandOnMouseOver=false
import { QuartzTransformerPlugin } from "../types"
import { Root, Element } from "hast"
import type { RootContent, Text } from "hast"
import { Root as MdastRoot, Code } from "mdast"
import { toHtml } from "hast-util-to-html"
import { SKIP, visit } from "unist-util-visit"
import { toString } from "hast-util-to-string"
import { h, s } from "hastscript"
import { wikilinkRegex } from "./ofm"
import { findAndReplace as hastFindReplace } from "hast-util-find-and-replace"
import isAbsoluteUrl from "is-absolute-url"
import { FullSlug, simplifySlug, splitAnchor, stripSlashes, transformLink } from "../../util/path"
import path from "path"
import { FindAndReplaceList } from "hast-util-find-and-replace"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
// @ts-ignore
import script from "../../components/scripts/telescopic.inline.ts"
import content from "../../components/styles/telescopic.inline.scss"
import { svgOptions } from "../../components/svg"

interface Line {
  og: string // the original string to replace
  new: string // the replacement string
  replacements: Line[] // nested replacements to apply on the resultant line afterwards
}

interface Content {
  text: string // Original string content in the line
  replacements: Line[] // Sections of the original text to replace/expand
}

interface NewContent {
  text: string
  expansions?: NewContent[]
  separator?: string
}

type TelescopicOutput = NewContent[]

interface TelescopeNode {
  depth: number
  telescopicOut: TelescopicOutput
}

interface Config {
  /**
   * Character used to separate entries on the same level. Defaults to a single space (" ")
   */
  separator?: string
  /**
   * If true, allows sections to expand automatically on mouse over rather than requiring a click. Defaults to false.
   */
  shouldExpandOnMouseOver?: boolean
}

const createEmptyTextNode = (): Text => ({ type: "text", value: "" })
const wikilinkRegex = createWikilinkRegex()

function applyInlineFormatting(value: string): string {
  if (!value) {
    return value
  }

  let formatted = value

  // Normalize hard and soft line breaks
  formatted = formatted.replace(/\\n/g, "<br />")
  formatted = formatted.replace(/\\(?=\s|$)/g, "<br />")
  formatted = formatted.replace(/\n\s*\n\s*\n/g, "<hr />")
  formatted = formatted.replace(/\n\s*\n/g, "<hr />")
  formatted = formatted.replace(/\r?\n/g, "<br />")
  formatted = formatted.replace(/<br\s*\/?>/gi, "<br />")
  formatted = formatted.replace(/&softbreak;/gi, "<wbr />")
  formatted = formatted.replace(/<softbreak\s*\/?>/gi, "<wbr />")

  // Bold / strong emphasis (**text** or __text__)
  formatted = formatted.replace(/(?<!\\)\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>")
  formatted = formatted.replace(/(?<!\\)__([\s\S]+?)__/g, "<strong>$1</strong>")

  // Italic emphasis (*text* or _text_)
  formatted = formatted.replace(/(?<!\\)\*(?!\*)([^*\n]+?)\*(?!\*)/g, "<em>$1</em>")
  formatted = formatted.replace(/(?<!\\)_(?!_)([^_\n]+?)_(?!_)/g, "<em>$1</em>")

  // Strikethrough
  formatted = formatted.replace(/(?<!\\)~~([\s\S]+?)~~/g, "<del>$1</del>")

  // Restore escaped characters that should remain literal
  formatted = formatted.replace(/\\\*/g, "*")
  formatted = formatted.replace(/\\_/g, "_")
  formatted = formatted.replace(/\\~/g, "~")
  formatted = formatted.replace(/\\\\/g, "\\")

  return formatted
}

function inlineFragmentToNodes(value: string): RootContent[] {
  if (!value) {
    return []
  }

  const formatted = applyInlineFormatting(value)
  const fragment = fromHtmlIsomorphic(formatted, { fragment: true }) as Root
  const children = (fragment.children ?? []) as RootContent[]
  return children
}

/*****************/
/* PARSING LOGIC */
/*****************/

// Parses the input string and returns the output as a structured data format.
function parseMarkdown(mdContent: string): TelescopicOutput {
  // In future we might want to support full markdown in which case..
  //   const html = marked.parse(mdContent);
  //  convert into jsdom
  //   const lines = html.split("\n");
  // Also idea for "..." or ellipsis character to represent infinite expansion.

  const BulletSeparators = ["*", "-", "+"]
  const RegexEscapedBulletSeparators = ["\\*", "-", "\\+"]

  const lines = mdContent.split("\n")
  // NOTE: this should handle normalizing the depth (if its an indented list)
  const root: TelescopicOutput = []
  const nodeStack: TelescopeNode[] = [{ depth: 0, telescopicOut: root }]

  // This is essentially a trie data structure to parse out all the bullet points
  // The algorithm works by assuming that any time you encounter a longer depth than the current one,
  // you are moving onto the next line.
  const firstNonEmptyLine = lines.find((l) => l.trim().length > 0)
  const defaultDepth =
    // @ts-ignore
    firstNonEmptyLine?.match(`^\\s*(${RegexEscapedBulletSeparators.join("|")})`)?.[0]?.length - 1 ||
    0
  for (const line of lines) {
    const trimmedLine = line.trim()
    if (!trimmedLine.length) {
      continue
    }

    // validate that the trimmed line starts with the bullet indicators.
    if (!BulletSeparators.some((sep) => trimmedLine.startsWith(sep))) {
      console.log(`Invalid line found! ${line}`)
      continue
    }

    // count all spaces in front to get current depth
    const currentDepth =
      line.match(`^\\s*(${RegexEscapedBulletSeparators.join("|")})`)![0].length - 1

    // if greater depth, attach on to the last one
    // else you can pop back up to one with 1 less depth
    while (
      nodeStack.length - 1 >= 0 &&
      currentDepth <= nodeStack[nodeStack.length - 1].depth &&
      nodeStack[nodeStack.length - 1].depth > 0
    ) {
      nodeStack.pop()
    }

    const { telescopicOut, ...restLastNode } = nodeStack[nodeStack.length - 1]
    const strippedLine = trimmedLine.substring(1).replace(/^\s+/, "")
    // Add current content / node to the stack
    const currentContent: NewContent = { text: strippedLine, expansions: [] }
    if (currentDepth === defaultDepth) {
      telescopicOut.push(currentContent)
      nodeStack[nodeStack.length - 1] = {
        ...restLastNode,
        telescopicOut,
      }
    } else {
      telescopicOut[telescopicOut.length - 1].expansions!.push(currentContent)
      // add this current one as a replacement to the last upper level one.
      const newNode = {
        depth: currentDepth,
        telescopicOut: [currentContent],
      }
      nodeStack.push(newNode)
    }
  }

  return root
}

// Ideally this would not be needed (just used to convert between data structures currently).
function outputToContent(output: TelescopicOutput, separator: string = " "): Content {
  const parseReplacementsFromOutput = (out: TelescopicOutput): Line[] => {
    return out.flatMap((line: NewContent) => {
      if (!line.expansions?.length) {
        return []
      }
      const newText = line.expansions.map((line) => line.text).join(separator)

      return [
        {
          og: line.text,
          new: newText,
          replacements: line.expansions?.length ? parseReplacementsFromOutput(line.expansions) : [],
        },
      ]
    })
  }
  const text = output.map((line: NewContent) => line.text).join(separator)
  const replacements: Line[] = parseReplacementsFromOutput(output)

  return { text, replacements }
}

// Ideally this would not be needed (just used to convert between data structures currently).
function mdToContent(mdContent: string, separator: string = " "): Content {
  const output = parseMarkdown(mdContent)
  return outputToContent(output, separator)
}

function contentToHast(content: Content, opts: Config) {
  function processContent(line: Content) {
    let lastIndex = 0
    const nodes: RootContent[] = []
    let lineText = line.text

    for (let i = 0; i < line.replacements.length; i++) {
      const replacement = line.replacements[i]

      const index = lineText.indexOf(replacement.og, lastIndex)
      const [before, ...after] = lineText.split(replacement.og)
      lineText = after.join(replacement.og)

      // Add text before replacement
      nodes.push(...inlineFragmentToNodes(before))

      // Create telescopic node
      const summaryChildren = inlineFragmentToNodes(replacement.og)
      const expandedChildren = processContent({
        text: replacement.new,
        replacements: replacement.replacements,
      })

      const detail: Element = h("span", { class: "details close" }, [
        // Original text
        h(
          "span",
          { class: "summary" },
          summaryChildren.length ? summaryChildren : [createEmptyTextNode()],
        ),
        // Expanded content
        h(
          "span",
          { class: "expanded" },
          expandedChildren.length ? expandedChildren : [createEmptyTextNode()],
        ),
      ])

      nodes.push(detail)
      lastIndex = index + replacement.og.length
    }

    // Add remaining text
    // and a smoll refresh button
    if (lastIndex < lineText.length) {
      nodes.push(...inlineFragmentToNodes(lineText.slice(lastIndex)))
    }

    return nodes
  }

  // Helper to get fully expanded text
  return h(
    "div#telescope",
    { "data-expand": opts.shouldExpandOnMouseOver ? "hover" : "click" },
    processContent(content),
  )
}

const defaultOpts: Config = {
  separator: " ",
  shouldExpandOnMouseOver: false,
}

const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g

export const TelescopicText: QuartzTransformerPlugin<Partial<Config>> = (userOpts) => {
  const opts = { ...defaultOpts, ...userOpts }

  return {
    name: "TelescopicText",
    markdownPlugins() {
      return [
        () => (tree: MdastRoot, file) => {
          visit(tree, "code", (node: Code) => {
            let { lang, meta } = node
            if (lang === "telescopic" && meta) {
              // Parse the meta string to extract ID
              const idMatch = meta!.match(/id=["']?([^"'\s]+)["']?/)
              if (!idMatch) {
                console.error(
                  `Missing id for targeted telescopic node ${node.position}, skipping instead (${file.data.filePath!})`,
                )
                return SKIP
              }
              node.data = node.data || {}
              node.data.hProperties = { id: idMatch[1] + "-container" }
            }
          })
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => (tree: Root, file) => {
          const curSlug = simplifySlug(file.data.slug!)

          const wikilinksReplace = (
            _value: string,
            ...capture: string[]
          ): { dest: string; alias: string; dataSlug: string } => {
            let [rawFp, rawHeader, rawAlias] = capture
            const fp = rawFp?.trim() ?? ""
            const anchor = rawHeader?.trim() ?? ""
            const alias = rawAlias?.slice(1).trim()

            let dest = fp + anchor
            const ext: string = path.extname(dest).toLowerCase()
            if (isAbsoluteUrl(dest)) return { dest, alias: dest, dataSlug: dest }

            if (!ext.includes("pdf")) {
              dest = transformLink(file.data.slug!, dest, {
                allSlugs: ctx.allSlugs,
                strategy: "absolute",
              })
            }

            // url.resolve is considered legacy
            // WHATWG equivalent https://nodejs.dev/en/api/v18/url/#urlresolvefrom-to
            const url = new URL(dest, "https://base.com/" + stripSlashes(curSlug, true))
            const canonicalDest = url.pathname
            let [destCanonical, _destAnchor] = splitAnchor(canonicalDest)
            if (destCanonical.endsWith("/")) {
              destCanonical += "index"
            }

            // need to decodeURIComponent here as WHATWG URL percent-encodes everything
            const full = decodeURIComponent(stripSlashes(destCanonical, true)) as FullSlug
            return { dest, alias: alias ?? path.basename(fp), dataSlug: full }
          }

          const checkParsedCodeblock = ({ tagName, children }: Element): boolean => {
            if (tagName !== "pre" || !Array.isArray(children) || children.length === 0) {
              return false
            }

            const [code] = children as Element[]
            const { properties, tagName: codeTagName } = code
            return (
              codeTagName === "code" &&
              Boolean(properties.className) &&
              (properties.className as string[]).includes("language-telescopic")
            )
          }

          visit(
            tree,
            (node) => checkParsedCodeblock(node as Element),
            (node, index, parent) => {
              const code = (node as Element).children[0] as Element

              const replacements: FindAndReplaceList = [
                [
                  wikilinkRegex,
                  (match, ...link) => {
                    const { dest, alias } = wikilinksReplace(match, ...link)

                    return toHtml(h("a", { href: dest }, { type: "text", value: alias }))
                  },
                ],
                [
                  linkRegex,
                  (_match, value, href) => {
                    return toHtml(h("a", { href }, { type: "text", value }))
                  },
                ],
              ]
              hastFindReplace(code, replacements)

              const content = mdToContent(toString(code), opts.separator)
              parent!.children.splice(
                index!,
                1,
                h(
                  "div.telescopic-container",
                  { id: code.properties.id },
                  h(
                    "span",
                    { class: "expand", type: "button" },
                    s(
                      "svg",
                      {
                        ...svgOptions,
                        height: 12,
                        width: 12,
                        strokewidth: 1,
                        stroke: "currentColor",
                        fill: "var(--lightgray)",
                        title: "expand all state",
                        ariaLabel: "expand all state",
                      },
                      s("use", { href: "#plus-icon" }),
                    ),
                  ),
                  h(
                    "span",
                    { class: "replay", type: "button" },
                    s(
                      "svg",
                      {
                        ...svgOptions,
                        height: 12,
                        width: 12,
                        fill: "var(--lightgray)",
                        title: "refresh telescopic text state",
                        ariaLabel: "refresh telescopic text state",
                      },
                      s("use", { href: "#refetch-icon" }),
                    ),
                  ),
                  contentToHast(content, opts),
                ),
              )
            },
          )
        },
      ]
    },
    externalResources() {
      return {
        js: [{ script, contentType: "inline", loadTime: "afterDOMReady" }],
        css: [{ content, spaPreserve: true, inline: true }],
      }
    },
  }
}
