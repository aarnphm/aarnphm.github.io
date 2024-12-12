import { QuartzTransformerPlugin } from "../types"
import { Root as MdRoot } from "mdast"
import { Root as HTMLRoot, Literal, Element } from "hast"
import { visit } from "unist-util-visit"
// @ts-ignore
import Lexer from "pseudocode/src/Lexer.js"
// @ts-ignore
import Parser from "pseudocode/src/Parser.js"
// @ts-ignore
import Renderer from "pseudocode/src/Renderer.js"
import { s, h } from "hastscript"
import { fromHtml } from "hast-util-from-html"
import { extractInlineMacros } from "../../util/latex"

export interface Options {
  code: string
  css: string
  removeCaptionCount?: boolean
  renderer?: RendererOptions
}

/**
 * Options of the renderer itself. These are a subset of the options that can be passed to the Quartz plugin.
 * See the PseudocodeOptions type for the full list of options.
 */
interface RendererOptions {
  /**
   * The indent size of inside a control block, e.g. if, for, etc. The unit must be in 'em'. Default value: '1.2em'.
   */
  indentSize?: string
  /**
   * The delimiters used to start and end a comment region. Note that only line comments are supported. Default value: '//'.
   */
  commentDelimiter?: string
  /**
   * The punctuation that follows line number. Default value: ':'.
   */
  lineNumberPunc?: string
  /**
   * Whether line numbering is enabled. Default value: false.
   */
  lineNumber?: boolean
  /**
   * Whether block ending, like `end if`, end `procedure`, etc., are showned. Default value: false.
   */
  noEnd?: boolean
  /**
   * Set the caption counter to this new value.
   */
  captionCount?: number
  /**
   * Whether to set scope lines
   */
  scopeLines?: boolean
  /**
   * The prefix in the title of the algorithm. Default value: 'Algorithm'.
   */
  titlePrefix?: string

  mathEngine?: "katex" | "mathjax"
  mathRenderer?: (input: string) => string
}

const defaultOptions: Options = {
  code: "pseudo",
  css: "latex-pseudo",
  removeCaptionCount: false,
  renderer: {
    indentSize: "0.6em",
    commentDelimiter: "  ▷",
    lineNumberPunc: ":",
    lineNumber: true,
    noEnd: false,
    scopeLines: false,
    captionCount: undefined,
    titlePrefix: "Algorithm",
    mathEngine: "katex",
    mathRenderer: undefined,
  },
}

function renderToString(input: string, options?: RendererOptions) {
  if (input === null || input === undefined) throw new ReferenceError("Input cannot be empty")

  const lexer = new Lexer(input)
  const parser = new Parser(lexer)
  const renderer = new Renderer(parser, options)
  if (options?.mathEngine || options?.mathRenderer) {
    renderer.backend ??= {}
    renderer.backend.name ??= options?.mathEngine
    renderer.backend.driver ??= {}
    renderer.backend.driver.renderToString ??= options?.mathRenderer
  }
  return renderer.toMarkup()
}

/**
 * Experimental feature to remove the caption count from the title of the rendered pseudocode using a RegEx.
 *
 * @param renderedMarkup The HTML markup that was generated from LaTex by pseudocode.js
 * @param captionValue The value used for the title of the rendered pseudocode (by default 'Algorithm')
 * @returns The HTML markup without the caption count
 */
function removeCaptionCount(renderedMarkup: string, captionValue: string): string {
  // Escape potential special regex characters in the custom caption
  const escapedCaption = captionValue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

  const regex = new RegExp(`<span class="ps-keyword">${escapedCaption} [-]?\\d+[ ]?<\\/span>`, "g")
  return renderedMarkup.replace(regex, `<span class="ps-keyword">${captionValue} </span>`)
}

function parseMeta(meta: string | null, opts: Options) {
  if (!meta) meta = ""

  const lineNumberMatch = meta.match(/lineNumber=(false|true|0|1)/i)
  const lnum = lineNumberMatch?.[1] ?? null
  let enableLineNumber: boolean
  if (lnum) {
    enableLineNumber = lnum === "true" || lnum === "1"
  } else {
    enableLineNumber = opts.renderer?.lineNumber
  }
  meta = meta.replace(lineNumberMatch?.[0] ?? "", "")

  return { enableLineNumber, meta }
}

export const Pseudocode: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  /**
   * Used to store the LaTeX raw string content in order as they are found in the markdown file.
   * They will be processed in the same order later on to be converted to HTML.
   */
  const latexBlock: string[] = []

  return {
    name: "Pseudocode",
    markdownPlugins() {
      return [
        () => (tree: MdRoot, file) => {
          visit(tree, "code", (node) => {
            let { lang, meta, value } = node
            if (lang === opts.code) {
              const { enableLineNumber } = parseMeta(meta!, opts)
              latexBlock.push(value)
              node.type = "html" as "code"
              node.value = `<pre class="${opts.css}" data-line-number=${enableLineNumber}></pre>`
            }
          })
          file.data.pseudocode = latexBlock.length !== 0
        },
      ]
    },
    htmlPlugins() {
      return [
        () => (tree: HTMLRoot, _file) => {
          visit(tree, "raw", (node: Literal, index, parent) => {
            const lineNoMatch = node.value.match(/data-line-number=([^>\s]+)/)
            if (!lineNoMatch || !node.value.includes(`class="${opts.css}"`)) {
              return
            }
            const lineNo = lineNoMatch[1].toLowerCase()
            const enableLineNumber = lineNo === "true"

            // PERF: we are currently doing one round trip from text -> html -> hast
            // pseudocode (katex backend) --|renderToString|--> html string --|fromHtml|--> hast
            // ideally, we should cut this down to render directly to hast
            const value = latexBlock.shift()
            const [inlineMacros, algo] = extractInlineMacros(value ?? "")
            // TODO: Might be able to optimize.
            // find all $ enclosements in source, and add the preamble.
            const mathRegex = /\$(.*?)\$/g
            const algoWithPreamble = algo.replace(mathRegex, (_, p1) => {
              return `$${inlineMacros}${p1}$`
            })

            const markup = renderToString(algoWithPreamble!, {
              ...opts?.renderer,
              lineNumber: enableLineNumber,
            })
            if (opts.removeCaptionCount) {
              node.value = removeCaptionCount(markup, opts?.renderer?.titlePrefix ?? "Algorithm")
            } else {
              node.value = markup
            }

            const htmlNode = fromHtml(node.value, { fragment: true })
            const renderedContainer = htmlNode.children[0] as Element
            renderedContainer.properties.dataInlineMacros = inlineMacros
            renderedContainer.properties.dataSettings = JSON.stringify(opts)

            const button: Element = h(
              "span",
              {
                type: "button",
                class: "clipboard-button ps-clipboard",
                ariaLabel: "Copy pseudocode to clipboard",
                ariaHidden: true,
                tabindex: -1,
              },
              [
                s("svg", { width: 16, height: 16, viewbox: "0 0 16 16", class: "copy-icon" }, [
                  s("use", { href: "#github-copy" }),
                ]),
                s("svg", { width: 16, height: 16, viewbox: "0 0 16 16", class: "check-icon" }, [
                  s("use", {
                    href: "#github-check",
                    fillRule: "evenodd",
                    fill: "rgb(63, 185, 80)",
                  }),
                ]),
              ],
            )
            const mathML: Element = h("span", { class: "ps-mathml" }, [
              h("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
                h("semantics", [
                  h("annotation", { encoding: "application/x-tex" }, [
                    { type: "text", value: JSON.stringify(algoWithPreamble) },
                  ]),
                ]),
              ]),
            ])

            renderedContainer.children = [button, mathML, ...renderedContainer.children]
            parent!.children.splice(index!, 1, renderedContainer)
          })
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    pseudocode: boolean
  }
}

declare module "mdast" {
  interface CodeData {
    pseudocode?: boolean | undefined
  }
}
