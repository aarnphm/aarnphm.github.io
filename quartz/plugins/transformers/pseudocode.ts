import { QuartzTransformerPlugin } from "../types"
import { Root as MdRoot } from "mdast"
import { Root as HTMLRoot, Literal } from "hast"
import { visit } from "unist-util-visit"
import { VFile } from "vfile"
// @ts-ignore
import Lexer from "pseudocode/src/Lexer.js"
// @ts-ignore
import Parser from "pseudocode/src/Parser.js"
// @ts-ignore
import Renderer from "pseudocode/src/Renderer.js"

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
    commentDelimiter: "//",
    lineNumberPunc: ":",
    lineNumber: true,
    noEnd: false,
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

export const Pseudocode: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  /**
   * Used to store the LaTex raw string content in order as they are found in the markdown file.
   * They will be processed in the same order later on to be converted to HTML.
   */
  const latexBlock: string[] = []

  return {
    name: "Pseudocode",
    markdownPlugins() {
      return [
        () => (tree: MdRoot, _file) => {
          visit(tree, "code", (node) => {
            if (node.lang === opts.code) {
              latexBlock.push(node.value)

              node.type = "html" as "code"
              node.value = `<pre class="${opts.css}"></pre>`
            }
          })
        },
      ]
    },
    htmlPlugins() {
      return [
        () => (tree: HTMLRoot, _file: VFile) => {
          visit(tree, "raw", (raw: Literal) => {
            if (raw.value !== `<pre class="${opts.css}"></pre>`) {
              return
            }

            const value = latexBlock.shift()
            const markup = renderToString(value!, opts?.renderer)
            if (opts.removeCaptionCount) {
              raw.value = removeCaptionCount(markup, opts?.renderer?.titlePrefix ?? "Algorithm")
            } else {
              raw.value = markup
            }
          })
        },
      ]
    },
  }
}
