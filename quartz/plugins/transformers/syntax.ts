import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { h, s } from "hastscript"
import rehypePrettyCode, { Options as CodeOptions, Theme as CodeTheme } from "rehype-pretty-code"
import { visit } from "unist-util-visit"
import { svgOptions } from "../../components/svg"
import type { Root as MdastRoot, Code as MdastCode } from "mdast"

interface Theme extends Record<string, CodeTheme> {
  light: CodeTheme
  dark: CodeTheme
}

interface Options extends CodeOptions {
  theme?: Theme
  keepBackground?: boolean
}

const defaultOptions: Options = {
  theme: {
    light: "github-light",
    dark: "github-dark",
  },
  keepBackground: false,
}

export const SyntaxHighlighting: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts: CodeOptions = { ...defaultOptions, ...userOpts }

  return {
    name: "SyntaxHighlighting",
    markdownPlugins() {
      // Detect code fences with `disableLineNumber=false` in the meta and annotate
      // them so the HTML stage can toggle CSS-based line numbers.
      return [
        () => {
          return (tree: MdastRoot) => {
            visit(tree, "code", (node: MdastCode) => {
              const meta = node.meta ?? ""
              // Match `disableLineNumber=false` (case-insensitive, allow spaces)
              const disableLN = /(?:^|\s)disableLineNumber\s*=\s*true(?:\s|$)/i.test(meta)
              if (disableLN) {
                // Attach an attribute that will flow to the HAST node
                // via remark-rehype as element properties.
                const dataAny = node as unknown as { data?: any }
                dataAny.data = dataAny.data ?? {}
                dataAny.data.hProperties = {
                  ...dataAny.data.hProperties,
                  "data-disable-line-number": "true",
                }
                // Strip the directive from meta so it doesn't leak into other tooling
                node.meta = meta
                  .replace(/(?:^|\s)disableLineNumber\s*=\s*false(?:\s|$)/gi, " ")
                  .trim()
              }
            })
          }
        },
      ]
    },
    htmlPlugins() {
      return [
        [rehypePrettyCode, opts],
        () => {
          return (tree) => {
            const isCodeblockTranspiled = ({ children, tagName }: Element) => {
              if (children === undefined || children === null) return false
              const maybeCodes = children.filter((c) => (c as Element).tagName === "code")
              return tagName === "pre" && maybeCodes.length != 0 && maybeCodes.length === 1
            }
            visit(
              tree,
              (node) => isCodeblockTranspiled(node as Element),
              (node, _idx, _parent) => {
                // If the child code element carries our disabling attribute,
                // mirror it on the <pre> wrapper to make CSS targeting easier.
                try {
                  const preEl = node as Element
                  const codeEl = preEl.children.find((c) => (c as Element).tagName === "code") as
                    | Element
                    | undefined
                  const hasDisable = Boolean(
                    codeEl?.properties && (codeEl.properties as any)["data-disable-line-number"],
                  )
                  if (hasDisable) {
                    preEl.properties = preEl.properties ?? {}
                    preEl.properties["data-disable-line-number"] = "true"
                    // Also mirror on the parent wrapper (figure/div) if present
                    const parentEl = _parent as Element | undefined
                    if (parentEl && typeof parentEl.tagName === "string") {
                      parentEl.properties = parentEl.properties ?? {}
                      parentEl.properties["data-disable-line-number"] = "true"
                    }
                  }
                } catch {}
                node.children = [
                  h("span.clipboard-button", { type: "button", ariaLabel: "copy source" }, [
                    s("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "copy-icon" }, [
                      s("use", { href: "#github-copy" }),
                    ]),
                    s("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "check-icon" }, [
                      s("use", { href: "#github-check" }),
                    ]),
                  ]),
                  ...node.children,
                ]
              },
            )
          }
        },
      ]
    },
  }
}
