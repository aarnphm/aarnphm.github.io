import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { h, s } from "hastscript"
import rehypePrettyCode, { Options as CodeOptions, Theme as CodeTheme } from "rehype-pretty-code"
import { visit } from "unist-util-visit"
import { svgOptions } from "../../components/renderPage"

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
