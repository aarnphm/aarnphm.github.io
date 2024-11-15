import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { clone, FullSlug, normalizeHastElement, FilePath } from "../util/path"
import { classNames } from "../util/lang"
import { visit } from "unist-util-visit"
import { Node, Element, ElementContent, Root } from "hast"
import { mergeIsomorphic } from "./renderPage"
import { htmlToJsx } from "../util/jsx"
import style from "./styles/reader.scss"
// @ts-ignore
import readerScript from "./scripts/reader.inline"

const headerRegex = new RegExp(/h[1-6]/)

export default (() => {
  const Reader: QuartzComponent = ({ displayClass, fileData, allFiles }: QuartzComponentProps) => {
    // do some cleaning ast, so we need to clone not to affect the original nodes
    const { htmlAst, slug, filePath } = fileData
    const ast = clone(htmlAst) as Node

    // NOTE: this blockquote logic is similar from renderPage with some slight deviations:
    // - we don't add title, and no backlinks to original transclude, simply just dump the content into the container
    // - for the parent blockquote we dump the children directly
    visit(ast, "element", (node: Element, idx: number, parent: Element) => {
      if (node.tagName === "blockquote") {
        const classNames = (node.properties?.className ?? []) as string[]

        if (classNames.includes("transclude")) {
          const inner = node.children[0] as Element
          const transcludeTarget = inner.properties["data-slug"] as FullSlug
          const page = allFiles.find((f) => f.slug === transcludeTarget)
          if (!page) {
            return
          }

          let blockRef = node.properties.dataBlock as string | undefined
          if (blockRef?.startsWith("#^")) {
            // block transclude
            blockRef = blockRef.slice("#^".length)
            let blockNode = page.blocks?.[blockRef]
            if (blockNode) {
              if (blockNode.tagName === "li") {
                blockNode = {
                  type: "element",
                  tagName: "ul",
                  properties: {},
                  children: [blockNode],
                }
              }

              parent.children.splice(
                idx,
                1,
                normalizeHastElement(blockNode, slug as FullSlug, transcludeTarget),
              )
            }
          } else if (blockRef?.startsWith("#") && page.htmlAst) {
            // header transclude
            blockRef = blockRef.slice(1)
            let startIdx = undefined
            let startDepth = undefined
            let endIdx = undefined
            for (const [i, el] of page.htmlAst.children.entries()) {
              // skip non-headers
              if (!(el.type === "element" && el.tagName.match(headerRegex))) continue
              const depth = Number(el.tagName.substring(1))

              // looking for our blockref
              if (startIdx === undefined || startDepth === undefined) {
                // skip until we find the blockref that matches
                if (el.properties?.id === blockRef) {
                  startIdx = i
                  startDepth = depth
                }
              } else if (depth <= startDepth) {
                // looking for new header that is same level or higher
                endIdx = i
                break
              }
            }

            if (startIdx === undefined) {
              return
            }

            parent.children.splice(
              idx,
              1,
              ...[
                ...(page.htmlAst.children.slice(startIdx, endIdx) as ElementContent[]).map(
                  (child) =>
                    normalizeHastElement(child as Element, slug as FullSlug, transcludeTarget),
                ),
              ],
            )
          } else if (page.htmlAst) {
            // page transclude
            parent.children.splice(
              idx,
              1,
              ...[
                ...(page.htmlAst.children as ElementContent[]).map((child) =>
                  normalizeHastElement(child as Element, slug as FullSlug, transcludeTarget),
                ),
              ],
            )
          }
        }

        // here we simplify the callout
        if (classNames.includes("is-collapsible")) {
          // We need to unparse collapsible callout
          node.properties.className = ["callout", node.properties["data-callout"] as string]
          node.properties.style = ""
        }
        if (classNames.includes("callout")) {
          visit(node, "element", (descendant) => {
            if (descendant.tagName === "div") {
              const classNames = (descendant.properties?.className ?? []) as string[]
              if (classNames.includes("callout-title")) {
                // Filter out callout-icon and fold-callout-icon divs
                descendant.children = descendant.children.filter((children) => {
                  if (children.type === "element") {
                    const childClassNames = (children.properties?.className ?? []) as string[]
                    return !(
                      childClassNames.includes("callout-icon") ||
                      childClassNames.includes("fold-callout-icon")
                    )
                  }
                  return true
                })
              }
            }
          })
        }
      }
    })
    // keep the same references and footnotes parsing, but append reader for isomorphic ID
    mergeIsomorphic(ast as Root, "reader")
    // remove all alias and popover in modified AST for better reading
    visit(ast, "element", (node: Element) => {
      if (node.tagName === "a") {
        const classNames = (node.properties?.className ?? []) as string[]
        if (classNames.includes("internal")) {
          node.properties.className = ["internal"]
          node.properties["data-no-popover"] = true
        }
      }
    })
    // remove expand button for mermaid block
    visit(ast, "element", (node: Element, _index, parent: Element) => {
      if (
        node.tagName === "code" &&
        ((node.properties?.className ?? []) as string[]).includes("mermaid")
      ) {
        // Filter out the expand-button that appears before the code block
        parent.children = parent.children.filter((child) => {
          if (child.type === "element") {
            return (
              !((child.properties?.className ?? []) as string[]).includes("expand-button") &&
              child.properties.id !== "mermaid-container"
            )
          }
          return true
        })
      }
    })
    // cleanup colorscheme, we just need monotone for reader mode
    visit(ast, "element", (node: Element) => {
      // Handle code block with data-language
      if (node.tagName === "code" && node.properties.dataLanguage) {
        // Remove data-theme attribute if it exists
        if (node.properties.dataTheme) {
          delete node.properties.dataTheme
        }

        // Clean up shiki styles
        if (node.properties?.style) {
          const style = node.properties.style as string
          // Split style into individual properties
          const styles = style.split(";").filter((s) => s.trim())
          // Filter out any styles containing --shiki
          const cleanedStyles = styles.filter((s) => !s.includes("--shiki"))

          if (cleanedStyles.length > 0) {
            node.properties.style = cleanedStyles.join(";")
          } else {
            delete node.properties.style
          }
        }
      }

      // Also check pre tags as they sometimes contain the theme data
      if (node.tagName === "pre") {
        if (node.properties.dataTheme) {
          delete node.properties.dataTheme
        }

        if (node.properties?.style) {
          const style = node.properties.style as string
          const styles = style.split(";").filter((s) => s.trim())
          const cleanedStyles = styles.filter((s) => !s.includes("--shiki"))

          if (cleanedStyles.length > 0) {
            node.properties.style = cleanedStyles.join(";")
          } else {
            delete node.properties.style
          }
        }
      }
    })
    // Append suffix to header nodes
    visit(ast, "element", (node: Element) => {
      const suffix = "-reader"

      // Check if it's a header element using the existing headerRegex
      if (node.tagName.match(headerRegex)) {
        // Only modify if the header has an id
        if (node.properties?.id) {
          // Append the suffix to the existing id
          node.properties.id = `${node.properties.id}${suffix}`

          // Also update any anchor links within the header that reference the old id
          visit(node, "element", (child: Element) => {
            if (child.tagName === "a" && (child.properties?.href as string)?.startsWith("#")) {
              const href = child.properties.href as string
              child.properties.href = `${href}${suffix}`
            }
          })
        }
      }
    })
    // remove all hr
    visit(ast, "element", (node: Element, index: number, parent: Element) => {
      if (node.tagName === "hr") {
        parent.children.splice(index, 1)
      }
    })

    return (
      <div class={classNames(displayClass, "reader")} id="reader-view">
        <div class="reader-backdrop" />
        <div class="reader-container">
          <div class="reader-header">
            <button class="reader-close" aria-label="Close reader">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
          <div class="reader-content">{htmlToJsx(filePath as FilePath, ast)}</div>
        </div>
      </div>
    )
  }
  Reader.css = style
  Reader.afterDOMLoaded = readerScript

  return Reader
}) satisfies QuartzComponentConstructor
