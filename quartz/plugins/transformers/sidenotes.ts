import { QuartzTransformerPlugin } from "../types"
import { Root as HastRoot } from "hast"
import { findAndReplace as hastFindReplace } from "hast-util-find-and-replace"
import { h } from "hastscript"
import { toHtml } from "hast-util-to-html"
import { fromMarkdown } from "mdast-util-from-markdown"
import { toHast } from "mdast-util-to-hast"
import { createWikilinkRegex } from "../../util/wikilinks"
import isAbsoluteUrl from "is-absolute-url"
import { simplifySlug, splitAnchor, stripSlashes, transformLink } from "../../util/path"
import path from "path"
import { FindAndReplaceList } from "hast-util-find-and-replace"
// @ts-ignore
import script from "../../components/scripts/sidenotes.inline"
import content from "../../components/styles/sidenotes.inline.scss"

const wikilinkRegex = createWikilinkRegex()
const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g

export const Sidenotes: QuartzTransformerPlugin = () => {
  return {
    name: "Sidenotes",
    externalResources() {
      return {
        js: [{ script, contentType: "inline", loadTime: "afterDOMReady" }],
        css: [{ content, spaPreserve: true, inline: true }],
      }
    },
    htmlPlugins(ctx) {
      return [
        () => (tree: HastRoot, file) => {
          const curSlug = simplifySlug(file.data.slug!)
          let counter = 0

          const sidenoteRegex = /\{\{sidenotes(?:<([^>]+)>)?(?:\[([^\]]+)\])?:\s*([^}]+)\}\}/g

          const wikilinksReplace = (
            _value: string,
            ...capture: string[]
          ): { dest: string; alias: string } => {
            let [rawFp, rawHeader, rawAlias] = capture
            const fp = rawFp?.trim() ?? ""
            const anchor = rawHeader?.trim() ?? ""
            const alias = rawAlias?.slice(1).trim()

            let dest = fp + anchor
            const ext: string = path.extname(dest).toLowerCase()
            if (isAbsoluteUrl(dest)) return { dest, alias: dest }

            if (!ext.includes("pdf")) {
              dest = transformLink(file.data.slug!, dest, {
                allSlugs: ctx.allSlugs,
                strategy: "absolute",
              })
            }

            const url = new URL(dest, "https://base.com/" + stripSlashes(curSlug, true))
            const canonicalDest = url.pathname
            let [destCanonical, _destAnchor] = splitAnchor(canonicalDest)
            if (destCanonical.endsWith("/")) {
              destCanonical += "index"
            }

            return { dest, alias: alias ?? path.basename(fp) }
          }

          const processMarkdownContent = (content: string): string => {
            // Parse markdown to mdast
            const mdast = fromMarkdown(content)

            // Create replacements for wikilinks and regular links
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

            // Convert to hast
            const hast = toHast(mdast, { allowDangerousHtml: true }) as HastRoot

            // Apply wikilink replacements to text nodes
            hastFindReplace(hast, replacements)

            // Serialize to HTML
            return toHtml(hast, { allowDangerousHtml: true })
          }

          const parseAttributes = (attrs: string | undefined): Record<string, string> => {
            if (!attrs) return {}

            const result: Record<string, string> = {}
            attrs.split(",").forEach((attr) => {
              const [key, value] = attr.split(":").map((s) => s.trim())
              if (key && value) {
                result[key] = value
              }
            })
            return result
          }

          const replaceSidenote = (
            _match: string,
            attributes: string | undefined,
            label: string | undefined,
            content: string,
          ) => {
            counter++
            const processedContent = processMarkdownContent(content.trim())
            const attrs = parseAttributes(attributes)

            const hasLabel = label && label.trim().length > 0

            // create arrow SVG for dropdown indicator
            const arrowSvg = h(
              "svg.sidenote-arrow",
              {
                width: "10",
                height: "6",
                viewBox: "0 0 10 6",
                fill: "none",
                xmlns: "http://www.w3.org/2000/svg",
                "aria-hidden": "true",
              },
              [
                h("path", {
                  d: "M1 1L5 5L9 1",
                  stroke: "currentColor",
                  "stroke-width": "1.5",
                  "stroke-linecap": "round",
                  "stroke-linejoin": "round",
                }),
              ],
            )

            const labelElement = h(
              "span.sidenote-label",
              hasLabel ? {} : { "data-auto": "" },
              hasLabel
                ? [{ type: "text", value: label }, arrowSvg]
                : [{ type: "text", value: "▪" }, arrowSvg],
            )

            // build data attributes from parsed attributes
            const dataAttrs: Record<string, string> = {
              "data-content": processedContent,
              "data-sidenote-id": counter.toString(),
            }

            // map user attributes to data attributes
            if (attrs.dropdown === "true" || attrs.inline === "true") {
              dataAttrs["data-force-inline"] = "true"
            }
            if (attrs.left === "false") {
              dataAttrs["data-allow-left"] = "false"
            }

            return h("span.sidenote", dataAttrs, [labelElement])
          }

          // Find and replace sidenote syntax in text nodes
          hastFindReplace(tree, [
            [
              sidenoteRegex,
              (match: string, ...groups: string[]) => {
                const attributes = groups[0]
                const label = groups[1]
                const content = groups[2]
                return replaceSidenote(match, attributes, label, content)
              },
            ],
          ])
        },
      ]
    },
  }
}
