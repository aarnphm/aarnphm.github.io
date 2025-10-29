import { QuartzTransformerPlugin } from "../types"
import {
  Root as HastRoot,
  Parent as HastParent,
  RootContent as HastContent,
  Text as HastText,
} from "hast"
import { visit } from "unist-util-visit"
import { h } from "hastscript"
import { toHtml } from "hast-util-to-html"
import { toText } from "hast-util-to-text"
import { BuildCtx } from "../../util/ctx"
import { FullSlug, transformLink } from "../../util/path"
import {
  createWikilinkRegex,
  parseWikilink,
  resolveWikilinkTarget,
  singleWikilinkRegex as SINGLE_WIKILINK_REGEX,
} from "../../util/wikilinks"
import { VFile } from "vfile"
import path from "path"
import isAbsoluteUrl from "is-absolute-url"
// @ts-ignore
import script from "../../components/scripts/sidenotes.inline"
import content from "../../components/styles/sidenotes.inline.scss"

const SIDENOTE_PREFIX = "{{sidenotes"
const SIDENOTE_SUFFIX = "}}"

type ParentWithChildren = HastParent & { children: HastContent[] }
type Token = { kind: "text"; value: string } | { kind: "node"; node: HastContent }
type CollectedTokens = { tokens: Token[]; endIndex: number; tailText: string }
type ParsedTokens = {
  attrsText?: string
  labelText: string
  contentNodes: HastContent[]
}

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
        () => (tree: HastRoot, file: VFile) => {
          let counter = 0

          visit(tree, (node) => {
            if (!node || !("children" in node)) {
              return
            }
            processParent(node as ParentWithChildren)
          })

          function processParent(parent: ParentWithChildren) {
            for (let index = 0; index < parent.children.length; index++) {
              const child = parent.children[index]
              if (child.type !== "text") continue

              const textNode = child as HastText
              const markerIndex = textNode.value.indexOf(SIDENOTE_PREFIX)
              if (markerIndex === -1) continue

              if (markerIndex > 0) {
                const before: HastText = {
                  type: "text",
                  value: textNode.value.slice(0, markerIndex),
                }
                const after: HastText = { type: "text", value: textNode.value.slice(markerIndex) }
                parent.children.splice(index, 1, before, after)
                index++
              }

              const current = parent.children[index] as HastText
              current.value = current.value.slice(SIDENOTE_PREFIX.length)

              const extraction = extractSidenote(parent, index, file, ctx, () => ++counter)
              if (!extraction) continue

              parent.children.splice(index, extraction.removalCount, ...extraction.replacements)

              const adjust =
                extraction.replacements.length >= 2 ? extraction.replacements.length - 2 : 0
              index += adjust
            }
          }
        },
      ]
    },
  }
}

function extractSidenote(
  parent: ParentWithChildren,
  startIndex: number,
  file: VFile,
  ctx: BuildCtx,
  nextId: () => number,
) {
  const collected = collectTokens(parent, startIndex)
  if (!collected) return null

  const parsed = parseTokens(collected.tokens)
  if (!parsed) return null

  const { attrsText, labelText: rawLabel, contentNodes } = parsed
  const { attrs, internal } = parseAttributes(attrsText)

  const contentHtml = serializeContent(contentNodes)
  const internalHtml = renderInternalLinks(internal, file, ctx)
  const combinedHtml = combineContent(contentHtml, internalHtml)

  const labelText = deriveLabel(rawLabel)
  const sidenoteId = nextId()

  const replacements = buildReplacementNodes(
    labelText,
    attrs,
    combinedHtml,
    sidenoteId,
    collected.tailText,
  )

  return {
    removalCount: collected.endIndex - startIndex + 1,
    replacements,
  }
}

function collectTokens(parent: ParentWithChildren, startIndex: number): CollectedTokens | null {
  const tokens: Token[] = []
  let tailText = ""
  let endIndex = startIndex

  for (let idx = startIndex; idx < parent.children.length; idx++) {
    const node = parent.children[idx]

    if (node.type === "text") {
      let cursor = 0
      const value = (node as HastText).value

      while (cursor <= value.length) {
        const closeIndex = value.indexOf(SIDENOTE_SUFFIX, cursor)
        if (closeIndex === -1) {
          const segment = value.slice(cursor)
          if (segment.length > 0) {
            tokens.push({ kind: "text", value: segment })
          }
          break
        } else {
          const segment = value.slice(cursor, closeIndex)
          if (segment.length > 0) {
            tokens.push({ kind: "text", value: segment })
          }
          tailText = value.slice(closeIndex + SIDENOTE_SUFFIX.length)
          endIndex = idx
          return { tokens, endIndex, tailText }
        }
      }
    } else {
      tokens.push({ kind: "node", node })
    }

    endIndex = idx
  }

  return null
}

function parseTokens(tokens: Token[]): ParsedTokens | null {
  if (tokens.length === 0) return null

  let state: "header" | "content" = "header"
  let inLabel = false
  let inAngle = false

  let attrsText = ""
  let labelText = ""
  const contentNodes: HastContent[] = []

  for (const token of tokens) {
    if (state === "header") {
      if (token.kind === "text") {
        const value = token.value
        let index = 0
        while (index < value.length) {
          const char = value[index]

          if (char === "<" && !inLabel && !inAngle) {
            inAngle = true
            index++
            continue
          }
          if (char === ">" && inAngle) {
            inAngle = false
            index++
            continue
          }
          if (char === "[" && !inAngle && !inLabel) {
            inLabel = true
            index++
            continue
          }
          if (char === "]" && inLabel) {
            inLabel = false
            index++
            continue
          }
          if (inAngle) {
            attrsText += char
            index++
            continue
          }
          if (inLabel) {
            labelText += char
            index++
            continue
          }
          if (char === ":") {
            state = "content"
            const remainder = value.slice(index + 1)
            if (remainder.length > 0) {
              contentNodes.push({ type: "text", value: remainder } as HastText)
            }
            break
          }
          index++
        }
      } else {
        if (inLabel) {
          labelText += toText(token.node)
        } else if (inAngle) {
          attrsText += toText(token.node)
        }
      }
    } else {
      if (token.kind === "text") {
        if (token.value.length > 0) {
          contentNodes.push({ type: "text", value: token.value } as HastText)
        }
      } else {
        contentNodes.push(token.node)
      }
    }
  }

  if (state !== "content") return null

  return {
    attrsText: attrsText.trim() || undefined,
    labelText: labelText.trim(),
    contentNodes,
  }
}

function parseAttributes(attrsText?: string): { attrs: Record<string, string>; internal?: string } {
  if (!attrsText) return { attrs: {} }

  const attrs: Record<string, string> = {}
  let internal: string | undefined

  const attrRegex = /(\w+):\s*((?:\[\[[^\]]+\]\]|[^,])+?(?:,\s*\[\[[^\]]+\]\])*)\s*(?:,|$)/g
  let match: RegExpExecArray | null

  while ((match = attrRegex.exec(attrsText)) !== null) {
    const key = match[1]?.trim()
    if (!key) continue
    const value = (match[2] ?? "").trim()
    attrs[key] = value
    if (key === "internal") {
      internal = value
    }
  }

  return { attrs, internal }
}

function serializeContent(nodes: HastContent[]): string {
  if (nodes.length === 0) return ""
  const root: HastRoot = { type: "root", children: nodes }
  return toHtml(root, { allowDangerousHtml: true }).trim()
}

function renderInternalLinks(internalRaw: string | undefined, file: VFile, ctx: BuildCtx): string {
  if (!internalRaw) return ""

  const links: HastContent[] = []
  const regex = createWikilinkRegex()
  let match: RegExpExecArray | null

  while ((match = regex.exec(internalRaw)) !== null) {
    const parsed = parseWikilink(match[0])
    if (!parsed) continue

    const resolved = resolveWikilinkTarget(parsed, file.data.slug as FullSlug)
    const anchor = parsed.anchor ?? ""
    let destination = parsed.target + anchor
    let dataSlug: string | undefined

    if (resolved) {
      dataSlug = resolved.slug
      const destWithAnchor = `${resolved.slug}${anchor}`
      destination = transformLink(file.data.slug as FullSlug, destWithAnchor, {
        allSlugs: ctx.allSlugs,
        strategy: "absolute",
      })
    } else if (!isAbsoluteUrl(destination)) {
      destination = transformLink(file.data.slug as FullSlug, destination, {
        allSlugs: ctx.allSlugs,
        strategy: "absolute",
      })
    }

    const display =
      parsed.alias ||
      path.basename(parsed.target, path.extname(parsed.target)) ||
      parsed.target ||
      "link"

    const link = h(
      "a",
      {
        href: destination,
        className: ["internal"],
        ...(dataSlug ? { "data-slug": dataSlug } : {}),
      },
      [h("span.indicator-hook"), display],
    )

    links.push(link)
  }

  if (links.length === 0) return ""

  const separator = h("hr.sidenote-separator", { className: "sidenote-separator" })
  const interleaved: HastContent[] = []
  links.forEach((link, idx) => {
    if (idx > 0) {
      interleaved.push({ type: "text", value: ", " } as HastText)
    }
    interleaved.push(link)
  })

  const container = h("div.sidenote-linked-notes", ["linked notes: ", ...interleaved])

  return toHtml({ type: "root", children: [separator, container] }, { allowDangerousHtml: true })
}

function combineContent(contentHtml: string, internalHtml: string): string {
  if (!internalHtml) return contentHtml
  if (!contentHtml) return internalHtml
  return `${contentHtml}${internalHtml}`
}

function deriveLabel(rawLabel: string): string {
  let labelText = rawLabel.trim()
  if (!labelText) return ""

  const wikilinkMatch = SINGLE_WIKILINK_REGEX.exec(labelText)
  if (wikilinkMatch) {
    const parsed = parseWikilink(wikilinkMatch[0])
    if (parsed) {
      labelText =
        parsed.alias || path.basename(parsed.target, path.extname(parsed.target)) || parsed.target
    }
  }
  SINGLE_WIKILINK_REGEX.lastIndex = 0

  return labelText
}

function buildReplacementNodes(
  labelText: string,
  attrs: Record<string, string>,
  combinedHtml: string,
  sidenoteId: number,
  tailText: string,
): HastContent[] {
  const arrowDownSvg = h(
    "svg.sidenote-arrow.sidenote-arrow-down",
    {
      width: "8",
      height: "5",
      viewBox: "0 0 8 5",
      xmlns: "http://www.w3.org/2000/svg",
      "aria-hidden": "true",
    },
    [
      h("path", {
        d: "M0 0L8 0L4 5Z",
        fill: "currentColor",
      }),
    ],
  )

  const hasLabel = labelText.length > 0
  const labelElement = h(
    "span.sidenote-label",
    hasLabel ? {} : { "data-auto": "" },
    hasLabel
      ? [{ type: "text", value: labelText } as HastText, arrowDownSvg]
      : [{ type: "text", value: "▪" } as HastText, arrowDownSvg],
  )

  const dataAttrs: Record<string, string> = {
    "data-content": combinedHtml,
    "data-sidenote-id": String(sidenoteId),
  }

  if (attrs.dropdown === "true" || attrs.inline === "true") {
    dataAttrs["data-force-inline"] = "true"
  }
  if (attrs.left === "false") {
    dataAttrs["data-allow-left"] = "false"
  }

  const sidenoteElement = h("span.sidenote", dataAttrs, [labelElement])
  const replacements: HastContent[] = [sidenoteElement]

  if (tailText.length > 0) {
    replacements.push({ type: "text", value: tailText } as HastText)
  }

  return replacements
}
