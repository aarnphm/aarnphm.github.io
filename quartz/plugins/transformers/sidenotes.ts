import { QuartzTransformerPlugin } from "../types"
import { h } from "hastscript"
import { toHtml } from "hast-util-to-html"
import { toHast } from "mdast-util-to-hast"
import { RootContent, Html, Root as MdastRoot, Text } from "mdast"
import { toString } from "mdast-util-to-string"
import isAbsoluteUrl from "is-absolute-url"
import path from "path"
import { BuildCtx } from "../../util/ctx"
import { transformLink } from "../../util/path"
import { Parent } from "unist"
import { visit } from "unist-util-visit"
import { VFile } from "vfile"
// @ts-ignore
import script from "../../components/scripts/sidenotes.inline"
import content from "../../components/styles/sidenotes.inline.scss"
import {
  createWikilinkRegex,
  singleWikilinkRegex as SINGLE_WIKILINK_REGEX,
} from "../../util/wikilinks"

const SIDENOTE_PREFIX = "{{sidenotes"
const SIDENOTE_SUFFIX = "}}"

type ParentWithChildren = Parent & { children: RootContent[] }

type Token = { kind: "text"; value: string } | { kind: "node"; node: RootContent }

type ExtractResult = {
  removalCount: number
  replacements: RootContent[]
}

type CollectedTokens = {
  tokens: Token[]
  endIndex: number
  tailText: string
}

type ParsedTokens = {
  attrsText?: string
  labelText: string
  contentNodes: RootContent[]
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
    markdownPlugins(ctx) {
      return [
        () => (tree: MdastRoot, file: VFile) => {
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
              if (child.type !== "text") {
                continue
              }

              const textNode = child as Text
              let value = textNode.value
              const markerIndex = value.indexOf(SIDENOTE_PREFIX)
              if (markerIndex === -1) {
                continue
              }

              if (markerIndex > 0) {
                const before: Text = { type: "text", value: value.slice(0, markerIndex) }
                const after: Text = { type: "text", value: value.slice(markerIndex) }
                parent.children.splice(index, 1, before, after)
                index++
              }

              const extraction = extractSidenote(parent, index, file, ctx, () => ++counter)
              if (!extraction) {
                continue
              }

              parent.children.splice(index, extraction.removalCount, ...extraction.replacements)
              index += extraction.replacements.length - 1
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
): ExtractResult | null {
  const collected = collectTokens(parent, startIndex)
  if (!collected) {
    return null
  }

  const parsed = parseTokens(collected.tokens)
  if (!parsed) {
    return null
  }

  const { attrsText, labelText: rawLabel, contentNodes } = parsed

  const { attrs, internal } = parseAttributes(attrsText)
  const contentHtml = contentNodesToHtml(contentNodes)
  const internalHtml = renderInternalLinks(internal, file, ctx)
  const combinedHtml = internalHtml ? contentHtml + internalHtml : contentHtml

  let labelText = rawLabel.trim()
  const labelMatch = SINGLE_WIKILINK_REGEX.exec(labelText)
  if (labelMatch) {
    const target = (labelMatch[1] ?? "").trim()
    const alias = labelMatch[3]?.trim()
    labelText = alias || path.basename(target, path.extname(target)) || target
  } else {
    labelText = labelText.replace(/\[\[|\]\]/g, "").trim()
  }
  SINGLE_WIKILINK_REGEX.lastIndex = 0

  const sidenoteId = nextId()
  const sidenoteHtml = createSidenoteHtml(labelText, attrs, combinedHtml, sidenoteId)

  const replacements: RootContent[] = [{ type: "html", value: sidenoteHtml } as Html]
  if (collected.tailText.length > 0) {
    replacements.push({ type: "text", value: collected.tailText })
  }

  return {
    removalCount: collected.endIndex - startIndex + 1,
    replacements,
  }
}

function collectTokens(parent: ParentWithChildren, startIndex: number): CollectedTokens | null {
  const tokens: Token[] = []

  const startNode = parent.children[startIndex]
  if (!startNode || startNode.type !== "text") {
    return null
  }

  const startValue = (startNode as Text).value
  if (!startValue.startsWith(SIDENOTE_PREFIX)) {
    return null
  }

  let tailText = ""
  let endIndex = startIndex

  for (let idx = startIndex; idx < parent.children.length; idx++) {
    const node = parent.children[idx]

    if (node.type === "text") {
      const value = (node as Text).value
      let offset = idx === startIndex ? SIDENOTE_PREFIX.length : 0

      if (offset > value.length) {
        return null
      }

      while (offset <= value.length) {
        const closeIndex = value.indexOf(SIDENOTE_SUFFIX, offset)
        if (closeIndex === -1) {
          const segment = value.slice(offset)
          if (segment.length > 0) {
            tokens.push({ kind: "text", value: segment })
          }
          break
        } else {
          const segment = value.slice(offset, closeIndex)
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
  if (tokens.length === 0) {
    return null
  }

  let state: "header" | "content" = "header"
  let inLabel = false
  let inAngle = false
  let attrsText = ""
  const labelParts: string[] = []
  const contentNodes: RootContent[] = []

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

          if (inAngle) {
            attrsText += char
            index++
            continue
          }

          if (char === "[" && !inLabel) {
            inLabel = true
            index++
            continue
          }

          if (char === "]" && inLabel) {
            inLabel = false
            index++
            continue
          }

          if (inLabel) {
            labelParts.push(char)
            index++
            continue
          }

          if (char === ":") {
            state = "content"
            const remainder = value.slice(index + 1)
            if (remainder.length > 0) {
              contentNodes.push({ type: "text", value: remainder })
            }
            break
          }

          index++
        }
      } else {
        if (inLabel) {
          labelParts.push(toString(token.node))
        } else if (inAngle) {
          attrsText += toString(token.node)
        }
      }
    } else {
      if (token.kind === "text") {
        if (token.value.length > 0) {
          contentNodes.push({ type: "text", value: token.value })
        }
      } else {
        contentNodes.push(token.node)
      }
    }
  }

  if (state !== "content") {
    return null
  }

  return {
    attrsText: attrsText.trim() || undefined,
    labelText: labelParts.join(""),
    contentNodes,
  }
}

function parseAttributes(attrsText?: string): { attrs: Record<string, string>; internal?: string } {
  if (!attrsText) {
    return { attrs: {} }
  }

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

function contentNodesToHtml(nodes: RootContent[]): string {
  if (nodes.length === 0) {
    return ""
  }

  const root: MdastRoot = { type: "root", children: nodes }
  const hast = toHast(root, { allowDangerousHtml: true })
  return toHtml(hast, { allowDangerousHtml: true })
}

function renderInternalLinks(internalRaw: string | undefined, file: VFile, ctx: BuildCtx): string {
  if (!internalRaw) {
    return ""
  }

  const links: string[] = []
  let match: RegExpExecArray | null
  const regex = createWikilinkRegex()

  while ((match = regex.exec(internalRaw)) !== null) {
    const target = (match[1] ?? "").trim()
    if (!target) continue
    const anchor = match[2] ? `#${match[2].trim()}` : ""
    const alias = match[3]?.trim()

    let destination = `${target}${anchor}`
    const ext = path.extname(destination).toLowerCase()
    if (!isAbsoluteUrl(destination) && !ext.includes("pdf")) {
      destination = transformLink(file.data.slug!, destination, {
        allSlugs: ctx.allSlugs,
        strategy: "absolute",
      })
    }

    const display = alias || path.basename(target, path.extname(target)) || target
    links.push(toHtml(h("a", { href: destination }, display)))
  }

  if (links.length === 0) {
    return ""
  }

  return `<hr class="sidenote-separator" /><div class="sidenote-linked-notes">linked notes: ${links.join(
    ", ",
  )}</div>`
}

function createSidenoteHtml(
  labelText: string,
  attrs: Record<string, string>,
  contentHtml: string,
  sidenoteId: number,
): string {
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
      ? [{ type: "text", value: labelText }, arrowDownSvg]
      : [{ type: "text", value: "▪" }, arrowDownSvg],
  )

  const dataAttrs: Record<string, string> = {
    "data-content": contentHtml,
    "data-sidenote-id": sidenoteId.toString(),
  }

  if (attrs.dropdown === "true" || attrs.inline === "true") {
    dataAttrs["data-force-inline"] = "true"
  }
  if (attrs.left === "false") {
    dataAttrs["data-allow-left"] = "false"
  }

  return toHtml(h("span.sidenote", dataAttrs, [labelElement]), {
    allowDangerousHtml: true,
  })
}
