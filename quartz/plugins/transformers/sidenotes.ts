import { QuartzTransformerPlugin } from "../../types/plugin"
import { Root as MdastRoot } from "mdast"
import { Element as HastElement, Text as HastText, ElementContent, Root as HastRoot } from "hast"
import { visit } from "unist-util-visit"
import type { Node } from "unist"
import { h } from "hastscript"
import { BuildCtx } from "../../util/ctx"
import { FullSlug, transformLink } from "../../util/path"
import { VFile } from "vfile"
import path from "path"
import isAbsoluteUrl from "is-absolute-url"
import { parseWikilink, resolveWikilinkTarget, createWikilinkRegex } from "../../util/wikilinks"
// @ts-ignore
import script from "../../components/scripts/sidenotes.inline"
import content from "../../components/styles/sidenotes.inline.scss"
import type {
  Sidenote,
  SidenoteReference,
} from "../../extensions/micromark-extension-ofm-sidenotes"

export const Sidenotes: QuartzTransformerPlugin = () => {
  return {
    name: "Sidenotes",
    externalResources() {
      return {
        js: [{ script, contentType: "inline", loadTime: "afterDOMReady" }],
        css: [{ content, spaPreserve: true, inline: true }],
      }
    },
    markdownPlugins() {
      return [
        () => (tree: MdastRoot, file: VFile) => {
          let counter = 0

          visit(tree, ["sidenote", "sidenoteReference"], (node: Sidenote | SidenoteReference) => {
            const sidenoteId = ++counter
            const baseId = buildSidenoteDomId(file, sidenoteId)

            if (node.type === "sidenote") {
              const parsed = node.data?.sidenoteParsed
              if (!parsed) return

              const props = parsed.properties || {}
              const forceInline = props.dropdown === "true" || props.inline === "true"
              const allowLeft = props.left !== "false"
              const allowRight = props.right !== "false"

              node.data = {
                ...node.data,
                sidenoteId,
                baseId,
                forceInline,
                allowLeft,
                allowRight,
                internal: props.internal,
                label: parsed.label || "",
              } as Record<string, any>
            } else if (node.type === "sidenoteReference") {
              node.data = { ...node.data, sidenoteId, baseId } as Record<string, any>
            }
          })
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => {
          return (tree: HastRoot, file: VFile) => {
            const definitions = new Map<string, ElementContent[]>()

            visit(tree, "element", (node, index, parent) => {
              if (node.properties?.dataType === "sidenote-def") {
                const label = node.properties.label as string
                const contentDiv = node.children[1] as HastElement
                if (contentDiv && label) {
                  definitions.set(label, contentDiv.children)
                }

                if (parent && typeof index === "number") {
                  parent.children.splice(index, 1)
                  return index
                }
              }
            })

            visit(
              tree,
              (node): node is Node =>
                node.type === "element" &&
                (node.properties?.dataType === "sidenote" ||
                  node.properties?.dataType === "sidenote-ref"),
              (node: Sidenote | SidenoteReference, index, parent) => {
                if (index === undefined || !parent) return

                const {
                  sidenoteId,
                  baseId,
                  forceInline,
                  allowLeft,
                  allowRight,
                  label: labelRaw,
                  internal: internalLinks,
                } = node.properties

                const labelContainer = node.children!.find((c: any) =>
                  c.properties?.className?.includes("sidenote-label-hast"),
                ) as HastElement | undefined

                const contentContainer = node.children!.find((c: any) =>
                  c.properties?.className?.includes("sidenote-content-hast"),
                ) as HastElement | undefined

                const labelHast = labelContainer?.children ?? []
                let contentHast: ElementContent[] = []

                if (node.properties.dataType === "sidenote-ref") {
                  const defContent = definitions.get(labelRaw)
                  if (defContent) {
                    contentHast = defContent
                  } else {
                    contentHast = [{ type: "text", value: "[Missing definition]" }]
                  }
                } else {
                  contentHast = contentContainer?.children ?? []
                }

                const internal = renderInternalLinks(internalLinks, file, ctx)
                const combinedContent = [...contentHast, ...internal]

                const finalLabel = labelHast.length > 0 ? labelHast : deriveLabel(labelRaw)

                parent.children.splice(
                  index,
                  1,
                  ...buildSidenoteHast(
                    finalLabel,
                    forceInline === true,
                    allowLeft !== false,
                    allowRight !== false,
                    combinedContent,
                    sidenoteId as number,
                    baseId as string,
                  ),
                )
                return index
              },
            )
          }
        },
      ]
    },
  }
}

function renderInternalLinks(
  wikilinks: string[] | undefined,
  file: VFile,
  ctx: BuildCtx,
): ElementContent[] {
  if (!wikilinks || wikilinks.length === 0) return []

  const links: HastElement[] = []
  const regex = createWikilinkRegex()

  for (const wl of wikilinks) {
    let match: RegExpExecArray | null
    while ((match = regex.exec(wl)) !== null) {
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
      } else if (!isAbsoluteUrl(destination, { httpOnly: false })) {
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
  }

  if (links.length === 0) return []

  const separator = h("span.sidenote-separator", {
    role: "presentation",
  })

  const interleaved: ElementContent[] = []
  links.forEach((link, idx) => {
    if (idx > 0) {
      interleaved.push({ type: "text", value: ", " } as HastText)
    }
    interleaved.push(link)
  })

  const container = h("span.sidenote-linked-notes", ["linked notes: ", ...interleaved])

  return [separator, container]
}

function deriveLabel(rawLabel: string): ElementContent[] {
  let labelText = rawLabel.trim()
  if (!labelText) return []

  const wikilinkMatch = /\[\[([^\]]+)\]\]/.exec(labelText)
  if (wikilinkMatch) {
    const parsed = parseWikilink(wikilinkMatch[0])
    if (parsed) {
      labelText =
        parsed.alias || path.basename(parsed.target, path.extname(parsed.target)) || parsed.target
    }
  }

  return [{ type: "text", value: labelText }]
}

function buildSidenoteHast(
  label: ElementContent[],
  forceInline: boolean,
  allowLeft: boolean,
  allowRight: boolean,
  combinedContent: ElementContent[],
  sidenoteId: number,
  baseId: string,
): HastElement[] {
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

  const hasLabel = label.length > 0

  const labelProps: Record<string, string> = {
    id: `${baseId}-label`,
    "aria-controls": `${baseId}-content`,
  }

  const labelElement = h(
    "span.sidenote-label",
    hasLabel ? labelProps : { ...labelProps, "data-auto": "" },
    hasLabel ? [...label, arrowDownSvg] : [{ type: "text", value: "▪" } as HastText, arrowDownSvg],
  )

  const dataAttrs: Record<string, string> = {
    id: baseId,
    "data-sidenote-id": String(sidenoteId),
  }

  if (forceInline) dataAttrs["data-force-inline"] = "true"
  if (!allowLeft) dataAttrs["data-allow-left"] = "false"
  if (!allowRight) dataAttrs["data-allow-right"] = "false"

  const sidenoteElement = h("span.sidenote", dataAttrs, [labelElement])

  const contentElement = h(
    "span.sidenote-content",
    {
      id: `${baseId}-content`,
      "data-sidenote-id": String(sidenoteId),
      "data-sidenote-for": baseId,
      "aria-hidden": "true",
    },
    combinedContent.length > 0
      ? combinedContent
      : ([{ type: "text", value: "" }] as ElementContent[]),
  )

  return [sidenoteElement, contentElement]
}

function buildSidenoteDomId(file: VFile, sidenoteId: number): string {
  const rawSlug = (file.data.slug as string | undefined) ?? "note"
  const sanitized = rawSlug.replace(/[^A-Za-z0-9_-]/g, "-")
  return `sidenote-${sanitized}-${sidenoteId}`
}
