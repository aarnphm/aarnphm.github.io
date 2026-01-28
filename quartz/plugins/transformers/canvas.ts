import path from "path"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../../types/plugin"
import { BuildCtx } from "../../util/ctx"
import { slugifyFilePath, slugAnchor, FilePath, FullSlug, resolveRelative } from "../../util/path"
import {
  createWikilinkRegex,
  extractWikilinks,
  resolveWikilinkTarget,
  Wikilink,
} from "../../util/wikilinks"
import { parseJsonCanvas } from "./jcast"
import { JcastCanvas, isJcastCanvasNode } from "./jcast/types"
import { visitJcast } from "./jcast/visitor"

export const JsonCanvas: QuartzTransformerPlugin = () => {
  return {
    name: "JsonCanvas",
    markdownPlugins() {
      return [
        () => {
          return (tree, file) => {
            // handle canvas wikilink embeds (for any file)
            visit(tree, "wikilink", (node: Wikilink) => {
              const wikilink = node.data?.wikilink
              if (!wikilink?.target) return

              if (!wikilink.target.endsWith(".canvas")) return
              if (wikilink.embed) {
                const hProperties: Record<string, any> = {
                  "data-canvas-file": wikilink.target,
                  "data-canvas-anchor": wikilink.anchor,
                  "data-canvas-title": wikilink.target,
                  "data-canvas-embed": true,
                  "data-slug": wikilink.target.replace(".canvas", ""),
                }
                node.data = { ...node.data!, hName: "div", hProperties }
              }
            })

            const isCanvasFile = file.path?.endsWith(".canvas")

            if (!isCanvasFile) {
              file.data.jsonCanvas = false
              return
            }

            // Mark as bases file
            file.data.jsonCanvas = true
            // clear tree to prevent markdown parsing of JSON content
            tree.children = []
          }
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => async (tree, file) => {
          const slug = file.data.slug as FullSlug

          if (file.data.jsonCanvas) {
            try {
              const canvasContent = String(file.value)
              const jcast = await processCanvasFile(canvasContent, ctx, slug)

              // store raw content for emitter to copy to output
              file.data.canvasContent = canvasContent

              // store processed canvas data
              file.data.canvas = jcast

              // extract links for graph
              const linkedSlugs: string[] = []
              for (const [, node] of jcast.data.nodeMap) {
                if (node.data?.resolved?.slug) {
                  linkedSlugs.push(node.data.resolved.slug)
                }
                if (node.data?.wikilinks) {
                  for (const link of node.data.wikilinks) {
                    if (link.resolvedSlug) {
                      linkedSlugs.push(link.resolvedSlug)
                    }
                  }
                }
              }
              file.data.links = linkedSlugs as any[]

              // extract searchable text
              const textContent: string[] = []
              const fileNodes: string[] = []
              for (const [, node] of jcast.data.nodeMap) {
                const canvasNode = node.data?.canvas
                if (!canvasNode) continue

                if (canvasNode.type === "text" && canvasNode.text) {
                  textContent.push(canvasNode.text)
                } else if (canvasNode.type === "file" && canvasNode.file) {
                  fileNodes.push(canvasNode.file)
                }
              }

              const searchableContent = [
                ...textContent,
                ...fileNodes.map((f) => path.basename(f, ".md")),
              ].join(" ")

              file.data.text = searchableContent
            } catch (error) {
              console.error(`Failed to process canvas ${file.data.slug}:`, error)
            }
            return
          }

          visit(tree, "element", (node) => {
            if (node.properties?.["data-canvas-embed"] === true) {
              try {
                const canvasFile = node.properties["data-canvas-file"]
                const canvasTitle = node.properties["data-canvas-title"] || canvasFile
                const canvasSlug = node.properties["data-slug"]

                // strip .canvas extension for slug and link
                const linkHref = canvasSlug || canvasFile?.replace(/\.canvas$/, "") || ""
                const resourceBase = linkHref.startsWith("/") ? linkHref : `/${linkHref}`

                // embed configuration: static preview, no interaction
                const embedConfig = {
                  drag: false,
                  zoom: false,
                  forceStrength: 0,
                  linkDistance: 150,
                  collisionRadius: 50,
                  useManualPositions: true,
                  showInlineContent: false,
                  showPreviewOnHover: false,
                  previewMaxLength: 0,
                }

                // transform into full canvas embed container
                node.tagName = "div"
                node.properties = {
                  class: "canvas-embed-container",
                  "data-canvas": `${resourceBase}.canvas`,
                  "data-meta": `${resourceBase}.meta.json`,
                  "data-cfg": JSON.stringify(embedConfig),
                  "data-canvas-title": canvasTitle,
                  "data-embed": "true",
                  "data-navigate-to": linkHref,
                  style:
                    "position: relative; width: 100%; height: clamp(400px, 50vh, 800px); cursor: pointer; margin: 2rem auto;",
                }

                // loading placeholder
                node.children = [
                  {
                    type: "element",
                    tagName: "div",
                    properties: { class: "canvas-loading" },
                    children: [{ type: "text", value: "loading canvas..." }],
                  },
                ]
              } catch (error) {
                const target =
                  node.properties?.["data-canvas-file"] || node.properties?.["data-canvas-title"]
                console.error(`Failed to embed canvas ${target}:`, error)
                node.tagName = "div"
                node.properties = {
                  class: "canvas-embed-error",
                  style:
                    "color: var(--gray); padding: 1em; border: 1px solid var(--lightgray); border-radius: 4px;",
                }
                node.children = [{ type: "text", value: `Failed to load canvas: ${target}` }]
              }
            }
          })
        },
      ]
    },
  }
}

export function collectCanvasMeta(jcast: JcastCanvas): Record<string, any> {
  const meta: Record<string, any> = {}
  for (const [id, node] of jcast.data.nodeMap) {
    if (!node.data) {
      continue
    }

    const entry: Record<string, any> = {}

    if (node.data.resolved) {
      entry.slug = node.data.resolved.slug
      entry.href = node.data.resolved.href
      entry.displayName = node.data.resolved.displayName
      if (node.data.resolved.description) {
        entry.description = node.data.resolved.description
      }
      if (node.data.resolved.content) {
        entry.content = node.data.resolved.content
      }
    }

    if (node.data.resolvedText) {
      entry.resolvedText = node.data.resolvedText
    }

    if (node.data.wikilinks && node.data.wikilinks.length > 0) {
      entry.wikilinks = node.data.wikilinks.map((w) => ({
        ...w.link,
        resolvedSlug: w.resolvedSlug,
        resolvedHref: w.resolvedHref,
        missing: Boolean(w.missing),
      }))
    }

    if (Object.keys(entry).length > 0) {
      meta[id] = entry
    }
  }

  return meta
}

export async function processCanvasFile(
  canvasContent: string,
  ctx: BuildCtx,
  currentSlug: FullSlug,
): Promise<JcastCanvas> {
  const jcast = parseJsonCanvas(canvasContent)

  // resolve file nodes
  visitJcast(jcast, "canvasNode", (node) => {
    if (!isJcastCanvasNode(node)) return

    const canvasNode = node.data?.canvas
    if (!canvasNode) return

    if (canvasNode.type === "file") {
      let filePath = canvasNode.file
      if (!filePath) return

      // Optional subpath (e.g., "#Heading" or "^block-id") provided by Obsidian Canvas
      // Obsidian stores anchors separately in `subpath`. In some cases, users may
      // have persisted the `#anchor` inside `file` – handle both robustly.
      let rawSubpath = canvasNode.subpath || ""
      if (!rawSubpath && filePath.includes("#")) {
        const idx = filePath.indexOf("#")
        rawSubpath = filePath.slice(idx)
        filePath = filePath.slice(0, idx)
      }

      // normalize and resolve file path following wikilink procedures
      let normalizedPath = filePath.trim()

      // ensure .md extension if not present and not already another extension
      if (!normalizedPath.includes(".")) {
        normalizedPath = normalizedPath + ".md"
      }

      // convert to slug using same logic as wikilinks
      const slug = slugifyFilePath(normalizedPath as FilePath)
      const fileExists = ctx.allSlugs?.includes(slug)

      // extract display name (just the filename without extension)
      const displayName = normalizedPath.split("/").pop()?.replace(/\.md$/, "") || filePath

      // Normalize subpath to an anchor Obsidian-style
      let resolvedAnchor = ""
      if (rawSubpath) {
        // rawSubpath could be "#Heading", "#Heading#Sub", or "^block-id" (with or without leading '#')
        let sp = rawSubpath.trim()
        // Ensure leading marker is present
        if (!sp.startsWith("#") && !sp.startsWith("^")) {
          sp = "#" + sp
        }
        if (sp.startsWith("^")) {
          // Block reference – store as #^id
          resolvedAnchor = `#${sp}`
        } else {
          // Heading anchor – may contain nested headings; follow Obsidian's last-segment behavior
          const withoutHash = sp.slice(1)
          const lastSegment = withoutHash.split("#").pop()!.trim()
          const slugged = slugAnchor(lastSegment)
          resolvedAnchor = `#${slugged}`
        }
      }

      if (node.data) {
        node.data.resolvedPath = slug
        node.data.fileExists = fileExists
        // keep normalized path in raw canvas data for round-tripping
        if (node.data.canvas) {
          node.data.canvas.file = normalizedPath
        }
        // structured resolved info
        node.data.resolved = {
          slug,
          href: resolvedAnchor ? `${slug}${resolvedAnchor}` : slug,
          displayName,
        }
      }
      return
    }

    if (canvasNode.type === "text") {
      const rawText = canvasNode.text ?? ""
      if (!rawText) {
        return
      }

      const wikilinks = extractWikilinks(rawText)
      if (wikilinks.length === 0) {
        return
      }

      const resolvedLinks = wikilinks.map((link) => {
        const resolved = resolveWikilinkTarget(link, currentSlug)
        if (!resolved) {
          return {
            link,
            missing: true,
          }
        }

        const baseHref = resolveRelative(currentSlug, resolved.slug)
        const href = resolved.anchor ? `${baseHref}${resolved.anchor}` : baseHref

        return {
          link,
          resolvedSlug: resolved.slug,
          resolvedHref: href,
          missing: false,
        }
      })

      const regex = createWikilinkRegex()
      let idx = 0

      const rewrittenText = rawText.replace(regex, (value) => {
        const info = resolvedLinks[idx++]
        if (!info) {
          return value
        }

        const display = (info.link.alias ?? info.link.target ?? value).trim()
        if (!info.resolvedHref) {
          return display.length > 0 ? display : info.link.raw
        }

        const label = display.length > 0 ? display : info.resolvedHref
        return `[${label}](${info.resolvedHref})`
      })

      if (node.data) {
        node.data.wikilinks = resolvedLinks
        node.data.resolvedText = rewrittenText
      }
    }
  })

  return jcast
}

declare module "vfile" {
  interface DataMap {
    canvas?: JcastCanvas
    jsonCanvas?: boolean
    canvasContent?: string
  }
}
