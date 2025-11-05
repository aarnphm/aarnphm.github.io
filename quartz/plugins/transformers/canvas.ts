import { QuartzTransformerPlugin } from "../types"
import { Element } from "hast"
import { visit } from "unist-util-visit"
import { parseJsonCanvas } from "./jcast"
import { JcastCanvas, isJcastCanvasNode } from "./jcast/types"
import { visitJcast } from "./jcast/visitor"
import { slugifyFilePath, slugAnchor, FilePath } from "../../util/path"
import { h } from "hastscript"
import { BuildCtx } from "../../util/ctx"
import { QuartzPluginData } from "../vfile"
import fs from "fs/promises"
import path from "path"
import { glob } from "../../util/glob"

export interface CanvasOptions {
  /**
   * Enable embedding canvases in markdown via ![[file.canvas]]
   */
  enableEmbeds: boolean

  /**
   * Resolve file nodes to actual markdown files
   */
  resolveFileNodes: boolean

  /**
   * Default width for embedded canvases
   */
  defaultWidth: number

  /**
   * Default height for embedded canvases
   */
  defaultHeight: number
}

const defaultOptions: CanvasOptions = {
  enableEmbeds: true,
  resolveFileNodes: true,
  defaultWidth: 800,
  defaultHeight: 600,
}

async function writeCanvasAsset(
  ctx: BuildCtx,
  slug: string,
  suffix: string,
  payload: unknown,
): Promise<string> {
  const normalizedSlug = slug.replace(/\\/g, "/")
  const relativePath = `${normalizedSlug}${suffix}`
  const outputPath = path.join(ctx.argv.output, relativePath)
  await fs.mkdir(path.dirname(outputPath), { recursive: true })
  const data = typeof payload === "string" ? payload : JSON.stringify(payload)
  await fs.writeFile(outputPath, data, "utf-8")
  return `/${relativePath}`
}

async function writeCanvasMetaAsset(
  ctx: BuildCtx,
  slug: string,
  meta: Record<string, any>,
): Promise<string> {
  return writeCanvasAsset(ctx, slug, ".meta.json", meta)
}

function collectCanvasMeta(jcast: JcastCanvas): Record<string, any> {
  const meta: Record<string, any> = {}
  for (const [id, node] of jcast.data.nodeMap) {
    if (node.data?.resolved) {
      meta[id] = node.data.resolved
    }
  }
  return meta
}

export const JsonCanvas: QuartzTransformerPlugin<Partial<CanvasOptions>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "Canvas",
    markdownPlugins() {
      return [
        () => {
          return (tree, _file) => {
            visit(tree, "wikilink", (node: any) => {
              const wikilink = node.data?.wikilink
              if (!wikilink?.target) return

              if (!wikilink.target.endsWith(".canvas")) return

              const hProperties: Record<string, any> = {
                "data-canvas-file": wikilink.target,
                "data-canvas-anchor": wikilink.anchor,
                "data-canvas-title": wikilink.target,
              }
              if (wikilink.embed) {
                hProperties["data-canvas-embed"] = true
              }

              node.data = { ...node.data, hName: "div", hProperties }
            })
          }
        },
      ]
    },
    htmlPlugins(ctx: BuildCtx) {
      // build a map of canvas slugs to file paths
      let canvasFileMap: Map<string, string> | null = null

      return [
        () => {
          return async (tree, _file) => {
            if (!canvasFileMap) {
              canvasFileMap = new Map()
              const canvasFiles = await glob("**/*.canvas", ctx.argv.directory, [
                ...ctx.cfg.configuration.ignorePatterns,
              ])

              for (const canvasFile of canvasFiles) {
                // convert file path to slug (same as emitter does)
                const slug = slugifyFilePath(canvasFile as any, true)
                const fullPath = path.join(ctx.argv.directory, canvasFile)
                canvasFileMap.set(slug, fullPath)
              }
            }

            const canvasEmbeds: Array<{ node: Element; target: string; title?: string }> = []

            visit(tree, "element", (node: Element) => {
              if (node.properties?.["data-canvas-embed"] === true) {
                canvasEmbeds.push({
                  node,
                  target: node.properties["data-canvas-file"] as string,
                  title: node.properties["data-canvas-title"] as string | undefined,
                })
              }
            })

            for (const { node, target, title } of canvasEmbeds) {
              try {
                let targetPath = target
                if (!targetPath.endsWith(".canvas")) {
                  targetPath = targetPath + ".canvas"
                }

                const slug = slugifyFilePath(targetPath as FilePath, true)
                const canvasPath = canvasFileMap.get(slug)

                if (!canvasPath) {
                  console.warn(`Canvas embed not found: ${target} (slug: ${slug})`)
                  continue
                }

                const canvasContent = await fs.readFile(canvasPath, "utf-8")
                const jcast = await processCanvasFile(canvasContent, ctx, undefined)
                const meta = collectCanvasMeta(jcast)
                const metaPath = await writeCanvasMetaAsset(ctx, slug, meta)

                const embedConfig = {
                  drag: true,
                  zoom: true,
                  forceStrength: 0.3,
                  linkDistance: 150,
                  collisionRadius: 50,
                  useManualPositions: true,
                  showInlineContent: false,
                  showPreviewOnHover: true,
                  previewMaxLength: 300,
                }

                node.tagName = "div"
                node.properties = {
                  class: "canvas-embed-container",
                  "data-canvas": `${slug}.canvas`,
                  "data-meta": metaPath,
                  "data-cfg": JSON.stringify(embedConfig),
                  "data-canvas-bounds": JSON.stringify(jcast.data.bounds),
                  "data-canvas-title": title || path.basename(canvasPath, ".canvas"),
                  style: `width: ${opts.defaultWidth}px; height: ${opts.defaultHeight}px;`,
                }
                node.children = [
                  {
                    type: "element",
                    tagName: "div",
                    properties: { class: "canvas-loading" },
                    children: [{ type: "text", value: "Loading canvas..." }],
                  } as Element,
                ]
              } catch (error) {
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
          }
        },
      ]
    },
  }
}

export async function processCanvasFile(
  canvasContent: string,
  ctx: BuildCtx,
  allFiles?: QuartzPluginData[],
): Promise<JcastCanvas> {
  const jcast = parseJsonCanvas(canvasContent)

  // resolve file nodes
  visitJcast(jcast, "canvasNode", (node) => {
    if (!isJcastCanvasNode(node)) return

    const canvasNode = node.data?.canvas
    if (!canvasNode || canvasNode.type !== "file") return

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

    // find the actual file to get its description and content
    const targetFile = allFiles?.find((f) => f.slug === slug)
    const description = targetFile?.frontmatter?.description || ""

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
        description,
      }
    }
  })

  return jcast
}

/**
 * Render jcast to hast element for embedding
 */
export async function renderCanvasToHast(
  ctx: BuildCtx,
  slug: string,
  jcast: JcastCanvas,
  options?: {
    width?: number
    height?: number
    title?: string
  },
): Promise<Element> {
  const { title } = options || {}

  // collect resolved metadata per node (do not mutate raw canvas JSON)
  const meta = collectCanvasMeta(jcast)
  const metaPath = await writeCanvasMetaAsset(ctx, slug, meta)

  // default canvas configuration
  const defaultConfig = {
    drag: true,
    zoom: true,
    forceStrength: 0.3,
    linkDistance: 150,
    collisionRadius: 50,
    useManualPositions: true,
    showInlineContent: false,
    showPreviewOnHover: true,
    previewMaxLength: 300,
  }

  const container = h(
    "div.canvas-container",
    {
      "data-canvas": `${slug}.canvas`,
      "data-meta": metaPath,
      "data-cfg": JSON.stringify(defaultConfig),
      "data-canvas-bounds": JSON.stringify(jcast.data.bounds),
      "data-canvas-title": title || "",
      style: `position: relative;`,
    },
    [h("div.canvas-loading", "Loading canvas...")],
  )

  return container as Element
}

declare module "vfile" {
  interface DataMap {
    canvas?: JcastCanvas
  }
}
