import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import {
  FilePath,
  joinSegments,
  slugifyFilePath,
  pathToRoot,
  FullSlug,
  simplifySlug,
  SimpleSlug,
} from "../../util/path"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import { write } from "./helpers"
import { glob } from "../../util/glob"
import fs from "node:fs/promises"
import path from "path"
import { processCanvasFile, renderCanvasToHast } from "../transformers/canvas"
import { Root } from "hast"
import { QuartzPluginData } from "../vfile"
import CanvasComponent from "../../components/Canvas"

export const CanvasPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const header = sharedPageComponents.header.filter((component) => {
    const name = component.displayName || component.name || ""
    return name !== "Breadcrumbs" && name !== "StackedNotes"
  })

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    pageBody: Content(),
    header,
    beforeBody: [],
    afterBody: [CanvasComponent()],
  }

  const {
    head: Head,
    header: Header,
    beforeBody: BeforeBody,
    pageBody,
    afterBody,
    sidebar,
    footer: Footer,
  } = opts

  return {
    name: "CanvasPage",

    getQuartzComponents() {
      return [Head, ...Header, ...BeforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },

    async *emit(ctx, content, resources) {
      const { argv, cfg } = ctx

      // find all .canvas files
      const canvasFiles = await glob("**/*.canvas", argv.directory, [
        ...cfg.configuration.ignorePatterns,
      ])

      if (canvasFiles.length === 0) {
        return
      }

      const allFiles = content.map((c) => c[1].data)

      for (const canvasFile of canvasFiles) {
        try {
          const src = joinSegments(argv.directory, canvasFile) as FilePath
          const slug = slugifyFilePath(canvasFile as FilePath, true) as FullSlug

          const canvasContent = await fs.readFile(src, "utf-8")
          const jcast = await processCanvasFile(canvasContent, ctx, allFiles, slug)

          const bounds = jcast.data.bounds
          const width = bounds ? bounds.maxX - bounds.minX + 200 : 1200 // add padding
          const height = bounds ? bounds.maxY - bounds.minY + 200 : 800

          const title = path.basename(canvasFile, ".canvas")

          const canvasElement = await renderCanvasToHast(ctx, slug, jcast, { width, height, title })
          const canvasDataPath = (canvasElement.properties?.["data-canvas"] as string) || undefined
          const canvasMetaPath = (canvasElement.properties?.["data-meta"] as string) || undefined

          const tree: Root = {
            type: "root",
            children: [canvasElement],
          }

          const linkedSlugs: string[] = []
          for (const [, node] of jcast.data.nodeMap) {
            if (node.data?.resolved?.slug) {
              linkedSlugs.push(simplifySlug(node.data.resolved.slug as FullSlug))
            }

            if (node.data?.wikilinks) {
              for (const link of node.data.wikilinks) {
                if (link.resolvedSlug) {
                  linkedSlugs.push(simplifySlug(link.resolvedSlug as FullSlug))
                }
              }
            }
          }

          const fileData: QuartzPluginData = {
            slug,
            frontmatter: {
              title,
              tags: ["canvas"],
              pageLayout: "default" as any,
              ...(canvasDataPath ? { canvas: canvasDataPath } : {}),
              ...(canvasMetaPath ? { canvasMeta: canvasMetaPath } : {}),
            },
            htmlAst: tree,
            filePath: canvasFile as FilePath,
            text: "",
            links: linkedSlugs as SimpleSlug[],
          }

          const externalResources = pageResources(pathToRoot(slug), resources, ctx)
          const componentData: QuartzComponentProps = {
            ctx,
            fileData,
            externalResources,
            cfg: cfg.configuration,
            children: [],
            tree,
            allFiles,
          }

          const content = renderPage(ctx, slug, componentData, opts, externalResources, false)

          // write to output
          yield write({ ctx, content, slug, ext: ".html" })
        } catch (error) {
          console.error(`Failed to process canvas ${canvasFile}:`, error)
        }
      }
    },
  }
}
