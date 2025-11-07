import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot, simplifySlug, SimpleSlug, slugifyFilePath, FilePath } from "../../util/path"
import { h } from "hastscript"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import { write } from "./helpers"
import path from "path"
import { collectCanvasMeta } from "../transformers/canvas"
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
    async *partialEmit() {},
    async *emit(ctx, content, resources) {
      const { cfg } = ctx

      const allFiles = content.map((c) => c[1].data)

      for (const [_tree, file] of content) {
        // Only process files marked as canvas files
        if (!file.data.jsonCanvas || !file.data.canvas || !file.data.canvasContent) continue

        const slug = slugifyFilePath(file.data.relativePath!, true)
        const jcast = file.data.canvas

        const jscastMetadata = collectCanvasMeta(jcast)

        yield write({ ctx, content: JSON.stringify(jscastMetadata), slug, ext: ".meta.json" })

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

        const canvasElement = h(
          "div.canvas-container",
          {
            "data-canvas": file.data.slug,
            "data-meta": `${slug}.meta.json`,
            "data-cfg": JSON.stringify(defaultConfig),
            "data-canvas-bounds": JSON.stringify(jcast.data.bounds),
            "data-canvas-title": path.basename(file.data.filePath!, ".canvas"),
            style: `position: relative;`,
          },
          [h("div.canvas-loading", "Loading canvas...")],
        )

        const tree: Root = {
          type: "root",
          children: [canvasElement],
        }

        // use links already extracted by transformer
        const linkedSlugs = (file.data.links ?? []).map(simplifySlug) as SimpleSlug[]

        const fileData: QuartzPluginData = {
          ...file.data,
          slug,
          frontmatter: {
            title: `${slug}.canvas`,
            tags: ["canvas"],
            pageLayout: "default" as any,
          },
          htmlAst: tree,
          text: file.data.text ?? "",
          links: linkedSlugs,
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

        // write HTML page to output
        yield write({ ctx, content, slug, ext: ".html" })
      }
    },
  }
}
