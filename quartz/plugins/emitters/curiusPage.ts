import { QuartzEmitterPlugin } from "../types"
import { sharedPageComponents } from "../../../quartz.layout"
import { CuriusContent, Spacer } from "../../components"
import BodyConstructor from "../../components/Body"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import path from "path"
import { FilePath, FullSlug } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import chalk from "chalk"
import { defaultProcessedContent } from "../vfile"

export const CuriusPage: QuartzEmitterPlugin = () => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    beforeBody: [],
    left: [],
    right: [],
    pageBody: CuriusContent(),
    footer: Spacer(),
  }

  const { head: Head, pageBody, footer: Footer } = opts
  const Body = BodyConstructor()

  return {
    name: "CuriusPage",
    getQuartzComponents() {
      return [Head, Body, pageBody, Footer]
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const cfg = ctx.cfg.configuration
      const slug = "curius" as FullSlug
      const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)
      const path = url.pathname as FullSlug
      const externalResources = pageResources(path, resources)
      const [tree, vfile] = defaultProcessedContent({
        slug,
        text: "Curius",
        description: "curius.app",
        frontmatter: { title: "Curius", description: "curius.app", tags: [] },
      })
      const componentData: QuartzComponentProps = {
        fileData: vfile.data,
        externalResources,
        cfg: cfg,
        children: [],
        tree,
        allFiles: [],
      }
      return [
        await write({
          ctx,
          content: renderPage(slug, componentData, opts, externalResources),
          slug,
          ext: ".html",
        }),
      ]
    },
  }
}
