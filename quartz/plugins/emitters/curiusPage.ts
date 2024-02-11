import { QuartzEmitterPlugin } from "../types"
import { sharedPageComponents } from "../../../quartz.layout"
import {
  CuriusContent,
  CuriusNotes,
  Navigation as NavigationConstructor,
  Meta as MetaConstructor,
  Spacer,
  CuriusHeader,
  Head,
} from "../../components"
import BodyConstructor from "../../components/Body"
import HeaderConstructor from "../../components/Header"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import path from "path"
import { FilePath, FullSlug } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import chalk from "chalk"
import { defaultProcessedContent } from "../vfile"
import DepGraph from "../../depgraph"

export const CuriusPage: QuartzEmitterPlugin = () => {
  const Meta = MetaConstructor({ enableSearch: false })

  const opts: FullPageLayout = {
    head: Head(),
    header: [],
    beforeBody: [CuriusHeader()],
    left: [CuriusNotes(), Meta],
    right: [],
    pageBody: CuriusContent(),
    footer: NavigationConstructor({ prev: "/quotes", next: "/books" }),
  }

  const { head, header, beforeBody, pageBody, left, right, footer: Footer } = opts
  const Body = BodyConstructor()

  return {
    name: "CuriusPage",
    getQuartzComponents() {
      return [head, ...header, Body, ...beforeBody, pageBody, ...left, ...right, Footer]
    },
    async getDependencyGraph(ctx, content, _resources) {
      // TODO implement
      return new DepGraph<FilePath>()
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
          content: renderPage(cfg, slug, componentData, opts, externalResources),
          slug,
          ext: ".html",
        }),
      ]
    },
  }
}
