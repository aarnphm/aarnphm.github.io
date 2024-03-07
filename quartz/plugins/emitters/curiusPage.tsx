import { QuartzEmitterPlugin } from "../types"
import {
  CuriusContent,
  Navigation as NavigationConstructor,
  Meta as MetaConstructor,
  CuriusHeader,
  CuriusTrail,
  CuriusFriends,
  DesktopOnly,
} from "../../components"
import BodyConstructor from "../../components/Body"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, FullSlug, pathToRoot } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import DepGraph from "../../depgraph"
import { StaticResources } from "../../util/resources"
import { sharedPageComponents } from "../../../quartz.layout"

export const CuriusPage: QuartzEmitterPlugin = () => {
  const Meta = MetaConstructor({ enableSearch: false })

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    beforeBody: [CuriusHeader(), Meta],
    left: [CuriusFriends()],
    right: [DesktopOnly(CuriusTrail())],
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
    async getDependencyGraph(_ctx, _content, _resources) {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const cfg = ctx.cfg.configuration
      let componentData: QuartzComponentProps | undefined = undefined
      let externalResources: StaticResources | undefined = undefined
      let slug: FullSlug | undefined

      for (const [tree, file] of content) {
        slug = file.data.slug!
        if (slug === "curius") {
          externalResources = pageResources(pathToRoot(slug), resources)
          componentData = {
            ctx,
            fileData: file.data,
            externalResources,
            cfg,
            children: [],
            tree,
            allFiles: [],
          }
          break
        }
      }

      return [
        await write({
          ctx,
          content: renderPage(cfg, slug!, componentData!, opts, externalResources!),
          slug: slug ?? ("curius" as FullSlug),
          ext: ".html",
        }),
      ]
    },
  }
}
