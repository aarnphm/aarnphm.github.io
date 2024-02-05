import { QuartzEmitterPlugin } from "../types"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import HeaderConstructor from "../../components/Header"
import BodyConstructor from "../../components/Body"
import MetaConstructor from "../../components/Meta"
import NavigationConstructor from "../../components/Navigation"
import Spacer from "../../components/Spacer"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import path from "path"
// @ts-ignore
import keybindScript from "../../components/scripts/keybind.inline"
import { FilePath, FullSlug, pathToRoot } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import { ArticleTitle, Content } from "../../components"
import chalk from "chalk"
import { defaultProcessedContent } from "../vfile"

interface Options {
  slug: string[]
}

const defaultOptions: Options = {
  slug: [],
}

export const ZenPage: QuartzEmitterPlugin<Partial<Options>> = (opts?: Partial<Options>) => {
  const { slug: zenSlug } = { ...defaultOptions, ...opts }
  const Meta = MetaConstructor()
  const Navigation = NavigationConstructor()

  const pageOpts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    beforeBody: [ArticleTitle()],
    pageBody: Content(),
    left: [Meta],
    right: [],
    footer: Navigation,
  }

  const { head: Head, header, beforeBody, pageBody, left, right, footer } = pageOpts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  return {
    name: "ZenPage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        Meta,
        Navigation,
        ...header,
        ...beforeBody,
        pageBody,
        ...left,
        ...right,
      ]
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const fps: FilePath[] = []
      const cfg = ctx.cfg.configuration
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (zenSlug.includes(slug)) {
          const externalResources = pageResources(pathToRoot(slug), resources)

          externalResources.js.push({
            loadTime: "beforeDOMReady",
            contentType: "inline",
            spaPreserve: true,
            script: keybindScript,
          })

          const componentData: QuartzComponentProps = {
            fileData: file.data,
            externalResources,
            cfg: cfg,
            children: [],
            tree,
            allFiles,
          }
          const content = renderPage(cfg, slug, componentData, pageOpts, externalResources)

          const fp = await write({
            ctx,
            content,
            slug,
            ext: ".html",
          })

          fps.push(fp)
        }
      }
      return fps
    },
  }
}
