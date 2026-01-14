import { QuartzEmitterPlugin } from "../../types/plugin"
import { QuartzComponentProps } from "../../types/component"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { FullSlug } from "../../util/path"
import { sharedPageComponents } from "../../../quartz.layout"
import { NotFound } from "../../components"
import { defaultProcessedContent } from "../vfile"
import { write } from "./helpers"
import { i18n } from "../../i18n"

export const NotFoundPage: QuartzEmitterPlugin = () => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: NotFound(),
    beforeBody: [],
    sidebar: [],
    afterBody: [],
  }

  const { head: Head, pageBody, footer: Footer } = opts
  return {
    name: "404Page",
    getQuartzComponents() {
      return [Head, pageBody, Footer]
    },
    async *emit(ctx, _content, resources) {
      const cfg = ctx.cfg.configuration
      const slug = "404" as FullSlug

      const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)
      const path = url.pathname as FullSlug
      const notFound = i18n(cfg.locale).pages.error.title
      const [tree, vfile] = defaultProcessedContent({
        slug,
        text: notFound,
        description: notFound,
        frontmatter: { title: notFound, tags: [], pageLayout: "default" },
      })
      const externalResources = pageResources(path, resources, ctx)
      const componentData: QuartzComponentProps = {
        ctx,
        fileData: vfile.data,
        externalResources,
        cfg,
        children: [],
        tree,
        allFiles: [],
      }

      yield write({
        ctx,
        content: renderPage(ctx, slug, componentData, opts, externalResources),
        slug,
        ext: ".html",
      })
    },
    async *partialEmit() {},
  }
}
