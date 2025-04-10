import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, FullSlug, pathToRoot } from "../../util/path"
import { classNames } from "../../util/lang"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps, QuartzComponent } from "../../components/types"
import { ArticleTitle, ContentMeta } from "../../components"
import { Date, getDate } from "../../components/Date"
import { StaticResources } from "../../util/resources"
import { sharedPageComponents, defaultContentPageLayout } from "../../../quartz.layout"

import style from "../../components/styles/infinitePoem.scss"
//@ts-ignore
import infiniteScript from "../../components/scripts/infinite-poem.inline"

function InfiniteDate({ fileData, displayClass, cfg }: QuartzComponentProps) {
  return (
    <footer class={classNames(displayClass, "poetry-footer")}>
      <Date date={getDate(cfg, fileData)!} locale={cfg.locale} />
    </footer>
  )
}

function InfiniteContent() {
  const Infinite: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "infinite-poem", "popover-hint")} id="infinite-poem" />
    )
  }

  Infinite.css = style
  Infinite.afterDOMLoaded = infiniteScript

  return Infinite
}

export const InfinitePoemPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    sidebar: [],
    beforeBody: [ArticleTitle(), ContentMeta()],
    afterBody: [InfiniteDate],
    pageBody: InfiniteContent(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts

  return {
    name: "InfinitePage",
    getQuartzComponents() {
      return [Head, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const cfg = ctx.cfg.configuration
      let componentData: QuartzComponentProps | undefined = undefined
      let externalResources: StaticResources | undefined = undefined
      let slug: FullSlug | undefined

      for (const [tree, file] of content) {
        slug = file.data.slug!
        if (slug === "infinite-poem") {
          externalResources = pageResources(pathToRoot(slug), resources)

          componentData = {
            ctx,
            fileData: file.data,
            externalResources,
            cfg: cfg,
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
          slug: slug ?? ("infinite-poem" as FullSlug),
          ext: ".html",
        }),
      ]
    },
  }
}
