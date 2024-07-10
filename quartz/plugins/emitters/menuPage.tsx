import { QuartzEmitterPlugin } from "../types"
import {
  Meta as MetaConstructor,
  ContentMeta,
  Content as ContentConstructor,
  ArticleTitle,
  Spacer,
} from "../../components"
import BodyConstructor from "../../components/Body"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, FullSlug, pathToRoot } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../../components/types"
import DepGraph from "../../depgraph"
import { sharedPageComponents } from "../../../quartz.layout"
import { classNames } from "../../util/lang"

const MenuContent = (() => {
  const Content = ContentConstructor()

  const Element: QuartzComponent = (props: QuartzComponentProps) => {
    const { displayClass } = props
    return (
      <>
        <div class={classNames(displayClass, "menu")}>
          <Content {...props} />
          <hr />
          <MenuFooter {...props} />
        </div>
      </>
    )
  }

  return Element
}) satisfies QuartzComponentConstructor

function MenuFooter({ allFiles, fileData, displayClass, cfg }: QuartzComponentProps) {
  return (
    <footer class={classNames(displayClass, "menu-footer")}>
      <a href="../atelier-with-friends" class="internal alias" data-no-popover={true}>
        atelier with friends.
      </a>
    </footer>
  )
}

export const MenuPage: QuartzEmitterPlugin = () => {
  const Meta = MetaConstructor()

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    beforeBody: [Meta],
    left: [],
    right: [],
    pageBody: MenuContent(),
    footer: Spacer(),
  }

  const { head, header, beforeBody, pageBody, afterBody, left, right, footer: Footer } = opts
  const Body = BodyConstructor()

  return {
    name: "MenuPage",
    getQuartzComponents() {
      return [
        head,
        ...header,
        Body,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async getDependencyGraph(_ctx, _content, _resources) {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const fps: FilePath[] = []
      const cfg = ctx.cfg.configuration
      const allFiles = content.map((c) => c[1].data)

      let slug: FullSlug | undefined

      for (const [tree, file] of content) {
        slug = file.data.slug!
        if (file.data.frontmatter?.menu === true) {
          const externalResources = pageResources(pathToRoot(slug), resources)

          const componentData: QuartzComponentProps = {
            ctx,
            fileData: file.data,
            externalResources,
            cfg: cfg,
            children: [],
            tree,
            allFiles,
          }
          const content = renderPage(cfg, slug, componentData, opts, externalResources)

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
