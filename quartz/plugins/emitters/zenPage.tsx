import { QuartzEmitterPlugin } from "../types"
import HeaderConstructor from "../../components/Header"
import HeadConstructor from "../../components/Head"
import BodyConstructor from "../../components/Body"
import MetaConstructor from "../../components/Meta"
import NavigationConstructor from "../../components/Navigation"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, pathToRoot } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import { ArticleTitle, Content, Keybind as KeybindConstructor } from "../../components"
import DepGraph from "../../depgraph"

export const ZenPage: QuartzEmitterPlugin = () => {
  const Meta = MetaConstructor()
  const Navigation = NavigationConstructor()
  const Keybind = KeybindConstructor({ enableTooltip: false })

  const opts: FullPageLayout = {
    head: HeadConstructor(),
    header: [],
    beforeBody: [],
    pageBody: Content(),
    left: [Meta],
    right: [],
    footer: Navigation,
  }

  const { head: Head, header, beforeBody, pageBody, left, right, footer: Footer } = opts
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
        Keybind,
        ...header,
        ...beforeBody,
        pageBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async getDependencyGraph(ctx, content, _resources) {
      // Example graph:
      // nested/file.md --> nested/file.html
      //          \-------> nested/index.html
      // TODO implement
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const fps: FilePath[] = []
      const cfg = ctx.cfg.configuration
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (file.data.frontmatter?.zen === true) {
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
