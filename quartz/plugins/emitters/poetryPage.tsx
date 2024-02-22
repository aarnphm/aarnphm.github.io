import { QuartzEmitterPlugin } from "../types"
import HeaderConstructor from "../../components/Header"
import HeadConstructor from "../../components/Head"
import BodyConstructor from "../../components/Body"
import MetaConstructor from "../../components/Meta"
import NavigationConstructor from "../../components/Navigation"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, pathToRoot } from "../../util/path"
import { classNames } from "../../util/lang"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentConstructor, QuartzComponentProps } from "../../components/types"
import { ArticleTitle, Content, ContentMeta, Spacer } from "../../components"
import DepGraph from "../../depgraph"
import { Date, getDate } from "../../components/Date"

function PoetryFooter({ allFiles, fileData, displayClass, cfg }: QuartzComponentProps) {
  return (
    <footer class={classNames(displayClass, "poetry-footer")}>
      <Date date={getDate(cfg, fileData)!} locale={cfg.locale} />
    </footer>
  )
}

export const PoetryPage: QuartzEmitterPlugin = () => {
  const Meta = MetaConstructor()

  const opts: FullPageLayout = {
    head: HeadConstructor(),
    header: [],
    beforeBody: [ArticleTitle(), ContentMeta({ showReadingTime: false, showReturnHome: true })],
    pageBody: Content(),
    left: [Meta],
    right: [],
    footer: PoetryFooter,
  }

  const { head: Head, header, beforeBody, pageBody, left, right, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  return {
    name: "PoetryPage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        Meta,
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
        if (file.data.frontmatter?.poem === true) {
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
