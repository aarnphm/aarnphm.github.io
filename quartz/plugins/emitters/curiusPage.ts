import { QuartzEmitterPlugin } from "../types"
import { sharedPageComponents } from "../../../quartz.layout"
import { CuriusContent } from "../../components"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import path from "path"
import {
  FilePath,
  FullSlug,
  _stripSlashes,
  joinSegments,
  pathToRoot,
  simplifySlug,
} from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import { QuartzComponentProps } from "../../components/types"
import chalk from "chalk"

export const CuriusPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    beforeBody: [],
    left: [],
    right: [],
    pageBody: CuriusContent(),
    ...userOpts,
  }

  return {
    name: "CuriusContent",
    getQuartzComponents() {
      return [CuriusContent()]
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const fps: FilePath[] = []
      const cfg = ctx.cfg.configuration
      const allFiles = content.map((c) => c[1].data)

      let containsCurius = false
      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (slug === "curius") {
          containsCurius = true
          const externalResources = pageResources(pathToRoot(slug), resources)

          const componentData: QuartzComponentProps = {
            fileData: file.data,
            externalResources: externalResources,
            cfg: cfg,
            children: [],
            tree,
            allFiles,
          }

          const content = renderPage(slug, componentData, opts, externalResources)
          const fp = await write({
            ctx,
            content,
            slug,
            ext: ".html",
          })
          fps.push(fp)
        }
      }

      if (!containsCurius) {
        console.log(
          chalk.yellow(
            `\nWarning: you seem to be missing an \`curius.md\` page file at the root of your \`${ctx.argv.directory}\` folder. This may cause errors when deploying.`,
          ),
        )
      }

      return fps
    },
  }
}
