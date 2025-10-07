import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { ProcessedContent, QuartzPluginData, defaultProcessedContent } from "../vfile"
import { FullPageLayout } from "../../cfg"
import path from "path"
import {
  FullSlug,
  SimpleSlug,
  stripSlashes,
  joinSegments,
  pathToRoot,
  simplifySlug,
  slugifyFilePath,
} from "../../util/path"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { FolderContent } from "../../components"
import { write } from "./helpers"
import { i18n, TRANSLATIONS } from "../../i18n"
import { BuildCtx } from "../../util/ctx"
import { StaticResources } from "../../util/resources"
interface FolderPageOptions extends FullPageLayout {
  sort?: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

async function* processFolderInfo(
  ctx: BuildCtx,
  folderInfo: Record<SimpleSlug, ProcessedContent>,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  for (const [folder, folderContent] of Object.entries(folderInfo) as [
    SimpleSlug,
    ProcessedContent,
  ][]) {
    const slug = joinSegments(folder, "index") as FullSlug
    const [tree, file] = folderContent
    const cfg = ctx.cfg.configuration
    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData: file.data,
      externalResources,
      cfg,
      children: [],
      tree,
      allFiles,
    }

    const content = renderPage(ctx, slug, componentData, opts, externalResources, true)
    yield write({
      ctx,
      content,
      slug,
      ext: ".html",
    })
  }
}

function computeFolderInfo(
  folders: Set<SimpleSlug>,
  content: ProcessedContent[],
  locale: keyof typeof TRANSLATIONS,
): Record<SimpleSlug, ProcessedContent> {
  // Create default folder descriptions
  const folderInfo: Record<SimpleSlug, ProcessedContent> = Object.fromEntries(
    [...folders].map((folder) => [
      folder,
      defaultProcessedContent({
        slug: joinSegments(folder, "index") as FullSlug,
        frontmatter: {
          title: `${i18n(locale).pages.folderContent.folder}: ${folder}`,
          pageLayout: "default",
          tags: [],
        },
      }),
    ]),
  )

  // Update with actual content if available
  for (const [tree, file] of content) {
    const slug = stripSlashes(simplifySlug(file.data.slug!)) as SimpleSlug
    if (folders.has(slug)) {
      folderInfo[slug] = [tree, file]
    }
  }

  return folderInfo
}

function _getFolders(slug: FullSlug): SimpleSlug[] {
  var folderName = path.dirname(slug ?? "") as SimpleSlug
  const parentFolderNames = [folderName]

  while (folderName !== ".") {
    folderName = path.dirname(folderName ?? "") as SimpleSlug
    parentFolderNames.push(folderName)
  }
  return parentFolderNames
}

export const FolderPage: QuartzEmitterPlugin<Partial<FolderPageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: FolderContent({ sort: userOpts?.sort }),
    header: [...defaultListPageLayout.beforeBody],
    beforeBody: [],
    sidebar: [],
    afterBody: [],
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: "FolderPage",
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const mdFiles = content.map((c) => c[1].data)
      const cfg = ctx.cfg.configuration

      // Build folder set from all slugs (includes non-markdown files)
      const folders: Set<SimpleSlug> = new Set(
        ctx.allSlugs.flatMap((slug) =>
          _getFolders(slug as FullSlug).filter(
            (folderName) => folderName !== "." && folderName !== "tags",
          ),
        ),
      )

      const folderInfo = computeFolderInfo(folders, content, cfg.locale)
      yield* processFolderInfo(ctx, folderInfo, mdFiles, opts, resources)
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)
      const cfg = ctx.cfg.configuration

      // Find all folders that need to be updated based on changed files
      const affectedFolders: Set<SimpleSlug> = new Set()
      for (const changeEvent of changeEvents) {
        // Prefer slug from VFile if available (markdown)
        let baseSlug: FullSlug | undefined = changeEvent.file?.data.slug as FullSlug | undefined
        // For non-markdown or missing VFile, derive from path
        if (!baseSlug) {
          baseSlug = slugifyFilePath(changeEvent.path)
        }
        if (!baseSlug) continue

        const folders = _getFolders(baseSlug).filter(
          (folderName) => folderName !== "." && folderName !== "tags",
        )
        folders.forEach((folder) => affectedFolders.add(folder))
      }

      // If there are affected folders, rebuild their pages
      if (affectedFolders.size > 0) {
        const folderInfo = computeFolderInfo(affectedFolders, content, cfg.locale)
        yield* processFolderInfo(ctx, folderInfo, allFiles, opts, resources)
      }
    },
  }
}
