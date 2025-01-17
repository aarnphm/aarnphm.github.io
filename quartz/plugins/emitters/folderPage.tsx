import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { HtmlContent, QuartzPluginData, defaultProcessedContent } from "../vfile"
import { FullPageLayout } from "../../cfg"
import path from "path"
import {
  FilePath,
  FullSlug,
  SimpleSlug,
  stripSlashes,
  joinSegments,
  pathToRoot,
  simplifySlug,
} from "../../util/path"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { FolderContent, Spacer } from "../../components"
import { write } from "./helpers"
import { i18n } from "../../i18n"
import DepGraph from "../../depgraph"
import { byDateAndAlphabetical } from "../../components/PageList"

interface FolderPageOptions extends FullPageLayout {
  sort?: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

export const FolderPage: QuartzEmitterPlugin<Partial<FolderPageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: FolderContent({ sort: userOpts?.sort }),
    header: [...defaultListPageLayout.beforeBody],
    beforeBody: [],
    sidebar: [],
    afterBody: [],
    footer: Spacer(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: "FolderPage",
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async getDependencyGraph(_ctx, content, _resources) {
      const graph = new DepGraph<FilePath>()
      const folders = getFolders(_ctx.allSlugs)

      content.map(([_tree, vfile]) => {
        const slug = vfile.data.slug
        if (!slug) return

        // Add dependencies for containing folders
        const containingFolders = getAllFoldersFromPath(path.dirname(slug as string) as SimpleSlug)
        containingFolders.forEach((folder) => {
          if (folders.has(folder)) {
            graph.addEdge(vfile.data.filePath!, joinSegments(folder, "index.html") as FilePath)
          }
        })
      })

      return graph
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const fps: FilePath[] = []
      const allFiles = content.map((c) => c[1].data)
      const cfg = ctx.cfg.configuration

      // Use allSlugs to get all folders, including those without markdown files
      const folders = getFolders(ctx.allSlugs)

      const folderDescriptions: Record<string, HtmlContent> = Object.fromEntries(
        [...folders].map((folder) => [
          folder,
          defaultProcessedContent({
            slug: joinSegments(folder, "index") as FullSlug,
            frontmatter: {
              title: `${i18n(cfg.locale).pages.folderContent.folder}: ${folder}`,
              tags: ["folder"],
              pageLayout: "default",
            },
            dates: allFiles
              .filter((f) => {
                const fileSlug = stripSlashes(simplifySlug(f.slug!))
                return fileSlug.startsWith(folder) && fileSlug !== folder
              })
              .sort(byDateAndAlphabetical(cfg))
              .at(0)?.dates,
          }),
        ]),
      )

      for (const [tree, file] of content) {
        const slug = stripSlashes(simplifySlug(file.data.slug!)) as SimpleSlug
        if (folders.has(slug)) {
          folderDescriptions[slug] = [tree, file]
        }
      }

      for (const folder of folders) {
        const slug = joinSegments(folder, "index") as FullSlug
        const [tree, file] = folderDescriptions[folder]
        const externalResources = pageResources(pathToRoot(slug), file.data, resources)
        const componentData: QuartzComponentProps = {
          ctx,
          fileData: file.data,
          externalResources,
          cfg,
          children: [],
          tree,
          allFiles,
        }

        const content = renderPage(ctx, slug, componentData, opts, externalResources, "full-col")
        const fp = await write({
          ctx,
          content,
          slug,
          ext: ".html",
        })

        fps.push(fp)
      }
      return fps
    },
  }
}

function getAllFoldersFromPath(slug: SimpleSlug): SimpleSlug[] {
  const folders = new Set<SimpleSlug>()
  let currentPath = slug

  while (currentPath !== "." && currentPath !== "") {
    folders.add(currentPath)
    currentPath = path.dirname(currentPath) as SimpleSlug
  }

  return [...folders]
}

function getFolders(allSlugs: FullSlug[]): Set<SimpleSlug> {
  const folders = new Set<SimpleSlug>()

  // Add all folders from all file paths (both md and non-md)
  for (const slug of allSlugs) {
    const dirPath = path.dirname(slug)
    const containingFolders = getAllFoldersFromPath(dirPath as SimpleSlug)
    containingFolders.forEach((folder) => folders.add(folder))
  }

  // Filter out special folders and empty string
  return new Set(
    [...folders].filter(
      (folder) =>
        folder !== "." && folder !== "" && folder !== "tags" && !folder.startsWith("tags/"),
    ),
  )
}
