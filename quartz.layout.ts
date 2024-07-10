import { Data } from "vfile"
import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"
import { FileNode } from "./quartz/components/ExplorerNode"
import { globalGraphConfig as globalGraph } from "./quartz/components/Meta"
import { QuartzComponent } from "./quartz/components/types"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Spacer(),
}

function recentFilter(path: string, excludePaths: string[] = []) {
  return (f: Data) => {
    const slug = f.slug!
    if (slug.startsWith(path + "/")) {
      const subpath = slug.slice(path.length + 1).split("/")[0]
      return (
        !excludePaths.includes(subpath) &&
        f.slug! !== path + "/index" &&
        !f.frontmatter?.noindex &&
        !f.frontmatter?.construction
      )
    }
    return false
  }
}

interface Options {
  enableRecentNotes: boolean
  enableExplorer: boolean
  enableMeta: boolean
  enableDarkmode: boolean
}

const defaultOptions: Options = {
  enableRecentNotes: false,
  enableExplorer: false,
  enableMeta: false,
  enableDarkmode: false,
}

const left = (userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const left: QuartzComponent[] = [Component.Search()]

  if (opts.enableDarkmode) left.push(Component.Darkmode())

  const desktopOnly: QuartzComponent[] = []

  if (opts.enableMeta) left.push(Component.Meta({ enableSearch: false, enableDarkMode: false }))

  if (opts.enableRecentNotes)
    desktopOnly.push(
      Component.RecentNotes({
        title: "Notes Récentes",
        limit: 5,
        filter: recentFilter("thoughts", ["university"]),
        linkToMore: "thoughts/" as SimpleSlug,
      }),
    )

  if (opts.enableExplorer)
    desktopOnly.push(
      Component.Explorer({
        filterFn: (node: FileNode) => {
          return !["tags", "university"].some((path) => node.name.includes(path))
        },
      }),
    )

  desktopOnly.push(Component.Keybind({ enableTooltip: false }))

  left.push(...desktopOnly.flatMap(Component.DesktopOnly))

  return { left }
}

const right = () => {
  return {
    right: [
      ...left().left,
      Component.Graph({ globalGraph, localGraph: { showTags: false } }),
      Component.DesktopOnly(Component.TableOfContents()),
      Component.Backlinks(),
    ],
  }
}

const beforeBody = (
  enableContentMeta: boolean = true,
  enableTagList: boolean = true,
  enableArticleTitle: boolean = true,
) => {
  const beforeBody: QuartzComponent[] = []
  if (enableArticleTitle) beforeBody.push(Component.ArticleTitle())
  if (enableContentMeta) beforeBody.push(Component.ContentMeta())
  if (enableTagList) beforeBody.push(Component.TagList())
  return { beforeBody }
}

const enableMeta = true
const enableExplorer = false
const enableRecentNotes = false

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  ...beforeBody(),
  ...right(),
  left: [],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  ...beforeBody(false, false, false),
  ...left({ enableExplorer, enableMeta, enableRecentNotes }),
  right: [],
}
