import { Data } from "vfile"
import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"
import { FileNode } from "./quartz/components/ExplorerNode"
import { QuartzComponent } from "./quartz/components/types"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  footer: Component.MinimalFooter(),
}

function recentFilter(path: string, excludePaths: string[] = []) {
  return (f: Data) => {
    const slug = f.slug!
    if (slug.startsWith(path + "/")) {
      const subpath = slug.slice(path.length + 1).split("/")[0]
      return (
        !excludePaths.includes(subpath) && f.slug! !== path + "/index" && !f.frontmatter?.noindex
      )
    }
    return false
  }
}

const left = (
  enableRecentNotes: boolean = false,
  enableExplorer: boolean = false,
  enableMeta: boolean = false,
) => {
  const left: QuartzComponent[] = [
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.Keybind({ enableTooltip: false }),
  ]
  const recentNotes = [
    Component.RecentNotes({
      title: "Recent Writing",
      limit: 3,
      filter: recentFilter("posts"),
      linkToMore: "posts/" as SimpleSlug,
    }),
    Component.RecentNotes({
      title: "Recent Notes",
      limit: 3,
      filter: recentFilter("thoughts", ["university"]),
      linkToMore: "thoughts/" as SimpleSlug,
    }),
  ]
  const desktopOnly = [
    Component.Graph({
      globalGraph: { linkDistance: 50 },
      localGraph: { repelForce: 0.79, centerForce: 0.2, scale: 1.04, linkDistance: 40 },
    }),
    Component.Backlinks(),
    Component.TableOfContents(),
  ]

  if (enableMeta) left.push(Component.Meta({ enableSearch: false, enableDarkMode: false }))

  if (enableRecentNotes) desktopOnly.push(...recentNotes)
  if (enableExplorer)
    desktopOnly.push(
      Component.Explorer({
        filterFn: (node: FileNode) => {
          return !["tags", "university"].some((path) => node.name.includes(path))
        },
      }),
    )
  left.push(...desktopOnly.flatMap(Component.DesktopOnly))

  return { left }
}

const right = () => {
  return { right: [Component.MobileOnly(Component.Backlinks())] }
}

const beforeBody = (enableContentMeta: boolean = true, enableTagList: boolean = true) => {
  const beforeBody: QuartzComponent[] = [Component.ArticleTitle()]
  if (enableContentMeta) beforeBody.push(Component.ContentMeta())
  if (enableTagList) beforeBody.push(Component.TagList())
  return { beforeBody }
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  ...beforeBody(),
  ...left(),
  ...right(),
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  ...beforeBody(false, false),
  ...left(false, true, true),
  right: [],
}
