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

const left = (enableRecentNotes: boolean = false): Partial<PageLayout> => {
  const left: QuartzComponent[] = [
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
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
  if (enableRecentNotes) desktopOnly.push(...recentNotes)
  left.push(...desktopOnly.flatMap(Component.DesktopOnly))
  return { left }
}

const right = (): Partial<PageLayout> => {
  return { right: [Component.MobileOnly(Component.Backlinks())] }
}

const beforeBody = (
  enableContentMeta: boolean = true,
  enableTagList: boolean = true,
): Partial<PageLayout> => {
  const beforeBody: QuartzComponent[] = [Component.ArticleTitle()]
  if (enableContentMeta) beforeBody.push(Component.ContentMeta())
  if (enableTagList) beforeBody.push(Component.TagList())
  return { beforeBody }
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  ...(right() as PageLayout),
  ...left(),
  ...beforeBody(),
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  ...(beforeBody(false, false) as PageLayout),
  left: [
    Component.Meta({ enableSearch: false, enableDarkMode: false }),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.Explorer({
        filterFn: (node: FileNode) => {
          return !["tags", "university"].some((path) => node.name.includes(path))
        },
      }),
    ),
  ],
  right: [],
}
