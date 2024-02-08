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
  footer: Component.MinimalFooter({
    links: { GitHub: "https://github.com/aarnphm", Twitter: "https://twitter.com/aarnphm_" },
  }),
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

interface Options {
  enableRecentNotes: boolean
  enableExplorer: boolean
  enableMeta: boolean
  listView: boolean
}

const defaultOptions: Options = {
  enableRecentNotes: false,
  enableExplorer: false,
  enableMeta: false,
  listView: false,
}

const left = (userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const left: QuartzComponent[] = [
    Component.Search(),
    Component.MobileOnly(Component.Spacer()),
    Component.Darkmode(),
    Component.Keybind({ enableTooltip: false }),
  ]

  const desktopOnly = []

  if (!opts.listView) desktopOnly.push(Component.TableOfContents())

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

  left.push(...desktopOnly.flatMap(Component.DesktopOnly))

  return { left }
}

const right = () => {
  return {
    right: [
      Component.Graph({
        globalGraph: { linkDistance: 50 },
        localGraph: { repelForce: 0.79, centerForce: 0.2, scale: 1.04, linkDistance: 40 },
      }),
      Component.Backlinks(),
    ],
  }
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
  ...left({ enableRecentNotes: true }),
  ...right(),
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  ...beforeBody(false, false),
  ...left({ enableExplorer: true, enableMeta: true, listView: true }),
  right: [],
}
