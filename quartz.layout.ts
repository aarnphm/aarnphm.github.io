import { Data } from "vfile"
import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"
import { FileNode } from "./quartz/components/ExplorerNode"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/aarnphm",
      Twitter: "https://twitter.com/aarnphm_",
    },
  }),
}

function filterFunc(path: string, excludePaths: string[] = []) {
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

const explorerFilterFn = (node: FileNode) => {
  return !["university", "papers", "tags"].some((path) => node.name.includes(path))
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle(), Component.ContentMeta(), Component.TagList()],
  left: [
    Component.MobileOnly(Component.Explorer()),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.RecentNotes({
        title: "Recent Writing",
        limit: 3,
        filter: filterFunc("posts"),
        linkToMore: "posts/" as SimpleSlug,
      }),
    ),
    Component.DesktopOnly(
      Component.RecentNotes({
        title: "Recent Notes",
        limit: 3,
        filter: filterFunc("dump", ["university"]),
        linkToMore: "dump/" as SimpleSlug,
      }),
    ),
  ],
  right: [
    Component.Graph({
      globalGraph: { linkDistance: 50 },
      localGraph: { repelForce: 0.79, centerForce: 0.2, scale: 1.04, linkDistance: 40 },
    }),
    Component.Backlinks(),
    Component.DesktopOnly(Component.TableOfContents()),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle()],
  left: [
    Component.MobileOnly(Component.Explorer()),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(Component.Explorer({ filterFn: explorerFilterFn })),
  ],
  right: [],
}
