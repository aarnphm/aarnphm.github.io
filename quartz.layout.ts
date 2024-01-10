import { Data } from "vfile"
import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/aarnphm",
      Twitter: "https://twitter.com/aarnphm_",
      Bento: "https://bento.me/aarnphm",
    },
  }),
}

function filterFunc(path: string, excludePaths: string[] = []) {
  return (f: Data) => {
    const slug = f.slug!
    if (slug.startsWith(path + "/")) {
      const subpath = slug.slice(path.length + 1).split("/")[0]
      return !excludePaths.includes(subpath) && f.slug! !== path + "/index" && !f.frontmatter?.noindex
    }
    return false
  }
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
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
        filter: filterFunc("dump", ['university']),
        linkToMore: "dump/" as SimpleSlug,
      }),
    ),
    Component.DesktopOnly(Component.TableOfContents()),
  ],
  right: [Component.Graph({ localGraph: { showTags: false }, globalGraph: { showTags: true } }), Component.Backlinks()],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
  ],
  right: [],
}
