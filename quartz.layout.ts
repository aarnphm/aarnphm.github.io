import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

export const afterBodyComponents: Partial<SharedLayout> = {
  afterBody: [
    Component.Comments({
      provider: "giscus",
      options: {
        repo: "aarnphm/sites",
        repoId: "R_kgDOLbqALg",
        category: "Announcements",
        categoryId: "DIC_kwDOLbqALs4ChE6l",
        reactionsEnabled: false,
      },
    }),
    Component.MinimalFooter({
      links: { github: "https://github.com/aarnphm", twitter: "https://twitter.com/aarnphm_" },
      showInfo: true,
    }),
  ],
}

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Spacer(),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs({ rootName: "~", style: "unique", spacerSymbol: "/" }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  right: [
    Component.Search(),
    Component.DesktopOnly(Component.Keybind({ enableTooltip: false })),
    Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
  left: [Component.DesktopOnly(Component.Sidenotes())],
  // left: [],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs({ rootName: "~", style: "full", spacerSymbol: "/" })],
  left: [Component.Search(), Component.DesktopOnly(Component.Keybind({ enableTooltip: false }))],
  right: [],
}
