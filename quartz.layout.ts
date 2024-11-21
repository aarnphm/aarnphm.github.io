import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.DesktopOnly(Component.Toolbar()),
    Component.Image(),
    Component.MinimalFooter({
      links: { github: "https://github.com/aarnphm", twitter: "https://twitter.com/aarnphm_" },
    }),
  ],
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
    Component.Keybind(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Graph(),
    Component.Backlinks(),
    Component.Reader(),
  ],
  left: [Component.DesktopOnly(Component.Sidenotes())],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs({ rootName: "~", style: "full", spacerSymbol: "/" })],
  left: [Component.Search(), Component.DesktopOnly(Component.Keybind({ enableTooltip: false }))],
  right: [],
}
