import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.Search(),
    Component.Breadcrumbs({ rootName: "~", style: "unique", spacerSymbol: "/" }),
  ],
  afterBody: [
    Component.DesktopOnly(Component.Toolbar()),
    Component.Backlinks(),
    Component.MinimalFooter({
      links: { github: "https://github.com/aarnphm", twitter: "https://twitter.com/aarnphm_" },
    }),
  ],
  footer: Component.Spacer(),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
    Component.Spacer(),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Graph(),
    Component.Reader(),
    Component.Image(),
  ],
  left: [Component.DesktopOnly(Component.Sidenotes())],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Search(),
    Component.Keybind(),
    Component.Breadcrumbs({ rootName: "~", style: "full", spacerSymbol: "/" }),
  ],
  left: [],
  right: [],
}
