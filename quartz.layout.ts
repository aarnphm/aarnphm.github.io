import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.PageTitle(),
    Component.Breadcrumbs({ rootName: "~", style: "unique", spacerSymbol: "/" }),
    Component.Keybind(),
    Component.Search(),
  ],
  afterBody: [Component.Backlinks()],
  footer: Component.Footer({
    links: {
      github: "https://github.com/aarnphm",
      twitter: "https://twitter.com/aarnphm_",
      bsky: "https://bsky.app/profile/aarnphm.xyz",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.Byline(Component.TagList(), Component.ContentMeta()),
  ],
  left: [Component.DesktopOnly(Component.TableOfContents())],
  right: [
    Component.Graph(),
    Component.Reader(),
    Component.Image(),
    Component.Mermaid(),
    Component.DesktopOnly(Component.Toolbar()),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs({ rootName: "~", style: "full", spacerSymbol: "/" }),
    Component.Keybind(),
    Component.Search(),
  ],
  left: [],
  right: [],
}
