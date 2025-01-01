import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.PageTitle(),
    Component.Breadcrumbs({ rootName: "~", style: "unique", spacerSymbol: "/" }),
    Component.Keybind(),
    Component.StackedNotes(),
    Component.Search(),
    Component.Graph(),
    Component.Image(),
  ],
  afterBody: [Component.Recommendations(), Component.Backlinks()],
  footer: Component.Footer({
    layout: "minimal",
    links: {
      github: "https://github.com/aarnphm",
      twitter: "https://twitter.com/aarnphm_",
      bsky: "https://bsky.app/profile/aarnphm.xyz",
      feed: "/feed.xml",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.Byline(Component.TagList(), Component.ContentMeta()),
  ],
  sidebar: [
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Reader(),
    Component.DesktopOnly(Component.Toolbar()),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.PageTitle(),
    Component.Breadcrumbs({ rootName: "~", style: "full", spacerSymbol: "/" }),
    Component.Keybind(),
    Component.StackedNotes(),
    Component.Search(),
    Component.Graph(),
    Component.Image(),
  ],
  sidebar: [],
}
