import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.StackedNotes(),
    Component.Breadcrumbs({
      rootName: "~",
      spacerSymbol: "/",
      showCurrentPage: true,
    }),
    Component.Image(),
    Component.Graph(),
    Component.Palette(),
    Component.Keybind(),
    Component.Search(),
    Component.Darkmode(),
  ],
  afterBody: [Component.Recommendations(), Component.Backlinks()],
  footer: Component.Footer({
    layout: "minimal",
    links: {
      github: "https://github.com/aarnphm",
      twitter: "https://twitter.com/aarnphm_",
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
  sidebar: [Component.DesktopOnly(Component.TableOfContents()), Component.Reader()],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.StackedNotes(),
    Component.Breadcrumbs({ rootName: "~", spacerSymbol: "/" }),
    Component.Image(),
    Component.Graph(),
    Component.Palette(),
    Component.Keybind(),
    Component.Search(),
  ],
  sidebar: [],
}
