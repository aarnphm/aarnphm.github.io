import { PageLayout, SharedLayout } from './quartz/cfg'
import * as Component from './quartz/components'
import { unitOf } from './quartz/util/course'

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [
    Component.VisitorWeather(),
    Component.HomepageTrail(),
    Component.Cursor(),
    Component.Breadcrumbs({
      rootName: '~',
      spacerSymbol: '/',
      leadingWindow: 3,
      trailingWindow: 2,
    }),
    Component.StackedNotes(),
    Component.Image(),
    Component.Palette(),
    Component.Keybind(),
    Component.Search(),
    Component.Darkmode(),
    Component.CodeCopy(),
  ],
  afterBody: [
    Component.CourseUnitFoot(),
    Component.FrUnitFoot(),
    Component.InquiryMap(),
    Component.License(),
    Component.Backlinks(),
  ],
  footer: Component.Footer({
    layout: 'minimal',
    links: {
      'CC BY-NC-SA': 'https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en',
      github: 'https://github.com/aarnphm',
      privacy: '/privacy',
      'terms of service': '/terms-of-service',
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.NotebookRuntimeLoader(),
    // the course crumb is the trail on unit pages (the site breadcrumbs stay off them)
    Component.CourseUnitBar(),
    Component.FrUnitBar(),
    // the university course trail on notes and notebooks inside a course folder
    Component.UniversityCrumbBar(),
    Component.ArticleTitle(),
    // a problem set page is a course page: no note byline, the desk sits under the title
    Component.ConditionalRender({
      component: Component.Byline(
        Component.Flex({
          classNames: ['first'],
          // OCW notes add a `your notes` cell under the tags
          components: [
            { Component: Component.TagList(), align: 'start' },
            { Component: Component.CourseUnitNotes(), align: 'start' },
          ],
          direction: 'column',
          gap: '0.5rem',
        }),
        Component.ContentMeta(),
      ),
      condition: ({ fileData }) => unitOf(fileData.slug ?? '')?.kind !== 'psets',
    }),
    Component.CoursePsetDesk(),
  ],
  sidebar: [Component.DesktopOnly(Component.TableOfContents()), Component.Reader()],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs({ rootName: '~', spacerSymbol: '/', trailingWindow: 1 }),
    Component.ArticleTitle({ enableDescription: false, visuallyHidden: true }),
    Component.StackedNotes(),
    Component.Image(),
    Component.Palette(),
    Component.Keybind(),
    Component.Search(),
    Component.Darkmode(),
    Component.CodeCopy(),
  ],
  sidebar: [],
}
