import { PageLayout, SharedLayout } from './quartz/cfg'
import * as Component from './quartz/components'
import { usesApparatus } from './quartz/util/apparatus'
import { unitOf } from './quartz/util/course'

// /fr and the course pages carry their own head; the description and meta line stay off them
const hidesPageMeta = ({ fileData }: { fileData: { slug?: string } }) => {
  const slug = fileData.slug ?? ''
  return slug.startsWith('fr/') || slug.startsWith('courses/')
}

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
      colophon: '/colophon',
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
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: props => !hidesPageMeta(props),
    }),
    Component.ConditionalRender({
      component: Component.ArticleTitle({ enableDescription: false }),
      condition: hidesPageMeta,
    }),
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
        // a page with the margin apparatus (/fr, `layout: technical`) keeps only its tags here
        Component.ConditionalRender({
          component: Component.ContentMeta(),
          condition: props => !hidesPageMeta(props) && !usesApparatus(props.fileData),
        }),
      ),
      // where the meta line is hidden, an untagged page has nothing left for the byline to show
      condition: props => {
        const kind = unitOf(props.fileData.slug ?? '')?.kind
        if (kind === 'psets') return false
        return (
          !hidesPageMeta(props) || kind === 'notes' || !!props.fileData.frontmatter?.tags?.length
        )
      },
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
