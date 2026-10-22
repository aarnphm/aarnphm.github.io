import { Element, Node, Root } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { render } from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import { QuartzPluginData } from '../../plugins/vfile'
import { QuartzComponentProps } from '../../types/component'
import { clone } from '../../util/clone'
import { hasCourseCrumb, pdfResourceOf } from '../../util/course'
import { BuildCtx } from '../../util/ctx'
import { htmlToJsx } from '../../util/jsx'
import { classNames } from '../../util/lang'
import { classPrefixPattern } from '../../util/page-styles'
import { FullSlug } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { encryptContent } from '../../util/protected'
import { JSResourceToScriptElement, StaticResources } from '../../util/resources'
import { isSlidePage } from '../../util/section-icon'
import { HomepageIdentityHeading } from '../AgentIdentity'
import CodeCopy from '../CodeCopy'
import Darkmode from '../Darkmode'
import Graph from '../Graph'
import HomepageTrail from '../HomepageTrail'
import Image from '../Image'
import Keybind from '../Keybind'
import Palette from '../Palette'
import Search from '../Search'
import { renderStravaActivities } from '../strava-activities'
import { QuartzIcon } from '../svg'
import VisitorWeather from '../VisitorWeather'
import { renderBaseEmbeds } from './base-embeds'
import { CuriusContent, CuriusFriends, CuriusTrail } from './curius'
import { getBibList, getFootnotesList } from './footnotes-citations'
import { renderFrMargins } from './fr-margins'
import {
  ElementComponent,
  Functions,
  Header,
  Headings,
  MenuFooter,
  PoetryFooter,
  PressKit,
  updateStreamDataFromTree,
  updateTocDataFromTree,
  type RenderComponents,
  type RenderPageOptions,
} from './page-chrome'
import { applyTractatusLayout } from './tractatus'
import { resolveTransclusions } from './transclude'

export function renderPageWithMetadata(
  ctx: BuildCtx,
  slug: FullSlug,
  componentData: QuartzComponentProps,
  components: RenderComponents,
  pageResources: StaticResources,
  isFolderTag?: boolean,
  isBoxy: boolean = false,
  userOptions?: RenderPageOptions,
): { html: string; fileData: QuartzPluginData } {
  const renderPerf = new PerfTimer()
  componentData = { ...componentData, fileData: { ...componentData.fileData }, pageSlug: slug }
  const renderOptions: RenderPageOptions = {
    skipProtected: false,
    forEmail: false,
    skipSearch: false,
    ...userOptions,
  }
  const skipProtected = renderOptions?.skipProtected ?? renderOptions?.forEmail ?? false
  if (skipProtected) {
    delete componentData.fileData.protectedPassword
  }

  const retrievalNodes: Element[] = []
  if (
    (componentData.tree as Root).children.length === 0 &&
    !componentData.fileData.protectedPassword
  ) {
    const tocOpts = componentData.fileData.tocOptions
    if (tocOpts) {
      delete componentData.fileData.toc
      if (tocOpts.display) {
        componentData.fileData.tocOptions = { ...tocOpts, sourceEntries: 0 }
      }
    }
    logBuildSpan(ctx.argv, 'render:transclude', slug, renderPerf.elapsedMs())
    logBuildSpan(ctx.argv, 'render:prepare', slug, renderPerf.elapsedMs())
  } else {
    // make a deep copy of the tree so we don't remove the transclusion references
    // for the file cached in contentMap in build.ts
    const root = clone(componentData.tree) as Root
    const visited = new Set<FullSlug>([slug])
    // NOTE: set componentData.tree to the edited html that has transclusions rendered

    const transcluded = resolveTransclusions(
      root,
      componentData,
      { visited },
      componentData.fileData.flashcards ? { dynalist: false } : undefined,
    )
    let tree = transcluded.tree
    if (transcluded.readingTime) componentData.fileData.readingTime = transcluded.readingTime
    logBuildSpan(ctx.argv, 'render:transclude', slug, renderPerf.elapsedMs())
    renderBaseEmbeds(tree, componentData)
    renderStravaActivities(tree, slug)
    renderFrMargins(tree, componentData, ctx)

    if (componentData.fileData.frontmatter?.pageLayout === 'technical-tractatus') {
      applyTractatusLayout(tree)
    }

    if (componentData.fileData.protectedPassword) {
      const password = componentData.fileData.protectedPassword as string

      // Convert final tree to HTML
      const finalHtml = toHtml(tree, { allowDangerousHtml: true })

      // Encrypt the final HTML
      const encrypted = encryptContent(finalHtml, password)

      // Replace tree with password prompt overlay
      tree = {
        type: 'root',
        children: [
          h(
            'div.protected-content-wrapper',
            {
              dataProtected: 'true',
              dataSlug: componentData.fileData.slug,
              dataEncryptedContent: encodeURIComponent(JSON.stringify(encrypted)),
            },
            [
              h('.password-prompt-overlay', { id: 'password-prompt', style: 'display: flex;' }, [
                h('.password-prompt-container', [
                  h('p', 'this content is protected'),
                  h('form.password-form', [
                    h('input.password-input', {
                      type: 'password',
                      placeholder: 'enter password',
                      autocomplete: 'off',
                      required: true,
                      id: 'protected-password-input',
                      name: 'password input for protected page',
                    }),
                    h('button.password-submit', { type: 'submit' }, 'unlock'),
                  ]),
                  h(
                    'p.password-error',
                    { style: 'display: none; color: var(--rose); margin-top: 2rem;' },
                    'incorrect password',
                  ),
                ]),
              ]),
            ],
          ),
        ],
      } as Root

      // Clean up password from file.data
      delete componentData.fileData.protectedPassword
    }

    // NOTE: Finally, we dump out the data-references and data-footnotes down to page footer, if exists
    const toRemove: Array<{ parent: Element; index: number; node: Element }> = []
    const referenceSections: Element[] = []
    const footnoteSections: Element[] = []
    let referenceOrder = Number.MAX_SAFE_INTEGER
    let footnoteOrder = Number.MAX_SAFE_INTEGER

    visit(tree, { tagName: 'section' }, (node, index, parent) => {
      const isReference = node.properties?.dataReferences === ''
      const isFootnote = node.properties?.dataFootnotes === ''
      if (isReference || isFootnote) {
        const className = Array.isArray(node.properties.className)
          ? node.properties.className
          : (node.properties.className = [])
        className.push('main-col')
        toRemove.push({ parent: parent as Element, index: index!, node })

        if (isReference) {
          referenceSections.push(node as Element)
          referenceOrder = Math.min(referenceOrder, index ?? Number.MAX_SAFE_INTEGER)
        } else {
          footnoteSections.push(node as Element)
          footnoteOrder = Math.min(footnoteOrder, index ?? Number.MAX_SAFE_INTEGER)
        }
      }
    })

    // remove in reverse order to maintain correct indices
    toRemove.sort((a, b) => b.index - a.index)
    for (const { parent, index } of toRemove) {
      parent.children.splice(index, 1)
    }

    const mergeSectionLists = (sections: Element[], listTag: 'ol' | 'ul'): Element | undefined => {
      if (sections.length === 0) return undefined
      const base = sections[0]
      const listGetter = listTag === 'ol' ? getFootnotesList : getBibList
      const targetList = listGetter(base)
      if (!targetList) return base

      // collect all items before mutating any list to avoid losing base children
      const allItems: Element[] = []
      for (const section of sections) {
        const list = listGetter(section)
        if (!list) continue
        allItems.push(...((list.children as Element[]) ?? []))
      }

      const seenIds = new Set<string>()
      targetList.children = []

      for (const child of allItems) {
        const childId = (child?.properties?.id as string | undefined) ?? ''
        if (childId && seenIds.has(childId)) continue
        if (childId) seenIds.add(childId)
        targetList.children.push(child)
      }

      return base
    }

    const mergedReferences = mergeSectionLists(referenceSections, 'ul')
    const mergedFootnotes = mergeSectionLists(footnoteSections, 'ol')

    const orderedNodes: Array<{ order: number; node: Element }> = []
    if (mergedReferences) orderedNodes.push({ order: referenceOrder, node: mergedReferences })
    if (mergedFootnotes) orderedNodes.push({ order: footnoteOrder, node: mergedFootnotes })
    orderedNodes.sort((a, b) => a.order - b.order)
    retrievalNodes.push(...orderedNodes.map(entry => entry.node))
    logBuildSpan(ctx.argv, 'render:prepare', slug, renderPerf.elapsedMs())

    componentData.tree = tree
    updateTocDataFromTree(tree, componentData)
    updateStreamDataFromTree(tree, componentData)
  }
  isFolderTag = isFolderTag ?? false

  const skipSearch = renderOptions?.skipSearch ?? renderOptions?.forEmail ?? false
  if (slug === 'index' && !skipSearch) {
    components = {
      ...components,
      header: [
        VisitorWeather(),
        HomepageTrail(),
        Image(),
        Graph({ localGraph: undefined }),
        Search(),
        Palette(),
        Keybind(),
        CodeCopy(),
        Darkmode(),
      ],
      sidebar: [],
      afterBody: [],
      beforeBody: [],
      pageBody: (props: QuartzComponentProps) => {
        const { displayClass } = props
        const Element = ElementComponent(false)

        return (
          <div class={classNames(displayClass, 'landing')}>
            <HomepageIdentityHeading />
            <Element {...props} />
          </div>
        )
      },
    }
  } else if (slug === 'curius') {
    components = {
      ...components,
      header: [],
      beforeBody: [],
      sidebar: [CuriusFriends, CuriusTrail],
      pageBody: CuriusContent,
      afterBody: [],
    }
  } else if (slug === 'lyd') {
    components = { ...components, beforeBody: [], sidebar: [], afterBody: [] }
  }

  if (componentData.fileData.frontmatter?.poem) {
    components = { ...components, footer: PoetryFooter }
  }

  let isMenu = false
  if (componentData.fileData.frontmatter?.menu) {
    isMenu = true
    components = {
      ...components,
      header: [],
      beforeBody: [],
      sidebar: [],
      afterBody: [Functions],
      footer: MenuFooter,
    }
  }

  if (componentData.fileData.frontmatter?.pageLayout === 'letter-poem') {
    components = { ...components, header: [], sidebar: [], afterBody: [], beforeBody: [] }
  }

  const {
    head: Head,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    sidebar,
    footer: Footer,
  } = components

  // TODO: https://thesolarmonk.com/posts/a-spacebar-for-the-web style
  const lang =
    (componentData.fileData.frontmatter?.lang ?? componentData.cfg.locale)?.split('-')[0] ?? 'en'
  const pageLayout = componentData.fileData.frontmatter?.pageLayout ?? 'default'
  const isSlides = isSlidePage(slug, componentData.fileData.frontmatter?.slides)
  const isFlashcards = componentData.fileData.flashcards != null || pageLayout === 'flashcards'
  const isArena = slug === 'arena' || slug.startsWith('arena/')
  const isCurius = slug === 'curius'
  // The PDF reader shell renders once at `read` and the worker serves it for every `/read/...`
  // path; course PDF resource pages carry the same shell under their own slug.
  const isPdfReader =
    slug === 'read' ||
    pdfResourceOf(componentData.fileData) !== null ||
    componentData.fileData.pdfReader?.chrome === 'course'
  const isFullBleed = isArena || isPdfReader
  const isArenaSubpage = slug.startsWith('arena/') && slug !== 'arena'
  const isBase = componentData.fileData.bases ?? false
  const isCanvas = componentData.fileData.filePath?.endsWith('.canvas') ?? false

  const contentAttrs = { 'data-plain': !isBoxy }
  renderPerf.addEvent('preact')

  const body = render(
    <>
      <body
        data-slug={slug}
        data-language={lang}
        data-menu={isMenu}
        data-slides={isSlides}
        data-flashcards={isFlashcards}
        data-layout={pageLayout}
        data-pdf-reader={isPdfReader}
        data-is-folder-tag={isFolderTag}
        data-is-base={isBase}
        data-is-canvas={isCanvas}
        data-tags={(componentData.fileData.frontmatter?.tags ?? []).join(' ')}
        data-arena-subpage={isArenaSubpage}
        data-protected={componentData.fileData.frontmatter?.protected ?? false}
      >
        <main
          id="quartz-root"
          data-course-crumb={hasCourseCrumb(componentData.fileData) ? '' : undefined}
          class={classNames(undefined, 'page', slug === 'index' ? 'grid' : '')}
          style={
            slug === 'index'
              ? undefined
              : isSlides || isFlashcards
                ? { display: 'flex', flexDirection: 'column' }
                : isPdfReader
                  ? { display: 'flex', flexDirection: 'column', height: '100dvh' }
                  : { display: 'flex', flexDirection: 'column', minHeight: '100vh' }
          }
        >
          <Header {...componentData}>
            {header.map(HeaderComponent => (
              <HeaderComponent {...componentData} />
            ))}
          </Header>
          <section id="stacked-notes-container" class="all-col">
            <div id="stacked-notes-main">
              <div class="stacked-notes-column" />
            </div>
          </section>
          <div
            class={classNames(undefined, 'all-col', 'grid', 'page-body-grid')}
            style={{ flex: '1 1 auto' }}
          >
            {beforeBody.length > 0 && (
              <section
                class={classNames(
                  undefined,
                  'page-header',
                  'popover-hint',
                  isFullBleed ? 'all-col' : 'all-col grid',
                )}
              >
                {beforeBody.map(BodyComponent => (
                  <BodyComponent {...componentData} />
                ))}
              </section>
            )}
            <section
              class={classNames(
                undefined,
                'page-content',
                slug === 'index' ? 'side-col' : isFullBleed ? 'all-col' : 'grid all-col',
              )}
              {...contentAttrs}
            >
              {sidebar.length > 0 && (
                <aside class="aside-container left-col">
                  {sidebar.map(BodyComponent => (
                    <BodyComponent {...componentData} />
                  ))}
                </aside>
              )}
              <Content {...componentData} />
              {!isSlides && !isFlashcards && !isFullBleed && !isCurius && (
                <>
                  <div id="wc-modal" class="wc-modal">
                    <div class="wc-inner" />
                  </div>
                </>
              )}
              <Headings {...componentData} />
              <PressKit {...componentData} />
            </section>
            {!isCurius && !isPdfReader && (!isFolderTag || afterBody.length > 0) && (
              <section class="page-footer popover-hint grid all-col">
                {!isFolderTag &&
                  retrievalNodes.length > 0 &&
                  htmlToJsx(componentData.fileData.filePath!, {
                    type: 'root',
                    children: retrievalNodes,
                  } as Node)}
                {afterBody.length > 0 &&
                  afterBody.map(BodyComponent => <BodyComponent {...componentData} />)}
                {!isFolderTag && slug !== 'index' && <Footer {...componentData} />}
              </section>
            )}
          </div>
          <QuartzIcon filePath={componentData.fileData.filePath!} />
        </main>
      </body>
      {pageResources.js
        .filter(resource => resource.loadTime === 'afterDOMReady')
        .map(res => JSResourceToScriptElement(res, true))}
      {/* Cloudflare Web Analytics */}
      {!ctx.argv.serve && !ctx.argv.watch && (
        <script
          defer
          src={'https://static.cloudflareinsights.com/beacon.min.js'}
          data-cf-beacon='{"token": "3b6a9ecda4294f8bb5770c2bfb44078c"}'
          crossOrigin={'anonymous'}
          data-persist={true}
        />
      )}
      {/* End Cloudflare Web Analytics */}
    </>,
  )
  // The body renders first so the head links only the page-scoped sheets its markup uses.
  const { externalResources } = componentData
  const css = externalResources.css.filter(
    ({ classPrefixes }) =>
      !classPrefixes ||
      (classPrefixes.some(prefix => body.includes(prefix)) &&
        classPrefixPattern(classPrefixes).test(body)),
  )
  const head = render(
    <html lang={lang}>
      <Head {...componentData} externalResources={{ ...externalResources, css }} />
    </html>,
  )
  const html = `<!DOCTYPE html>${head.slice(0, -'</html>'.length)}${body}</html>`

  logBuildSpan(ctx.argv, 'render:preact', slug, renderPerf.elapsedMs('preact'))
  logBuildSpan(ctx.argv, 'render:total', slug, renderPerf.elapsedMs())
  return { html, fileData: componentData.fileData }
}
