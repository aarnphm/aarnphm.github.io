import { readFile, rm } from 'node:fs/promises'
import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { HeadingsConstructor, Content } from '../../components'
import HeaderConstructor from '../../components/Header'
import {
  pageResources,
  renderPage,
  CuriusContent,
  CuriusFriends,
  CuriusNavigation,
} from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { escapeHTML } from '../../util/escape'
import { FilePath, FullSlug, joinSegments, pathToRoot } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { StaticResources } from '../../util/resources'
import { PageTitlePatch, pageTitlePatchEvents } from '../../util/title-patch'
import { QuartzPluginData } from '../vfile'
import { write, writeKnownChanged } from './helpers'

const contentPageConcurrency = 32

const isContentPage = (fileData: QuartzPluginData): boolean => {
  const slug = fileData.slug
  if (!slug) return false
  return !(
    slug.endsWith('/index') ||
    slug.startsWith('tags/') ||
    fileData.bases ||
    fileData.jsonCanvas ||
    fileData.streamData ||
    fileData.flashcards ||
    fileData.frontmatter?.layout === 'masonry' ||
    fileData.frontmatter?.layout === 'triathlon'
  )
}

async function processContent(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = fileData.slug!
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const perf = new PerfTimer()
  const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
  logBuildSpan(ctx.argv, 'contentPage:render', slug, perf.elapsedMs())
  return write({ ctx, content, slug, ext: '.html' })
}

async function deleteContent(ctx: BuildCtx, slug: FullSlug): Promise<void> {
  const dest = joinSegments(ctx.argv.output, `${slug}.html`) as FilePath
  await rm(dest, { force: true })
}

function replaceRequired(html: string, search: string, replacement: string): string | undefined {
  const next = html.replace(search, replacement)
  return next === html ? undefined : next
}

function replaceOptional(html: string, search: string, replacement: string): string {
  return html.replace(search, replacement)
}

function patchContentPageTitleHtml(html: string, patch: PageTitlePatch): string | undefined {
  const previousTitle = escapeHTML(patch.previousTitle)
  const currentTitle = escapeHTML(patch.currentTitle)
  let next = html
  const withTitle = replaceRequired(
    next,
    `<title>${previousTitle}</title>`,
    `<title>${currentTitle}</title>`,
  )
  if (!withTitle) return undefined
  const withOpenGraph = replaceRequired(
    withTitle,
    `property="og:title" content="${previousTitle}"`,
    `property="og:title" content="${currentTitle}"`,
  )
  if (!withOpenGraph) return undefined
  const withTwitter = replaceRequired(
    withOpenGraph,
    `name="twitter:title" content="${previousTitle}"`,
    `name="twitter:title" content="${currentTitle}"`,
  )
  if (!withTwitter) return undefined
  next = withTwitter
  next = replaceOptional(
    next,
    `data-breadcrumbs="true">${previousTitle}</a>`,
    `data-breadcrumbs="true">${currentTitle}</a>`,
  )
  next = replaceOptional(
    next,
    `class="article-title">${previousTitle}</h1>`,
    `class="article-title">${currentTitle}</h1>`,
  )
  return next
}

async function patchContentPageTitle(
  ctx: BuildCtx,
  patch: PageTitlePatch,
): Promise<FilePath | undefined> {
  const pathToPage = joinSegments(ctx.argv.output, `${patch.slug}.html`) as FilePath
  const currentHtml = await readFile(pathToPage, 'utf8')
  const patchedHtml = patchContentPageTitleHtml(currentHtml, patch)
  if (!patchedHtml || patchedHtml === currentHtml) return undefined
  return writeKnownChanged({ ctx, content: patchedHtml, slug: patch.slug, ext: '.html' })
}

export const ContentPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Headings = HeadingsConstructor()

  return {
    name: 'ContentPage',
    getQuartzComponents() {
      return [
        Head,
        Header,
        CuriusFriends,
        CuriusContent,
        CuriusNavigation,
        Headings,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...sidebar,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)
      const pages = content.filter(([, file]) => isContentPage(file.data))
      const files = await mapConcurrent(pages, contentPageConcurrency, ([tree, file]) =>
        processContent(ctx, tree, file.data, allFiles, opts, resources),
      )

      yield* files
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const titlePatches = pageTitlePatchEvents(changeEvents)
      if (
        titlePatches &&
        changeEvents.every(changeEvent => changeEvent.file && isContentPage(changeEvent.file.data))
      ) {
        const perf = new PerfTimer()
        const files = await mapConcurrent(titlePatches, defaultIoConcurrency, patch =>
          patchContentPageTitle(ctx, patch),
        )
        if (files.every(file => file !== undefined)) {
          logBuildSpan(
            ctx.argv,
            'contentPage:titlePatch',
            `${files.length} files`,
            perf.elapsedMs(),
          )
          yield* files
          return
        }
      }

      const allFiles = contentDataFor(content)

      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        const fileData = changeEvent.file.data
        if (changeEvent.type === 'delete') {
          if (isContentPage(fileData)) {
            await deleteContent(ctx, fileData.slug as FullSlug)
          }
          continue
        }
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          changedSlugs.add(fileData.slug!)
        }
      }

      const pages = content.filter(([, file]) => {
        const slug = file.data.slug!
        return changedSlugs.has(slug) && isContentPage(file.data)
      })
      const files = await mapConcurrent(pages, contentPageConcurrency, ([tree, file]) =>
        processContent(ctx, tree, file.data, allFiles, opts, resources),
      )

      yield* files
    },
  }
}
