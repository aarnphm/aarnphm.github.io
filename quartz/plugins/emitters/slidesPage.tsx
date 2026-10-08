import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { render } from 'preact-render-to-string'
import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { SlidesContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { prepareSlides, slidePageSlug, SlidesPageData } from '../../components/SlidesContent'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { escapeHTML } from '../../util/escape'
import { pathToRoot, joinSegments, FilePath, FullSlug } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write, removeWritten } from './helpers'

const emitterName = 'SlidesPage'

const deckSlugFor = (noteSlug: FullSlug): FullSlug => joinSegments(noteSlug, 'slides') as FullSlug
const fragmentDirFor = (noteSlug: FullSlug) => joinSegments('static', 'slides', noteSlug)

// The old single-page URL forwards to the first slide. Its #slide-N anchors were
// 0-based, so they map to page N + 1.
const redirectPage = (title: string) => `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHTML(title)}</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="slides/1">
<script>var m=/^#slide-(\\d+)$/.exec(location.hash);location.replace("slides/"+(m?+m[1]+1:1))</script>
<meta http-equiv="refresh" content="0; url=slides/1">
</head>
</html>
`

// Remove pages past the deck's end and fragments whose content changed.
async function pruneSlides(
  ctx: BuildCtx,
  noteSlug: FullSlug,
  pageCount: number,
  fragments: Set<string>,
): Promise<void> {
  const stale = async (dir: string, keep: (name: string) => boolean) => {
    const names = await fs.readdir(joinSegments(ctx.argv.output, dir)).catch(() => [])
    await Promise.all(
      names
        .filter(name => name.endsWith('.html') && !keep(name.slice(0, -5)))
        .map(name => removeWritten(ctx, joinSegments(dir, name.slice(0, -5)), '.html')),
    )
  }
  await stale(deckSlugFor(noteSlug), name => /^\d+$/.test(name) && Number(name) <= pageCount)
  await stale(fragmentDirFor(noteSlug), name => fragments.has(name))
}

async function deleteSlides(ctx: BuildCtx, noteSlug: FullSlug): Promise<void> {
  await removeWritten(ctx, deckSlugFor(noteSlug), '.html')
  await pruneSlides(ctx, noteSlug, 0, new Set())
}

async function processSlides(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const noteSlug = fileData.slug!
  const cfg = ctx.cfg.configuration
  // every slide page sits at the same depth, so they share relative resources
  const root = pathToRoot(slidePageSlug(noteSlug, 0))
  const externalResources = pageResources(root, resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const deck = prepareSlides(componentData)
  // content-addressed, so an unchanged slide keeps its URL and its cache entry
  const fragments = deck.sections.map((_, idx) => {
    const html = render(deck.body(idx))
    const hash = createHash('sha256').update(html).digest('hex').slice(0, 16)
    return { html, hash, slug: joinSegments(fragmentDirFor(noteSlug), hash) }
  })
  // extension-less: the asset server redirects a .html request to this form
  const urls = fragments.map(f => joinSegments(root, f.slug))

  const pages = deck.sections.map((_, active) => {
    const slug = slidePageSlug(noteSlug, active)
    const slidesPage: SlidesPageData = { deck, active, fragments: urls }
    const content = renderPage(
      ctx,
      slug,
      { ...componentData, slidesPage },
      opts,
      externalResources,
      false,
    )
    return write({ ctx, content, slug, ext: '.html' })
  })
  const written = await Promise.all([
    ...pages,
    ...fragments.map(f => write({ ctx, content: f.html, slug: f.slug as FullSlug, ext: '.html' })),
    write({
      ctx,
      content: redirectPage(fileData.frontmatter?.title ?? noteSlug),
      slug: deckSlugFor(noteSlug),
      ext: '.html',
    }),
  ])
  await pruneSlides(ctx, noteSlug, pages.length, new Set(fragments.map(f => f.hash)))
  return written
}

export const SlidesPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  // slim page layout for slides
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: SlidesContent(),
    ...userOpts,
    sidebar: [],
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: emitterName,
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        // skip tag pages and everything that isn’t a primary content page
        if (slug.endsWith('/index') || slug.startsWith('tags/')) continue
        // Only emit slides if explicitly enabled via frontmatter
        if (!file.data.frontmatter?.slides) continue
        yield* await processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)

      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type !== 'add' && changeEvent.type !== 'change') continue
        const slug = changeEvent.file.data.slug! as FullSlug
        if (changeEvent.file.data.frontmatter?.slides) {
          changedSlugs.add(slug)
        } else if (changeEvent.previousFile?.data.frontmatter?.slides) {
          await deleteSlides(ctx, slug)
        }
      }

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        if (slug.endsWith('/index') || slug.startsWith('tags/')) continue
        if (!file.data.frontmatter?.slides) continue
        yield* await processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
