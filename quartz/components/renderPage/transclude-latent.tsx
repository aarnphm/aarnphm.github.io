import type { Element, ElementContent, Root } from 'hast'
import { visit } from 'unist-util-visit'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { FullSlug } from '../../util/path'
import { renderDataFor, type BuildCtx } from '../../util/ctx'
import {
  pageHref,
  embedSettings,
  renderLatentEmbed,
  type EmbedResult,
} from '../../util/latent/embed/render'
import { axesSpec, openSidecar } from '../../util/latent/sidecar'
import { scopeHastIds } from '../../util/scope-element-ids'
import {
  hasHastClass,
  readTranscludeTarget,
  type TranscludeTarget,
} from '../../util/transclude-props'
import { type TranscludeScope } from './transclude-scope'

/**
 * `![[embeddings#focus=N#view=…]]`: a view of a `layout: latent` page quoted in a note
 * (docs/latent-embed.md). The figure needs the sidecar's vectors and the client's async panel
 * builders, so the content emitter prepares every embed a page holds before the page renders
 * (prepareLatentEmbeds); the synchronous transclusion pass then swaps the blockquote for the
 * prepared figure. It must not stay a blockquote: `blockquote.transclude:has(…)` rules would
 * anchor on it (site-has-restyle-cost).
 */

const prepared = new Map<string, Promise<EmbedResult | null>>()
const ready = new Map<string, EmbedResult | null>()

export function isLatentPage(page: QuartzPluginData | undefined): boolean {
  return page?.frontmatter?.pageLayout === 'latent'
}

function embedKey(page: QuartzPluginData, transclude: TranscludeTarget): string {
  return JSON.stringify([
    page.slug,
    page.latent?.hash ?? null,
    transclude.fragments,
    transclude.metadata ?? null,
    transclude.alias,
  ])
}

async function prepare(
  ctx: BuildCtx,
  page: QuartzPluginData,
  transclude: TranscludeTarget,
  origin: string,
): Promise<void> {
  const key = embedKey(page, transclude)
  if (ready.has(key)) return
  let pending = prepared.get(key)
  if (!pending) {
    pending = (async () => {
      const directory = page.latent?.directory ?? 'embeddings'
      const sidecar = openSidecar(ctx.argv.directory, directory, axesSpec(page.frontmatter?.axes))
      if (!sidecar) return null
      const t0 = performance.now()
      const caption = transclude.alias && transclude.alias !== 'undefined' ? transclude.alias : ''
      const result = await renderLatentEmbed({
        sidecar,
        pageSlug: page.slug ?? 'embeddings/index',
        scope: page.latent?.scope ?? [],
        fragments: transclude.fragments,
        settings: embedSettings(transclude.metadata),
        caption: caption || undefined,
        output: ctx.argv.output,
      })
      for (const warning of result.warnings) console.warn(`[latent] embed in ${origin}: ${warning}`)
      if (ctx.argv.verbose)
        console.log(`[latent] embed in ${origin}: ${(performance.now() - t0).toFixed(0)} ms`)
      return result
    })().catch(error => {
      console.warn(`[latent] embed in ${origin} failed:`, error)
      return null
    })
    prepared.set(key, pending)
  }
  ready.set(key, await pending)
  prepared.delete(key)
}

/**
 * Every latent embed the page's tree holds, and those of the notes it transcludes (to a small
 * depth), rendered before the page does. Cached per view for the process.
 */
export async function prepareLatentEmbeds(
  ctx: BuildCtx,
  tree: Root,
  bySlug: (slug: FullSlug) => QuartzPluginData | undefined,
  origin: string,
): Promise<void> {
  const seen = new Set<Root>()
  const jobs: Promise<void>[] = []
  const walk = (root: Root, depth: number) => {
    if (seen.has(root) || depth > 2) return
    seen.add(root)
    visit(root, 'element', node => {
      if (node.tagName !== 'blockquote' || !hasHastClass(node, 'transclude')) return
      const transclude = readTranscludeTarget(node)
      if (!transclude) return
      const page = bySlug(transclude.targetSlug)
      if (!page) return
      if (isLatentPage(page)) jobs.push(prepare(ctx, page, transclude, origin))
      else if (page.htmlAst) walk(page.htmlAst, depth + 1)
    })
  }
  walk(tree, 0)
  await Promise.all(jobs)
}

/**
 * The same, for emitters that render a page's content somewhere other than its own page: the
 * stream (/stream and its day routes) renders each entry's content, which an embed can sit in.
 * Views are cached per process, so every route that shows the entry reuses one render.
 */
export async function prepareLatentEmbedsFor(
  ctx: BuildCtx,
  allFiles: QuartzPluginData[],
  content: ElementContent[],
  origin: string,
): Promise<void> {
  if (content.length === 0) return
  const bySlug = renderDataFor(ctx, allFiles).bySlug
  await prepareLatentEmbeds(
    ctx,
    { type: 'root', children: content } as Root,
    s => bySlug.get(s),
    origin,
  )
}

function occurrence(scope: TranscludeScope): string {
  let prefix = `latent-${scope.expansion.occurrence}`
  while ([...scope.expansion.ids].some(id => id === prefix || id.startsWith(`${prefix}-`)))
    prefix += '-1'
  scope.expansion.ids.add(prefix)
  return prefix
}

/** Swap the blockquote for the prepared figure; false when the page is not a latent page. */
export function transcludeLatent(
  scope: TranscludeScope,
  node: Element,
  page: QuartzPluginData,
  transclude: TranscludeTarget,
): boolean {
  if (!isLatentPage(page)) return false
  const result = ready.get(embedKey(page, transclude))
  const figure: Element = result
    ? structuredClone(result.figure)
    : {
        // not prepared (an emitter other than the content page rendered it): a plain link
        type: 'element',
        tagName: 'figure',
        properties: { className: ['latent-embed'], dataState: 'notice' },
        children: [
          {
            type: 'element',
            tagName: 'a',
            properties: { className: ['internal'], href: pageHref(page.slug ?? 'embeddings') },
            children: [{ type: 'text', value: 'open this view of the embeddings ↗' }],
          },
        ],
      }
  const id = occurrence(scope)
  figure.properties.dataOccurrence = id
  scopeHastIds(figure, id)
  if (scope.componentData.fileData.frontmatter?.pageLayout === 'reflection') {
    visit(figure, 'element', (child, index, parent) => {
      if (parent && index !== undefined && hasHastClass(child, 'lat-embed-open')) {
        parent.children.splice(index, 1)
        return index
      }
    })
  }
  node.tagName = 'figure'
  node.properties = figure.properties
  node.children = figure.children
  // the blockquote's reference data stays on the node for dependency tracking
  return true
}
