import { loadCuriusPagePreview } from '../curius-page-preview'
import { notebookPreviewDetails } from '../notebook-preview-card'
import {
  previewHref,
  previewText,
  renderPagePreview,
  type PagePreviewRow,
} from '../page-preview-card'

function customPageKind(page: Document, target: URL): string | null {
  const data = page.body.dataset
  const slug = data.slug ?? ''
  if (slug === 'curius') return 'curius'
  if (slug === 'arena' || slug.startsWith('arena/')) return 'arena'
  if (slug === 'stream' || slug.startsWith('stream/on/')) {
    // Entry permalinks keep the existing isolated-entry preview.
    return target.hash || target.searchParams.has('entry') ? null : 'stream'
  }
  if (data.pdfReader === 'true') return 'reader'
  if (data.flashcards === 'true') return 'flashcards'
  if (data.slides === 'true') return 'slides'
  if (page.querySelector('article.notebook-page')) return 'notebook'
  if (data.isFolderTag === 'true') return 'collection'
  if (data.menu === 'true') return 'menu'
  if (page.querySelector('.masonry-container')) return 'masonry'
  if (slug === 'index') return 'home'
  return data.layout && data.layout !== 'default' ? data.layout : null
}

function text(page: ParentNode, selector: string): string {
  return page.querySelector(selector)?.textContent?.trim() ?? ''
}

function pageRows(page: Document, target: URL, kind: string): PagePreviewRow[] {
  if (page.body.dataset.protected === 'true') return []
  const selectors: Record<string, string> = {
    arena: '.arena-channel-row-header h2 a, .arena-block-content',
    stream: '.stream-legend-title, .stream-entry-title',
    collection: '.section-ul .note-link .desc',
    masonry: '.masonry-image',
  }
  const selector = selectors[kind] ?? 'article h2, article h3'
  const rows: PagePreviewRow[] = []
  for (const node of page.querySelectorAll<HTMLElement>(selector)) {
    if (node.closest('[hidden], [data-skip-preview], [data-protected="true"]')) continue
    if (kind === 'notebook' && node.closest('.notebook-code-cell, .notebook-runtime-toolbar'))
      continue
    const title = previewText(
      node instanceof HTMLImageElement ? node.alt : (node.textContent ?? ''),
      100,
    )
    if (!title) continue
    const anchor = node.closest<HTMLAnchorElement>('a[href]') ?? node.querySelector('a[href]')
    const href = anchor ? previewHref(anchor.getAttribute('href') ?? '', target.href) : undefined
    rows.push({ title, href })
    if (rows.length === 4) break
  }
  return rows
}

/** Custom layouts expose content through fresh text nodes, without mounting their application. */
export async function renderCustomPagePreview(
  page: Document,
  target: URL,
  inner: HTMLDivElement,
  signal: AbortSignal,
): Promise<boolean> {
  if (target.origin !== window.location.origin) return false
  const kind = customPageKind(page, target)
  if (!kind) return false
  if (kind === 'curius') {
    const preview = await loadCuriusPagePreview(target, signal)
    if (preview && !signal.aborted) renderPagePreview(preview, inner)
    return true
  }
  const notebook = kind === 'notebook' ? notebookPreviewDetails(page) : null
  const rows = pageRows(page, target, kind)
  if (signal.aborted) return true
  const title = previewText(
    text(page, 'h1.article-title, h1') ||
      page.querySelector<HTMLMetaElement>('meta[property="og:title"]')?.content ||
      page.title ||
      target.pathname,
    100,
  )
  const description =
    notebook?.description ??
    page.querySelector<HTMLMetaElement>('meta[name="description"]')?.content ??
    ''
  renderPagePreview(
    {
      kind,
      title,
      href: target.href,
      description: previewText(description, 240),
      meta: notebook?.meta ?? [],
      rows,
    },
    inner,
  )
  return true
}
