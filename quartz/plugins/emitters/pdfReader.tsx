import type { Element, Root, RootContent } from 'hast'
import { toString } from 'hast-util-to-string'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { CourseResourceStrip } from '../../components/CourseResource'
import PdfReaderPage from '../../components/pages/PdfReader'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { courseDirOf, courseFileSlug, pdfResourceOf, resourceOf } from '../../util/course'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import {
  FilePath,
  FullSlug,
  joinSegments,
  pathToRoot,
  QUARTZ,
  slugifyFilePath,
} from '../../util/path'
import {
  normalizePdfSlug,
  parsePdfFragment,
  PDF_MANIFEST_PATH,
  PDF_READER_SHELL_PATH,
  type PdfCitation,
  type PdfManifest,
  type PdfReaderData,
} from '../../util/pdf-marks'
import { pdfTitlesFor } from '../../util/pdf-titles'
import { StaticResources } from '../../util/resources'
import { ProcessedContent, defaultProcessedContent } from '../vfile'
import { write } from './helpers'

declare module 'vfile' {
  interface DataMap {
    /** Set on a course PDF resource page before render; the reader page inlines it. */
    pdfReader?: PdfReaderData
  }
}

export const PDF_READER_SLUG = 'read' as FullSlug
export const PDF_READER_TITLE_SLOT = 'PDF_READER_TITLE_SLOT'

const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'pdf-documents.json')
const lfsPointer =
  /^version https:\/\/git-lfs\.github\.com\/spec\/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?$/
const excerptBlocks = new Set(['p', 'li', 'blockquote', 'td', 'dd', 'figcaption'])
const MAX_EXCERPT = 280

interface IdentityEntry {
  signature: string
  doc: string
  bytes: number
}

import { onModuleRelease } from '../../util/module-caches'

let identityCache: Map<string, IdentityEntry> | undefined
onModuleRelease(() => {
  identityCache = undefined
})

async function loadIdentityCache(): Promise<Map<string, IdentityEntry>> {
  if (identityCache) return identityCache
  try {
    const raw: unknown = JSON.parse(await fs.readFile(cacheFile, 'utf8'))
    const entries = typeof raw === 'object' && raw !== null ? Reflect.get(raw, 'files') : null
    identityCache = new Map(
      typeof entries === 'object' && entries !== null
        ? (Object.entries(entries) as [string, IdentityEntry][])
        : [],
    )
  } catch {
    identityCache = new Map()
  }
  return identityCache
}

async function hashFile(file: string): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file, { highWaterMark: 1 << 20 })) hash.update(chunk)
  return hash.digest('hex')
}

/** The LFS oid is the SHA-256 of the bytes, so pointer files and smudged checkouts agree. */
async function documentIdentity(
  ctx: BuildCtx,
  fp: FilePath,
  cache: Map<string, IdentityEntry>,
): Promise<IdentityEntry | null> {
  const source = joinSegments(ctx.argv.directory, fp)
  const info = await fs.stat(source).catch(() => null)
  if (!info) return null
  const signature = `${info.mtimeMs}:${info.size}`
  const cached = cache.get(fp)
  if (cached?.signature === signature) return cached

  let entry: IdentityEntry | null = null
  if (info.size < 1024) {
    const pointer = lfsPointer.exec(await fs.readFile(source, 'utf8').catch(() => ''))
    if (pointer) entry = { signature, doc: pointer[1], bytes: Number(pointer[2]) }
  }
  entry ??= { signature, doc: await hashFile(source), bytes: info.size }
  cache.set(fp, entry)
  return entry
}

function prettyTitle(slug: string): string {
  const base = path.posix.basename(slug).replace(/\.pdf$/i, '')
  return base.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim() || base
}

function compactText(value: string): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length > MAX_EXCERPT ? `${text.slice(0, MAX_EXCERPT - 1).trimEnd()}…` : text
}

function citationTarget(node: Element, from: string): { slug: string; hash: string } | null {
  // rehype-raw re-parses notes with raw HTML and camelCases data attributes.
  const raw =
    node.tagName === 'a'
      ? node.properties.href
      : node.tagName === 'div'
        ? (node.properties['data-pdf-src'] ?? node.properties.dataPdfSrc)
        : undefined
  if (typeof raw !== 'string' || raw.startsWith('#') || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(raw)) {
    return null
  }
  // Bare PDF paths are root-relative (transformResourceUrl), `./` and `../` are relative to the note.
  const href = /^\.\.?\//.test(raw) ? raw : `/${raw.replace(/^\/+/, '')}`
  const url = new URL(href, `https://garden.invalid/${from}`)
  const slug = normalizePdfSlug(url.pathname)
  return slug ? { slug, hash: url.hash } : null
}

function collectCitations(
  tree: Root,
  from: string,
  title: string,
  hosted: ReadonlySet<string>,
  out: Map<string, PdfCitation[]>,
  titles: Map<string, string>,
) {
  const seen = new Set<string>()
  const walk = (node: Root | RootContent, block: Element | undefined) => {
    if (node.type !== 'element' && node.type !== 'root') return
    let context = block
    if (node.type === 'element') {
      if (excerptBlocks.has(node.tagName)) context = node
      const target = citationTarget(node, from)
      if (target && hosted.has(target.slug)) {
        const { page, markId } = parsePdfFragment(target.hash)
        const key = `${target.slug}\0${page ?? ''}\0${markId ?? ''}`
        if (!seen.has(key)) {
          seen.add(key)
          const citation: PdfCitation = {
            from,
            title,
            excerpt: context ? compactText(toString(context)) : '',
          }
          if (page) citation.page = page
          if (markId) citation.markId = markId
          const list = out.get(target.slug) ?? []
          list.push(citation)
          out.set(target.slug, list)
        }
        if (node.tagName === 'div') {
          // An embed alias can name the document when its folder has no title override.
          const alias = node.properties['data-pdf-title'] ?? node.properties.dataPdfTitle
          if (
            typeof alias === 'string' &&
            alias.trim() &&
            !/\.pdf$/i.test(alias.trim()) &&
            !titles.has(target.slug)
          ) {
            titles.set(target.slug, alias.trim())
          }
          // An embed's loading placeholder carries no prose worth excerpting.
          return
        }
      }
    }
    for (const child of node.children) walk(child, context)
  }
  walk(tree, undefined)
}

export async function buildPdfManifest(
  ctx: BuildCtx,
  content: ProcessedContent[],
): Promise<PdfManifest> {
  const cache = await loadIdentityCache()
  const pdfs = ctx.allFiles.filter(fp => path.extname(fp).toLowerCase() === '.pdf')
  const identities = await mapConcurrent(pdfs, defaultIoConcurrency, async fp => ({
    slug: slugifyFilePath(fp),
    identity: await documentIdentity(ctx, fp, cache),
  }))

  const hosted = new Set(identities.filter(item => item.identity).map(item => item.slug))
  const citations = new Map<string, PdfCitation[]>()
  const titles = new Map<string, string>()
  const namedTitles = pdfTitlesFor(content.map(([, file]) => file.data))
  // A course resource page is the PDF's own page: canonical path, never a citation.
  const readPaths = new Map<string, string>()
  for (const [tree, file] of content) {
    const from = file.data.slug
    if (!from || from.endsWith('.pdf')) continue
    const resource = resourceOf(file.data)
    if (resource) {
      const pdf = pdfResourceOf(file.data)
      if (pdf)
        readPaths.set(
          courseFileSlug(pdf.dir, pdf.record.file!),
          `/${pdf.slug.replace(/\/index$/, '')}/`,
        )
      continue
    }
    const title = file.data.frontmatter?.title ?? from
    collectCitations(tree as Root, from, title, hosted, citations, titles)
  }

  const documents: PdfManifest['documents'] = {}
  for (const { slug, identity } of identities.sort((a, b) => a.slug.localeCompare(b.slug))) {
    if (!identity) continue
    const namedTitle = namedTitles.get(slug)
    const aliasTitle = titles.get(slug)
    documents[slug] = {
      doc: identity.doc,
      bytes: identity.bytes,
      title: namedTitle ?? aliasTitle ?? prettyTitle(slug),
      titleSource: namedTitle ? 'frontmatter' : aliasTitle ? 'alias' : 'filename',
      citedBy: (citations.get(slug) ?? []).sort((a, b) => a.from.localeCompare(b.from)),
      ...(readPaths.has(slug) ? { readPath: readPaths.get(slug) } : {}),
    }
  }

  const live = new Set<string>(pdfs)
  for (const key of cache.keys()) if (!live.has(key)) cache.delete(key)
  await fs.mkdir(path.dirname(cacheFile), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify({ version: 1, files: Object.fromEntries(cache) }))
  return { version: 1, documents }
}

export const PdfReader: QuartzEmitterPlugin = () => {
  const header = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
  })
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    header,
    pageBody: PdfReaderPage(),
    beforeBody: [],
    sidebar: [],
    afterBody: [],
  }
  // Course PDF pages: the strip above the reader, the shared header, no footer components.
  const resourceOpts: FullPageLayout = { ...opts, beforeBody: [CourseResourceStrip()] }
  let previousDocuments: PdfManifest['documents'] = {}

  async function emitManifest(ctx: BuildCtx, content: ProcessedContent[]) {
    const manifest = await buildPdfManifest(ctx, content)
    return {
      manifest,
      file: write({
        ctx,
        content: JSON.stringify(manifest),
        slug: PDF_MANIFEST_PATH.slice(1).replace(/\.json$/, '') as FullSlug,
        ext: '.json',
      }),
    }
  }

  /** Renders the course PDF resource pages whose folder index passes `pdfResourceOf`. */
  async function emitResourcePages(
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
    manifest: PdfManifest,
    include: (slug: string, pdfSlug: string) => boolean,
  ): Promise<FilePath[]> {
    const allFiles = contentDataFor(content)
    const pages = content.flatMap(([tree, file]) => {
      const ref = pdfResourceOf(file.data)
      if (!ref) return []
      const pdfSlug = courseFileSlug(ref.dir, ref.record.file!)
      if (!include(ref.slug, pdfSlug)) return []
      return [{ tree, file, ref, pdfSlug }]
    })
    const cfg = ctx.cfg.configuration
    return mapConcurrent(pages, defaultIoConcurrency, async ({ tree, file, ref, pdfSlug }) => {
      const record = manifest.documents[pdfSlug]
      // No record (R2): the page still emits with the strip and a static message, no mount.
      file.data.pdfReader = record
        ? {
            mode: 'document',
            slug: pdfSlug,
            readPath: `/${ref.slug.replace(/\/index$/, '')}/`,
            chrome: 'course',
            document: record,
          }
        : undefined
      const externalResources = pageResources(pathToRoot(ref.slug), resources, ctx)
      const componentData: QuartzComponentProps = {
        ctx,
        fileData: file.data,
        externalResources,
        cfg,
        children: [],
        tree,
        allFiles,
      }
      const html = renderPage(ctx, ref.slug, componentData, resourceOpts, externalResources)
      return write({ ctx, content: html, slug: ref.slug, ext: '.html' })
    })
  }

  return {
    name: 'PdfReader',
    getQuartzComponents() {
      return [opts.head, ...header, ...resourceOpts.beforeBody, opts.pageBody, opts.footer]
    },
    async *emit(ctx, content, resources) {
      const { manifest, file } = await emitManifest(ctx, content)
      previousDocuments = manifest.documents
      yield await file
      yield* await emitResourcePages(ctx, content, resources, manifest, () => true)

      const cfg = ctx.cfg.configuration
      const url = new URL(`https://${cfg.baseUrl ?? 'example.com'}`)
      const [tree, vfile] = defaultProcessedContent({
        slug: PDF_READER_SLUG,
        text: '',
        description: 'Read and annotate the PDFs hosted in this garden.',
        frontmatter: { title: PDF_READER_TITLE_SLOT, tags: [], pageLayout: 'default' },
      })
      // The worker serves this shell at every /read/* depth, so resources resolve from the root.
      const externalResources = pageResources(url.pathname as FullSlug, resources, ctx)
      const componentData: QuartzComponentProps = {
        ctx,
        fileData: vfile.data,
        externalResources,
        cfg,
        children: [],
        tree,
        allFiles: [],
      }
      const html = renderPage(ctx, PDF_READER_SLUG, componentData, opts, externalResources)
      yield write({
        ctx,
        // The shell renders at depth 0, so its relative links all mean the site root.
        content: html.replaceAll('href="./', 'href="/').replaceAll('src="./', 'src="/'),
        slug: PDF_READER_SHELL_PATH.slice(1),
        ext: '.html',
      })
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const touchesPdfs = changeEvents.some(event =>
        ['.md', '.pdf', '.fc'].includes(path.extname(event.path).toLowerCase()),
      )
      if (!touchesPdfs) return
      const { manifest, file } = await emitManifest(ctx, content)
      yield await file
      // R13: a note, deck or home edit changes the strip of every PDF page in that course.
      const dirs = new Set(
        changeEvents.flatMap(event => {
          const dir = courseDirOf(slugifyFilePath(event.path as FilePath))
          return dir ? [dir] : []
        }),
      )
      const changed = (pdfSlug: string) =>
        JSON.stringify(manifest.documents[pdfSlug]) !== JSON.stringify(previousDocuments[pdfSlug])
      previousDocuments = manifest.documents
      yield* await emitResourcePages(
        ctx,
        content,
        resources,
        manifest,
        (slug, pdfSlug) =>
          (courseDirOf(slug) !== null && dirs.has(courseDirOf(slug)!)) || changed(pdfSlug),
      )
    },
  }
}
