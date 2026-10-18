import type { Element, Root } from 'hast'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { visit } from 'unist-util-visit'
import { mapConcurrent, defaultIoConcurrency } from '../../util/async-pool'
import { rgbToHex, spineInk } from '../../util/base/bookcase/geometry'
import { BuildCtx } from '../../util/ctx'
import {
  FilePath,
  FullSlug,
  QUARTZ,
  joinSegments,
  resolveRelative,
  slugifyFilePath,
} from '../../util/path'
import { write } from './helpers'

// 320 px covers the pulled cover face and the page panel at 2x.
const COVER_WIDTH = 320
const CACHE_VERSION = 1
const CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache', 'bookcase-covers')

type Cover = { key: string; color: string; data: Buffer }

const pending = new Map<string, Promise<Cover | null>>()
let slugIndex: { source: FilePath[]; map: Map<string, FilePath> } | undefined

function sourceFor(ctx: BuildCtx, slug: string): FilePath | undefined {
  if (slugIndex?.source !== ctx.allFiles) {
    const map = new Map<string, FilePath>()
    for (const file of ctx.allFiles) map.set(slugifyFilePath(file), file)
    slugIndex = { source: ctx.allFiles, map }
  }
  return slugIndex.map.get(slug)
}

async function readCache(key: string): Promise<Cover | null> {
  try {
    const [data, meta] = await Promise.all([
      fs.readFile(path.join(CACHE_DIR, `${key}.webp`)),
      fs.readFile(path.join(CACHE_DIR, `${key}.json`), 'utf8'),
    ])
    const { color } = JSON.parse(meta) as { color: string }
    return { key, color, data }
  } catch {
    return null
  }
}

async function processCover(file: string, relative: string): Promise<Cover | null> {
  const stat = await fs.stat(file)
  const key = createHash('sha1')
    .update(`${CACHE_VERSION}:${relative}:${stat.size}:${stat.mtimeMs}`)
    .digest('hex')
    .slice(0, 16)
  const cached = await readCache(key)
  if (cached) return cached

  const { width = 0, height = 0 } = await sharp(file, { failOn: 'none' }).metadata()
  if (width === 0 || height === 0) return null
  // The left edge of a cover scan sits next to the spine, so it is the closest colour to it.
  const strip = Math.max(1, Math.round(width * 0.06))
  const { channels } = await sharp(file, { failOn: 'none' })
    .extract({ left: 0, top: 0, width: strip, height })
    .stats()
  const [r, g = r, b = r] = channels.map(channel => channel.mean)
  const color = rgbToHex(r, g, b)
  const data = await sharp(file, { failOn: 'none' })
    .resize({ width: COVER_WIDTH, withoutEnlargement: true })
    .webp({ quality: 72 })
    .toBuffer()

  await fs.mkdir(CACHE_DIR, { recursive: true })
  await Promise.all([
    fs.writeFile(path.join(CACHE_DIR, `${key}.webp`), data),
    fs.writeFile(path.join(CACHE_DIR, `${key}.json`), JSON.stringify({ color })),
  ])
  return { key, color, data }
}

function coverFor(ctx: BuildCtx, slug: string): Promise<Cover | null> {
  const relative = sourceFor(ctx, slug)
  if (!relative) return Promise.resolve(null)
  const file = joinSegments(ctx.argv.directory, relative)
  let job = pending.get(file)
  if (!job) {
    job = processCover(file, relative).catch(() => null)
    pending.set(file, job)
    // Keep only settled results that a later edit can invalidate by mtime.
    void job.then(() => pending.delete(file))
  }
  return job
}

/**
 * Replaces the original cover URLs in rendered bookcase views with emitted thumbnails, and sets
 * each spine colour from its cover when the view does not name one.
 */
export async function emitBookcaseCovers(
  ctx: BuildCtx,
  views: { slug: FullSlug; tree: Root; view: { type: string } }[],
): Promise<FilePath[]> {
  const targets: { slug: FullSlug; node: Element; cover: string }[] = []
  for (const { slug, tree, view } of views) {
    if (view.type !== 'bookcase') continue
    visit(tree, 'element', (node: Element) => {
      const cover = node.properties?.dataBookcaseCover
      if (typeof cover === 'string' && cover) targets.push({ slug, node, cover })
    })
  }
  if (targets.length === 0) return []

  const covers = new Map<string, Promise<Cover | null>>()
  for (const { cover } of targets) if (!covers.has(cover)) covers.set(cover, coverFor(ctx, cover))
  const resolved = new Map<string, Cover | null>()
  await mapConcurrent([...covers], defaultIoConcurrency, async ([slug, job]) => {
    resolved.set(slug, await job)
  })

  for (const { slug, node, cover } of targets) {
    const result = resolved.get(cover)
    if (!result) continue
    const url = resolveRelative(slug, `static/bookcase/${result.key}.webp` as FullSlug)
    const properties = node.properties
    if (node.tagName === 'img') properties.src = url
    else properties.dataCover = url
    if (properties.dataSpineSampled !== undefined && typeof properties.style === 'string') {
      properties.style = properties.style.replace(
        /--spine:[^;]*;--spine-ink:[^;]*/,
        `--spine:${result.color};--spine-ink:${spineInk(result.color)}`,
      )
    }
  }

  const emitted = [...resolved.values()].filter((cover): cover is Cover => cover !== null)
  return mapConcurrent(emitted, defaultIoConcurrency, cover =>
    write({ ctx, slug: `static/bookcase/${cover.key}`, ext: '.webp', content: cover.data }),
  )
}
