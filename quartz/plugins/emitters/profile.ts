import type { Element } from 'hast'
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { visit } from 'unist-util-visit'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx } from '../../util/ctx'
import { FilePath, joinSegments, QUARTZ } from '../../util/path'
import { PROFILE_DIR } from '../transformers/profile'
import { ProcessedContent } from '../vfile'
import { write } from './helpers'

const CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache', 'profile')

function sources(content: ProcessedContent[]): Map<string, { rel: string; widths: number[] }> {
  const found = new Map<string, { rel: string; widths: number[] }>()
  for (const [tree, file] of content) {
    if (!file.data.profile) continue
    visit(tree, 'element', (node: Element) => {
      const value = node.properties?.dataProfileSource
      if (typeof value !== 'string') return
      const [rel, key, widths] = value.split('#')
      if (rel && key && widths) found.set(key, { rel, widths: widths.split(',').map(Number) })
    })
  }
  return found
}

async function variant(ctx: BuildCtx, key: string, rel: string, width: number) {
  const name = `${key}-${width}.webp`
  const cached = path.join(CACHE_DIR, name)
  let data: Buffer | undefined = await fs.readFile(cached).catch(() => undefined)
  if (!data) {
    data = await sharp(path.join(ctx.argv.directory, rel), { failOn: 'none' })
      .resize({ width, withoutEnlargement: true })
      .webp({ quality: 76 })
      .toBuffer()
    await fs.mkdir(CACHE_DIR, { recursive: true })
    await fs.writeFile(cached, data)
  }
  return write({ ctx, slug: `${PROFILE_DIR}/${key}-${width}`, ext: '.webp', content: data })
}

async function emitVariants(ctx: BuildCtx, content: ProcessedContent[]): Promise<FilePath[]> {
  const jobs = [...sources(content)].flatMap(([key, { rel, widths }]) =>
    widths.map(width => ({ key, rel, width })),
  )
  const written = await mapConcurrent(jobs, defaultIoConcurrency, ({ key, rel, width }) =>
    variant(ctx, key, rel, width).catch(() => null),
  )
  return written.filter((file): file is FilePath => file !== null)
}

/** Writes the resized srcset candidates the Profile transformer points at. */
export const ProfileFrames: QuartzEmitterPlugin = () => ({
  name: 'ProfileFrames',
  async emit(ctx, content) {
    return emitVariants(ctx, content)
  },
  async partialEmit(ctx, content) {
    return emitVariants(ctx, content)
  },
})
