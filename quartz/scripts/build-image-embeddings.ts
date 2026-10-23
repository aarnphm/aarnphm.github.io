import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { parseArgs } from 'node:util'
import type { LocalImageEncoder } from './image-embeddings/encoder'
import { readImageIgnorePatterns } from './image-embeddings/config'
import {
  IMAGE_ENCODER_CONTRACT,
  atomicWrite,
  imageVectorCache,
  publishImageGeneration,
  sha256,
  type ImageEncoderContract,
} from './image-embeddings/store'
import { collectImageSearchInventory } from './image-search-inventory'

interface ImageBuildReport {
  command: string
  status: string
  contract: ImageEncoderContract
  fingerprint: string
  inventoryDigest: string
  coverage: {
    localAssets: number
    publicAssets: number
    excludedAssets: number
    uniqueContents: number
  }
  encoded: number
  cacheHits: number
  pending: number
  failures: { sourcePath: string; reason: string }[]
  excluded: { sourcePath: string; reasons: string[] }[]
  inventoryIssues: { sourcePath: string; reference?: string; reason: string }[]
  generation?: string
  probes: Record<string, { id: string; score: number }[]>
  elapsedMs: number
}

const { values } = parseArgs({
  options: {
    content: { type: 'string', default: 'content' },
    'public-index': { type: 'string', default: 'public/static/contentIndex.json' },
    config: { type: 'string', default: 'quartz.config.ts' },
    out: { type: 'string' },
    cache: { type: 'string', default: 'quartz/.quartz-cache/image-embeddings' },
    report: { type: 'string', default: 'quartz/.quartz-cache/image-embeddings/report.json' },
    'inventory-only': { type: 'boolean', default: false },
    'include-excluded': { type: 'boolean', default: false },
    'max-new': { type: 'string' },
    'shard-size': { type: 'string', default: '1024' },
    threads: { type: 'string', default: '4' },
    backend: {
      type: 'string',
      default: process.platform === 'darwin' && process.arch === 'arm64' ? 'mlx' : 'onnx',
    },
    sample: { type: 'string', multiple: true },
    probe: { type: 'string', multiple: true },
  },
})

function positiveInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback
  const number = Number(value)
  if (!Number.isSafeInteger(number) || number < 1)
    throw new Error(`Expected a positive integer, received ${value}.`)
  return number
}

const started = performance.now()
const maxNew = positiveInteger(values['max-new'], Infinity)
const shardSize = positiveInteger(values['shard-size'], 1024)
const threads = positiveInteger(values.threads, 4)
const backend = values.backend
if (backend !== 'mlx' && backend !== 'onnx')
  throw new Error('Image encoder --backend must be mlx or onnx.')
const inventoryOptions = {
  contentDir: values.content,
  publicIndexPath: values['public-index'],
  ignorePatterns: await readImageIgnorePatterns(values.config),
}
const inventory = await collectImageSearchInventory(inventoryOptions)
const sample = values.sample
const allowedAssets = inventory.assets.filter(
  asset => !asset.exclusionReasons.includes('course-material'),
)
const assets = sample?.length
  ? allowedAssets.filter(
      asset => sample.includes(asset.sourcePath) || sample.includes(asset.assetId),
    )
  : values['include-excluded']
    ? allowedAssets
    : allowedAssets.filter(asset => asset.public)
if (
  sample?.some(
    requested =>
      !assets.some(asset => asset.sourcePath === requested || asset.assetId === requested),
  )
)
  throw new Error('One or more requested sample images were absent from the vault inventory.')
const outDir = path.resolve(
  values.out ??
    (sample?.length ? path.join(values.cache, 'sample-output') : 'content/embeddings/images'),
)
if (sample?.length && outDir === path.resolve('content/embeddings/images'))
  throw new Error(
    'Sample indexes require a separate --out directory to preserve the full public generation.',
  )
const contract =
  backend === 'mlx' && !values['inventory-only']
    ? await (await import('./image-embeddings/mlx-encoder')).mlxImageEncoderContract()
    : IMAGE_ENCODER_CONTRACT
const cache = imageVectorCache(values.cache, contract)
const report: ImageBuildReport = {
  command: `pnpm exec tsx quartz/scripts/build-image-embeddings.ts ${process.argv.slice(2).join(' ')}`,
  status: 'inventory',
  contract,
  fingerprint: cache.fingerprint,
  inventoryDigest: inventory.digest,
  coverage: {
    localAssets: inventory.assets.length,
    publicAssets: inventory.assets.filter(asset => asset.public).length,
    excludedAssets: inventory.assets.filter(asset => !asset.public).length,
    uniqueContents: new Set(assets.map(asset => asset.contentId).filter(Boolean)).size,
  },
  encoded: 0,
  cacheHits: 0,
  pending: 0,
  failures: [],
  excluded: inventory.assets
    .filter(asset => !asset.public)
    .map(asset => ({ sourcePath: asset.sourcePath, reasons: asset.exclusionReasons })),
  inventoryIssues: inventory.issues,
  probes: {},
  elapsedMs: 0,
}
// Reports include excluded local paths and never enter the public output directory.
const reportPath = path.resolve(values.report)
if (
  [outDir, path.resolve(values.content), path.resolve('public')].some(
    root => reportPath === root || reportPath.startsWith(`${root}${path.sep}`),
  )
)
  throw new Error(
    'The local image coverage report must remain outside content and the public index output.',
  )
const writeReport = async () => {
  report.elapsedMs = Math.round(performance.now() - started)
  await atomicWrite(reportPath, `${JSON.stringify(report, null, 2)}\n`)
}
await writeReport()
if (values['inventory-only']) {
  console.log(
    `Image inventory: ${report.coverage.localAssets} local, ${report.coverage.publicAssets} public, ${report.coverage.excludedAssets} excluded. Report: ${reportPath}`,
  )
} else {
  const vectors = new Map<string, Float32Array>()
  const unique = new Map(
    assets.filter(asset => asset.contentId).map(asset => [asset.contentId, asset]),
  )
  let encoder: LocalImageEncoder | undefined
  async function getEncoder(): Promise<LocalImageEncoder> {
    encoder ??=
      backend === 'mlx'
        ? await (await import('./image-embeddings/mlx-encoder')).createMlxImageEncoder(contract)
        : await (await import('./image-embeddings/encoder')).createImageEncoder(threads)
    return encoder
  }
  try {
    for (const [contentId, asset] of unique) {
      const cached = await cache.read(contentId)
      if (cached) {
        vectors.set(contentId, cached)
        report.cacheHits++
        continue
      }
      if (report.encoded >= maxNew) {
        report.pending++
        continue
      }
      try {
        // A changed source cannot be published under a stale inventory content identity.
        const bytes = await readFile(asset.absolutePath)
        if (sha256(bytes) !== contentId)
          throw new Error('Image bytes changed during inventory; rerun to refresh.')
        const vector = await (await getEncoder()).image(bytes)
        await cache.write(contentId, vector)
        vectors.set(contentId, vector)
        report.encoded++
        console.log(
          `Image embeddings: ${report.encoded} new, ${report.cacheHits} cached (${asset.sourcePath})`,
        )
        await writeReport()
      } catch (error) {
        report.failures.push({
          sourcePath: asset.sourcePath,
          reason: error instanceof Error ? error.message : String(error),
        })
        break
      }
    }
    if (report.failures.length || report.pending) {
      report.status = report.failures.length ? 'failed' : 'cached-only'
      console.log(
        `Image generation preserved: ${report.pending} hashes pending, ${report.failures.length} failures. Rerun to resume.`,
      )
    } else {
      const rows = assets
        .filter(asset => asset.public)
        .map(asset => {
          const vector = vectors.get(asset.contentId)
          if (!vector) throw new Error(`No vector for eligible image ${asset.sourcePath}.`)
          return { id: asset.assetId, vector }
        })
      const current = await collectImageSearchInventory(inventoryOptions)
      if (current.digest !== inventory.digest)
        throw new Error(
          'Vault image content or publication eligibility changed during indexing; rerun to resume safely.',
        )
      const manifest = await publishImageGeneration({
        outDir,
        rows,
        contract,
        shardSize,
        sourceDigests: Object.fromEntries(
          assets.filter(asset => asset.public).map(asset => [asset.assetId, asset.contentId]),
        ),
      })
      report.generation = manifest.generation
      report.status = 'published'
      for (const query of values.probe ?? []) {
        const queryVector = await (await getEncoder()).query(query)
        report.probes[query] = rows
          .map(row => ({
            id: row.id,
            score: row.vector.reduce(
              (sum, value, dimension) => sum + value * queryVector[dimension],
              0,
            ),
          }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 10)
      }
      console.log(
        `Published image embeddings: ${manifest.rows} rows, generation ${manifest.generation}.`,
      )
    }
  } finally {
    await encoder?.dispose()
    await writeReport()
  }
  if (report.failures.length) process.exitCode = 1
}
