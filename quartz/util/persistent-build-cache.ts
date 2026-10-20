import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import type { BuildCtx } from './ctx'
import { isParseSourceInput } from '../plugins/emitters/component-resources/change-classifier'
import {
  PROCESSED_CONTENT_CACHE_VERSION,
  restoreProcessedContentCache,
  type ProcessedContentSnapshot,
} from '../processors/parse'
import { defaultIoConcurrency, mapConcurrent } from './async-pool'
import { emitQuartzCacheReceipt, type QuartzCacheReceipt } from './cache-receipts'
import { pruneStaleOutputStages } from './staged-output'
import { isRecord } from './type-guards'

declare global {
  var __quartzBuildSourceInputs: string[] | undefined
}

const schema = z.object({
  version: z.literal(PROCESSED_CONTENT_CACHE_VERSION),
  inputs: z.string(),
  content: z.string(),
  parseIdentity: z.string(),
  published: z.array(z.string()),
  outputs: z.string(),
  blob: z.string().regex(/^parse-[a-f0-9-]+\.bin$/),
  digest: z.string(),
})
type Manifest = z.infer<typeof schema>
export type BuildInputs = { inputs: string; content: string; parseIdentity: string }

const hash = (value: string | Uint8Array): string =>
  createHash('sha256').update(value).digest('hex')
const shellEnvironment = new Set([
  '_',
  'SHLVL',
  'PWD',
  'OLDPWD',
  'TERM_SESSION_ID',
  'ITERM_SESSION_ID',
])
const cacheDirectory = (ctx: BuildCtx): string =>
  path.join(
    path.dirname(path.resolve(ctx.argv.output)),
    '.quartz-cache',
    `build-${hash(JSON.stringify([process.cwd(), path.resolve(ctx.argv.directory), path.resolve(ctx.argv.output)])).slice(0, 20)}`,
  )

function receipt(
  ctx: BuildCtx,
  decision: Omit<QuartzCacheReceipt, 'epoch' | 'profile' | 'manifestPath'>,
): void {
  emitQuartzCacheReceipt({
    epoch: ctx.buildId,
    profile: ctx.argv.watch
      ? 'watch'
      : process.env.NODE_ENV === 'production'
        ? 'production'
        : 'build',
    manifestPath: path.join(cacheDirectory(ctx), 'manifest.json'),
    ...decision,
  })
}

function missing(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT'
}

async function signature(filename: string): Promise<string> {
  return fs.stat(filename, { bigint: true }).then(
    info => `${filename}:${info.size}:${info.mtimeNs}:${info.ctimeNs}:${info.ino}:${info.mode}`,
    error => {
      if (error.code === 'ENOENT') return `${filename}:missing`
      throw error
    },
  )
}

async function filesIn(
  directory: string,
  excluded: (relative: string) => boolean = () => false,
): Promise<string[]> {
  const files: string[] = []
  async function walk(relative: string): Promise<void> {
    const entries = await fs
      .readdir(path.join(directory, relative), { withFileTypes: true })
      .catch(error => {
        if (error.code === 'ENOENT') return []
        throw error
      })
    const subdirs: string[] = []
    for (const entry of entries) {
      const next = path.join(relative, entry.name)
      if (excluded(next)) continue
      if (entry.isDirectory()) subdirs.push(next)
      else files.push(path.resolve(directory, next))
    }
    if (subdirs.length > 0) await Promise.all(subdirs.map(walk))
  }
  await walk('')
  return files.sort()
}

async function fingerprint(files: string[], memo?: Map<string, Promise<string>>): Promise<string> {
  const readSig = memo
    ? (filename: string) => {
        const key = path.resolve(filename)
        let pending = memo.get(key)
        if (!pending) {
          pending = fs.stat(key, { bigint: true }).then(
            info => `${info.size}:${info.mtimeNs}:${info.ctimeNs}:${info.ino}:${info.mode}`,
            error => {
              if (error.code === 'ENOENT') return 'missing'
              throw error
            },
          )
          memo.set(key, pending)
        }
        return pending.then(sig => `${filename}:${sig}`)
      }
    : signature
  return hash(
    (await mapConcurrent([...new Set(files)].sort(), defaultIoConcurrency, readSig)).join('\n'),
  )
}

// Generated runtime artifacts are outputs. Provider caches remain inputs, including newly added
// provider directories. A cache produced during a build delays admission until inputs stabilize.
const generatedCache = (relative: string): boolean =>
  /^(?:transpiled-|output-assets\.json$|pdf-documents\.json$|masonry-image-sizes\.json(?:$|\.\d+\.\d+\.tmp$)|triathlon\/data\.jsonl$|notebook-pages(?:\/|$)|notebook-assets(?:\/|$)|fonts(?:\/|$)|tikz(?:\/|$)|bookcase-covers(?:\/|$)|og-images(?:\/|$)|twitter-posts(?:\/|$)|citations\.json$)/.test(
    relative,
  )

export async function captureBuildInputs(ctx: BuildCtx): Promise<BuildInputs | undefined> {
  const reason = !ctx.argv.watch
    ? 'non-watch'
    : ctx.argv.force
      ? 'forced'
      : !globalThis.__quartzParseSourceInputs
        ? 'missing-parser-inputs'
        : undefined
  if (reason) {
    receipt(ctx, { phase: 'inputs', status: 'disabled', reason })
    return undefined
  }
  try {
    return await buildInputs(ctx)
  } catch (error) {
    console.warn('Unable to fingerprint persistent build inputs', error)
    receipt(ctx, { phase: 'inputs', status: 'disabled', reason: 'inputs-unavailable' })
    return undefined
  }
}

async function buildInputs(ctx: BuildCtx): Promise<BuildInputs> {
  const [sources, skills, providers] = await Promise.all([
    filesIn('quartz', relative => relative === '.quartz-cache'),
    filesIn('.agents/skills'),
    filesIn('quartz/.quartz-cache', generatedCache),
  ])
  const dependencies = ctx.cfg.plugins.transformers.flatMap(
    plugin => plugin.fileDependencies?.(ctx) ?? [],
  )
  const common = JSON.stringify([
    PROCESSED_CONTENT_CACHE_VERSION,
    process.versions.node,
    process.versions.v8,
    process.platform,
    process.arch,
    Intl.DateTimeFormat().resolvedOptions().timeZone,
    new Date().toDateString(),
    process.cwd(),
    path.resolve(ctx.argv.directory),
    path.resolve(ctx.argv.output),
    ctx.gitCommitSha,
    ctx.cfg.configuration,
    ctx.argv,
    Object.entries(process.env)
      .filter(([key]) => !shellEnvironment.has(key))
      .sort(([a], [b]) => a.localeCompare(b)),
    ctx.cfg.plugins.transformers.map(plugin => plugin.name),
    ctx.cfg.plugins.filters.map(plugin => plugin.name),
    ctx.cfg.plugins.emitters.map(plugin => plugin.name),
  ])
  const memo = new Map<string, Promise<string>>()
  const [external, parseSourcesHash, content, sourceTreeHash] = await Promise.all([
    fingerprint([...providers, ...dependencies, 'package.json', 'pnpm-lock.yaml', '.env'], memo),
    fingerprint((globalThis.__quartzParseSourceInputs ?? []).filter(isParseSourceInput), memo),
    fingerprint(
      ctx.allFiles.map(file => path.resolve(ctx.argv.directory, file)),
      memo,
    ),
    fingerprint(
      [
        ...sources,
        ...skills,
        ...(globalThis.__quartzBuildSourceInputs ?? []),
        'quartz.config.ts',
        'quartz.layout.ts',
      ],
      memo,
    ),
  ])
  const parseIdentity = hash(
    JSON.stringify([common, external, ctx.allFiles.slice().sort(), parseSourcesHash]),
  )
  return {
    content,
    parseIdentity,
    inputs: hash(JSON.stringify([common, external, sourceTreeHash, content])),
  }
}

export async function resourceBuildCheckpoint(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
): Promise<Manifest | undefined> {
  if (!inputs) return undefined
  try {
    const manifest = schema.parse(
      JSON.parse(await fs.readFile(path.join(cacheDirectory(ctx), 'manifest.json'), 'utf8')),
    )
    return manifest.parseIdentity === inputs.parseIdentity &&
      manifest.content === inputs.content &&
      manifest.outputs === (await outputFingerprint(ctx))
      ? manifest
      : undefined
  } catch {
    return undefined
  }
}

export async function commitResourceBuildCache(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
  previous: Manifest | undefined,
): Promise<void> {
  if (!inputs) return
  if (!previous) {
    receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'checkpoint-changed' })
    return
  }
  try {
    const after = await captureBuildInputs(ctx)
    if (after?.inputs !== inputs.inputs) {
      receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'inputs-unstable' })
      return
    }
    const directory = cacheDirectory(ctx)
    const current = schema.parse(
      JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8')),
    )
    if (current.blob !== previous.blob || current.inputs !== previous.inputs) {
      receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'checkpoint-changed' })
      return
    }
    const manifest: Manifest = { ...previous, ...inputs, outputs: await outputFingerprint(ctx) }
    const temporary = path.join(directory, `manifest-${randomUUID()}.tmp`)
    await fs.writeFile(temporary, JSON.stringify(manifest), { flag: 'wx' })
    await fs.rename(temporary, path.join(directory, 'manifest.json'))
    receipt(ctx, { phase: 'admission', status: 'saved', reason: 'resources-updated' })
  } catch (error) {
    console.warn('Unable to update persistent resource checkpoint', error)
    receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'unavailable' })
  }
}

async function outputFingerprint(ctx: BuildCtx): Promise<string> {
  // Include the file inventory and identities, so additions, removals and in-place edits invalidate.
  const index = await fs.stat(path.join(ctx.argv.output, 'index.html'))
  if (!index.isFile() || index.size === 0) throw new Error('Published index is absent')
  return fingerprint([
    // The dev manager owns local models independently of Quartz publication.
    ...(await filesIn(ctx.argv.output, relative => relative === 'models')),
    path.resolve('quartz/.quartz-cache/triathlon/data.jsonl'),
    path.resolve('quartz/.quartz-cache/pdf-documents.json'),
  ])
}

export async function restoreBuildCache(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
): Promise<{ unchanged: boolean; published: string[] } | undefined> {
  if (!inputs) return undefined
  const directory = cacheDirectory(ctx)
  let raw: unknown
  try {
    raw = JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8'))
  } catch (error) {
    receipt(ctx, {
      phase: 'restore',
      status: 'miss',
      reason: missing(error)
        ? 'missing-manifest'
        : error instanceof SyntaxError
          ? 'corrupt-manifest'
          : 'unavailable',
    })
    return undefined
  }
  const decoded = schema.safeParse(raw)
  if (!decoded.success) {
    receipt(ctx, {
      phase: 'restore',
      status: 'miss',
      reason:
        isRecord(raw) && raw.version !== PROCESSED_CONTENT_CACHE_VERSION
          ? 'incompatible-manifest'
          : 'corrupt-manifest',
    })
    return undefined
  }
  const manifest = decoded.data
  if (manifest.parseIdentity !== inputs.parseIdentity) {
    receipt(ctx, { phase: 'restore', status: 'miss', reason: 'parse-inputs-changed' })
    return undefined
  }
  try {
    const bytes = await fs.readFile(path.join(directory, manifest.blob))
    if (hash(bytes) !== manifest.digest || !restoreProcessedContentCache(bytes)) {
      receipt(ctx, { phase: 'restore', status: 'miss', reason: 'corrupt-snapshot' })
      return undefined
    }
  } catch (error) {
    receipt(ctx, {
      phase: 'restore',
      status: 'miss',
      reason: missing(error)
        ? 'missing-snapshot'
        : isRecord(error) && error.code === 'EACCES'
          ? 'unavailable'
          : 'corrupt-snapshot',
    })
    return undefined
  }
  const sameInputs = manifest.inputs === inputs.inputs
  const sameOutputs =
    sameInputs &&
    (await outputFingerprint(ctx).then(
      value => value === manifest.outputs,
      () => false,
    ))
  const unchanged = sameInputs && sameOutputs
  receipt(
    ctx,
    unchanged
      ? {
          phase: 'restore',
          status: 'output-hit',
          reason: 'unchanged',
          counts: { parsed: 0, rendered: 0, outputWrites: 0 },
        }
      : {
          phase: 'restore',
          status: 'parse-hit',
          reason: sameInputs ? 'output-changed' : 'build-inputs-changed',
        },
  )
  if (unchanged) {
    void pruneStaleOutputStages(ctx.argv.output).catch(error => {
      console.warn('Could not prune abandoned output stages:', error)
    })
  }
  return { unchanged, published: manifest.published }
}

export async function commitUnchangedParseBuildCache(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
  published: string[],
): Promise<boolean> {
  if (!inputs) return false
  try {
    const directory = cacheDirectory(ctx)
    const previous = schema.parse(
      JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8')),
    )
    if (
      previous.parseIdentity !== inputs.parseIdentity ||
      previous.content !== inputs.content ||
      previous.blob.length === 0
    ) {
      return false
    }
    await fs.access(path.join(directory, previous.blob))
    const after = await captureBuildInputs(ctx)
    if (after?.inputs !== inputs.inputs || after.parseIdentity !== inputs.parseIdentity) {
      receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'inputs-unstable' })
      return true
    }
    if (after.content !== inputs.content) {
      return false
    }
    const manifest: Manifest = {
      ...previous,
      ...inputs,
      published,
      outputs: await outputFingerprint(ctx),
    }
    const temporary = path.join(directory, `manifest-${randomUUID()}.tmp`)
    await fs.writeFile(temporary, JSON.stringify(manifest), { flag: 'wx' })
    await fs.rename(temporary, path.join(directory, 'manifest.json'))
    receipt(ctx, { phase: 'admission', status: 'saved', reason: 'parse-unchanged' })
    return true
  } catch {
    return false
  }
}

export async function commitBuildCache(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
  snapshot: ProcessedContentSnapshot | undefined,
  published: string[],
): Promise<void> {
  if (!inputs) return
  if (!snapshot || !('bytes' in snapshot)) {
    receipt(ctx, {
      phase: 'admission',
      status: 'skipped',
      reason: snapshot?.reason ?? 'missing-snapshot-result',
      detail: snapshot?.detail,
    })
    return
  }
  const { bytes } = snapshot
  try {
    const after = await captureBuildInputs(ctx)
    if (inputs.inputs !== after?.inputs || inputs.parseIdentity !== after.parseIdentity) {
      console.log('Persistent build cache skipped: inputs changed during the build')
      receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'inputs-unstable' })
      return
    }
    const directory = cacheDirectory(ctx)
    const previous = await fs
      .readFile(path.join(directory, 'manifest.json'), 'utf8')
      .then(
        text => schema.safeParse(JSON.parse(text)),
        () => undefined,
      )
      .catch(() => undefined)
    const blob = `parse-${randomUUID()}.bin`
    const manifest: Manifest = {
      version: PROCESSED_CONTENT_CACHE_VERSION,
      ...inputs,
      published,
      outputs: await outputFingerprint(ctx),
      blob,
      digest: hash(bytes),
    }
    await fs.mkdir(directory, { recursive: true })
    await fs.writeFile(path.join(directory, blob), bytes, { flag: 'wx' })
    const temporary = path.join(directory, `manifest-${randomUUID()}.tmp`)
    await fs.writeFile(temporary, JSON.stringify(manifest), { flag: 'wx' })
    await fs.rename(temporary, path.join(directory, 'manifest.json'))
    if (previous?.success && previous.data.blob !== blob) {
      await fs.rm(path.join(directory, previous.data.blob), { force: true }).catch(error => {
        console.warn('Unable to remove superseded parse cache', error)
      })
    }
    receipt(ctx, { phase: 'admission', status: 'saved', reason: 'published' })
  } catch (error) {
    console.warn('Unable to save persistent build cache', error)
    receipt(ctx, { phase: 'admission', status: 'skipped', reason: 'unavailable' })
  }
}
