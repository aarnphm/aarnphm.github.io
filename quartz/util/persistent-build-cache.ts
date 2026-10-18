import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import type { BuildCtx } from './ctx'
import { restoreProcessedContentCache } from '../processors/parse'
import { defaultIoConcurrency, mapConcurrent } from './async-pool'

declare global {
  var __quartzBuildSourceInputs: string[] | undefined
}

const schema = z.object({
  version: z.literal(1),
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
    for (const entry of entries) {
      const next = path.join(relative, entry.name)
      if (excluded(next)) continue
      if (entry.isDirectory()) await walk(next)
      else files.push(path.resolve(directory, next))
    }
  }
  await walk('')
  return files.sort()
}

async function fingerprint(files: string[]): Promise<string> {
  return hash(
    (await mapConcurrent([...new Set(files)].sort(), defaultIoConcurrency, signature)).join('\n'),
  )
}

// Generated runtime artifacts are outputs. Provider caches remain inputs, including newly added
// provider directories. A cache produced during a build delays admission until inputs stabilize.
const generatedCache = (relative: string): boolean =>
  /^(?:transpiled-|output-assets\.json$|pdf-documents\.json$|triathlon\/data\.jsonl$|notebook-pages(?:\/|$)|notebook-assets(?:\/|$)|fonts(?:\/|$)|tikz(?:\/|$)|bookcase-covers(?:\/|$)|og-images(?:\/|$)|citations\.json$)/.test(
    relative,
  )

export async function captureBuildInputs(ctx: BuildCtx): Promise<BuildInputs | undefined> {
  if (!ctx.argv.watch || ctx.argv.force || !globalThis.__quartzParseSourceInputs) return undefined
  try {
    return await buildInputs(ctx)
  } catch (error) {
    console.warn('Unable to fingerprint persistent build inputs', error)
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
    1,
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
  const external = await fingerprint([
    ...providers,
    ...dependencies,
    'package.json',
    'pnpm-lock.yaml',
    '.env',
  ])
  const parseIdentity = hash(
    JSON.stringify([
      common,
      external,
      ctx.allFiles.slice().sort(),
      await fingerprint(globalThis.__quartzParseSourceInputs ?? []),
    ]),
  )
  const content = await fingerprint(
    ctx.allFiles.map(file => path.resolve(ctx.argv.directory, file)),
  )
  return {
    content,
    parseIdentity,
    inputs: hash(
      JSON.stringify([
        common,
        external,
        await fingerprint([
          ...sources,
          ...skills,
          ...(globalThis.__quartzBuildSourceInputs ?? []),
          'quartz.config.ts',
          'quartz.layout.ts',
        ]),
        content,
      ]),
    ),
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
  if (!inputs || !previous) return
  try {
    const after = await captureBuildInputs(ctx)
    if (after?.inputs !== inputs.inputs) return
    const directory = cacheDirectory(ctx)
    const current = schema.parse(
      JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8')),
    )
    if (current.blob !== previous.blob || current.inputs !== previous.inputs) return
    const manifest: Manifest = { ...previous, ...inputs, outputs: await outputFingerprint(ctx) }
    const temporary = path.join(directory, `manifest-${randomUUID()}.tmp`)
    await fs.writeFile(temporary, JSON.stringify(manifest), { flag: 'wx' })
    await fs.rename(temporary, path.join(directory, 'manifest.json'))
  } catch (error) {
    console.warn('Unable to update persistent resource checkpoint', error)
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
  try {
    const directory = cacheDirectory(ctx)
    const manifest = schema.parse(
      JSON.parse(await fs.readFile(path.join(directory, 'manifest.json'), 'utf8')),
    )
    if (manifest.parseIdentity !== inputs.parseIdentity) return undefined
    const bytes = await fs.readFile(path.join(directory, manifest.blob))
    if (hash(bytes) !== manifest.digest || !restoreProcessedContentCache(bytes)) return undefined
    const unchanged =
      manifest.inputs === inputs.inputs && manifest.outputs === (await outputFingerprint(ctx))
    return { unchanged, published: manifest.published }
  } catch {
    // Persistence is an optimization. An incomplete or incompatible generation uses the normal build.
    return undefined
  }
}

export async function commitBuildCache(
  ctx: BuildCtx,
  inputs: BuildInputs | undefined,
  bytes: Buffer | undefined,
  published: string[],
): Promise<void> {
  if (!inputs || !bytes) return
  try {
    const after = await captureBuildInputs(ctx)
    if (inputs.inputs !== after?.inputs || inputs.parseIdentity !== after.parseIdentity) {
      console.log('Persistent build cache skipped: inputs changed during the build')
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
      version: 1,
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
    // Only immutable, superseded parse generations in this build's own directory are disposable.
    if (previous?.success && previous.data.blob !== blob)
      await fs.rm(path.join(directory, previous.data.blob), { force: true })
  } catch (error) {
    console.warn('Unable to save persistent build cache', error)
  }
}
