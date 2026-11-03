import { createHash } from 'node:crypto'
import { copyFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { isRecord, readNumber, readString } from '../util/type-guards'
import { latestModelArchiveEntry } from './pace-model-archive'
import { IMMUTABLE_CC, POINTER_CC, PUBLIC_BASE, r2put } from './pace-r2'

// The latent explorer's sparse autoencoders, published the way the triathlon models are:
// features.py archives every training run under .models-archive/latent-<dictionary>/<ts>-vN/,
// and this promotes the newest to R2 as models/latent-<dictionary>/v<N>/ with a latest.json
// pointer. Each version carries the weights (safetensors), the manifest, a model card, the
// dictionary's labels, and the two scripts that made them.
//
//   pnpm models:latent:publish            # PACE_PUBLISH_DRY=1 prints the uploads instead
//   pnpm models:latent:publish --force    # republish even when nothing changed

const DICTS = ['text', 'image', 'base'] as const
const ARCHIVE = process.env.PACE_ARCHIVE ?? '.models-archive'
const LABELS = process.env.LATENT_LABELS ?? 'content/embeddings/latent/labels.json'
const SOURCES = ['quartz/scripts/latent/features.py', 'quartz/scripts/latent/label.py']
const REGRESS_TOL = 0.1
const FORCE = process.argv.includes('--force')

interface LiveLatent {
  version: number
  sha256: string
  labelsHash: string
  fvu: number
  arch: string
}

function archKey(arch: unknown): string {
  if (!isRecord(arch)) return ''
  const prefixes = Array.isArray(arch.prefixes) ? arch.prefixes.join('/') : ''
  return `${readString(arch, 'type') ?? ''}:${readNumber(arch, 'features') ?? 0}:${readNumber(arch, 'k') ?? 0}:${prefixes}`
}

async function fetchLive(family: string): Promise<LiveLatent | null> {
  try {
    const res = await fetch(`${PUBLIC_BASE}/models/${family}/latest.json`)
    if (!res.ok) return null
    const json: unknown = await res.json()
    if (!isRecord(json)) return null
    const val = isRecord(json.val) ? json.val : {}
    return {
      version: readNumber(json, 'version') ?? 0,
      sha256: readString(json, 'sha256') ?? '',
      labelsHash: readString(json, 'labelsHash') ?? '',
      fvu: readNumber(val, 'fvu') ?? Number.POSITIVE_INFINITY,
      arch: archKey(json.arch),
    }
  } catch {
    return null
  }
}

function familyLabels(name: string): unknown {
  if (!existsSync(LABELS)) return {}
  const all: unknown = JSON.parse(readFileSync(LABELS, 'utf8'))
  return isRecord(all) && isRecord(all[name]) ? all[name] : {}
}

async function publishDict(name: string): Promise<void> {
  const family = `latent-${name}`
  const entry = latestModelArchiveEntry(join(ARCHIVE, family))
  if (!entry) {
    console.log(`${family}: nothing archived`)
    return
  }
  const vdir = entry.path
  const manifestRaw: unknown = JSON.parse(readFileSync(join(vdir, 'manifest.json'), 'utf8'))
  if (!isRecord(manifestRaw)) throw new Error(`${family}: manifest is not an object`)
  const st = readFileSync(join(vdir, 'model.safetensors'))
  const sha = createHash('sha256').update(st).digest('hex')
  if (sha !== readString(manifestRaw, 'sha256'))
    throw new Error(`${family}: safetensors sha256 mismatch`)

  const labelsJson = `${JSON.stringify(familyLabels(name), null, 1)}\n`
  const labelsHash = `sha256:${createHash('sha256').update(labelsJson).digest('hex')}`
  const val = isRecord(manifestRaw.val) ? manifestRaw.val : {}
  const fvu = readNumber(val, 'fvu') ?? Number.POSITIVE_INFINITY
  const arch = archKey(manifestRaw.arch)

  const live = await fetchLive(family)
  if (live && !FORCE && live.sha256 === sha && live.labelsHash === labelsHash) {
    console.log(`${family}: weights and labels unchanged (live v${live.version}); skipping`)
    return
  }
  if (live && !FORCE && live.arch === arch && fvu > live.fvu * (1 + REGRESS_TOL)) {
    console.log(
      `${family}: fvu ${fvu.toFixed(4)} regresses vs live ${live.fvu.toFixed(4)}; not promoting`,
    )
    return
  }

  const version = (live?.version ?? 0) + 1
  const base = `models/${family}/v${version}`
  writeFileSync(join(vdir, 'labels.json'), labelsJson)
  for (const src of SOURCES) copyFileSync(src, join(vdir, src.split('/').pop() ?? src))
  const stamped = { ...manifestRaw, version, archiveVersion: entry.version, labelsHash }
  writeFileSync(join(vdir, 'manifest.json'), JSON.stringify(stamped, null, 2))

  r2put(`${base}/model.safetensors`, join(vdir, 'model.safetensors'), IMMUTABLE_CC)
  r2put(`${base}/manifest.json`, join(vdir, 'manifest.json'), IMMUTABLE_CC)
  r2put(`${base}/README.md`, join(vdir, 'README.md'), IMMUTABLE_CC)
  r2put(`${base}/labels.json`, join(vdir, 'labels.json'), IMMUTABLE_CC)
  for (const src of SOURCES) {
    const file = src.split('/').pop() ?? src
    r2put(`${base}/${file}`, join(vdir, file), IMMUTABLE_CC)
  }
  const latest = {
    version,
    manifest: `${base}/manifest.json`,
    weights: `${base}/model.safetensors`,
    labels: `${base}/labels.json`,
    card: `${base}/README.md`,
    trainer: `${base}/features.py`,
    labeller: `${base}/label.py`,
    sha256: sha,
    labelsHash,
    datasetHash: readString(manifestRaw, 'datasetHash') ?? '',
    arch: manifestRaw.arch,
    val: { fvu, levels: Array.isArray(val.levels) ? val.levels : [] },
    promotedAt: Date.now(),
  }
  const latestPath = join(vdir, 'latest.json')
  writeFileSync(latestPath, JSON.stringify(latest, null, 2))
  r2put(`models/${family}/latest.json`, latestPath, POINTER_CC)
  console.log(`${family}: published v${version} (archive v${entry.version}, fvu ${fvu.toFixed(4)})`)
}

async function main(): Promise<void> {
  for (const name of DICTS) await publishDict(name)
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : err)
    process.exit(1)
  })
