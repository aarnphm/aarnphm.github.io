import { createHash } from 'node:crypto'
import { createReadStream, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { isRecord, readNumber, readString } from '../util/type-guards'
import { latestModelArchiveEntry } from './pace-model-archive'
import { IMMUTABLE_CC, POINTER_CC, PUBLIC_BASE, r2put } from './pace-r2'

// The Gemma 4 E4B transcoders (docs/latent-circuits.md, docs/latent-interp-contract.md),
// published like the passage SAEs: an export script archives a trained set under
// .models-archive/<family>/<ts>-vN/ (bf16 safetensors, manifest, card, labels, scripts) and this
// promotes the newest to R2 as models/<family>/v<N>/ with a latest.json.
//
//   latent-plt (per-layer, L<ll>.safetensors):
//     uv run quartz/scripts/latent/plt_export.py --plt quartz/.quartz-cache/latent-plt/<dir>
//   latent-clt (cross-layer, enc/ dec/ skip/ L<ll>.safetensors):
//     uv run quartz/scripts/latent/clt_export.py --clt <clt.py output dir>
//
//   PACE_PUBLISH_DRY=1 pnpm tsx --env-file=.env quartz/scripts/publish-latent-plt.ts [--family latent-clt]
//   pnpm tsx --env-file=.env quartz/scripts/publish-latent-plt.ts [--family latent-clt] [--force]

const FAMILIES = {
  'latent-plt': {
    sources: ['transcoders.py', 'attribute.py', 'plt_label.py', 'acts.py'],
    links: { trainer: 'transcoders.py', attribution: 'attribute.py', labeller: 'plt_label.py' },
  },
  'latent-clt': {
    sources: ['clt.py', 'gemma_torch.py'],
    links: { trainer: 'clt.py', engine: 'gemma_torch.py' },
  },
} as const
type Family = keyof typeof FAMILIES
const EXPORTS: Record<Family, string> = {
  'latent-plt': 'plt_export.py',
  'latent-clt': 'clt_export.py',
}

function familyArg(argv: string[]): Family {
  const i = argv.indexOf('--family')
  const name = i >= 0 ? argv[i + 1] : 'latent-plt'
  if (!(name in FAMILIES))
    throw new Error(`unknown --family ${name}: ${Object.keys(FAMILIES).join(', ')}`)
  return name as Family
}

const FAMILY = familyArg(process.argv)
const ARCHIVE = process.env.PACE_ARCHIVE ?? '.models-archive'
const SOURCES: readonly string[] = FAMILIES[FAMILY].sources
const LINKS: Readonly<Record<string, string>> = FAMILIES[FAMILY].links
const FORCE = process.argv.includes('--force')

async function sha256(file: string): Promise<string> {
  const h = createHash('sha256')
  for await (const chunk of createReadStream(file)) h.update(chunk as Buffer)
  return h.digest('hex')
}

async function fetchLive(): Promise<{ version: number; manifestHash: string } | null> {
  try {
    const res = await fetch(`${PUBLIC_BASE}/models/${FAMILY}/latest.json`)
    if (!res.ok) return null
    const json: unknown = await res.json()
    if (!isRecord(json)) return null
    return {
      version: readNumber(json, 'version') ?? 0,
      manifestHash: readString(json, 'manifestHash') ?? '',
    }
  } catch {
    return null
  }
}

async function main(): Promise<void> {
  const entry = latestModelArchiveEntry(join(ARCHIVE, FAMILY))
  if (!entry) {
    console.log(`${FAMILY}: nothing archived (run ${EXPORTS[FAMILY]} first)`)
    return
  }
  const vdir = entry.path
  const raw: unknown = JSON.parse(readFileSync(join(vdir, 'manifest.json'), 'utf8'))
  if (!isRecord(raw) || !isRecord(raw.files)) throw new Error(`${FAMILY}: bad manifest`)
  const files = Object.entries(raw.files)
  for (const [name, want] of files) {
    const got = await sha256(join(vdir, name))
    if (got !== want) throw new Error(`${FAMILY}: ${name} sha256 mismatch`)
  }
  const labelsHash = readString(raw, 'labelsHash') ?? ''
  const manifestHash = createHash('sha256')
    .update(JSON.stringify({ files: raw.files, labelsHash }))
    .digest('hex')

  const live = await fetchLive()
  if (live && !FORCE && live.manifestHash === manifestHash) {
    console.log(`${FAMILY}: weights and labels unchanged (live v${live.version}); skipping`)
    return
  }
  const version = (live?.version ?? 0) + 1
  const base = `models/${FAMILY}/v${version}`
  const stamped = { ...raw, version, archiveVersion: entry.version, manifestHash }
  writeFileSync(join(vdir, 'manifest.json'), `${JSON.stringify(stamped, null, 2)}\n`)

  for (const [name] of files) r2put(`${base}/${name}`, join(vdir, name), IMMUTABLE_CC)
  for (const name of ['labels.json', 'README.md', ...SOURCES])
    r2put(`${base}/${name}`, join(vdir, name), IMMUTABLE_CC)
  r2put(`${base}/manifest.json`, join(vdir, 'manifest.json'), IMMUTABLE_CC)

  const layers = files.map(([name]) => `${base}/${name}`)
  const latest = {
    version,
    manifest: `${base}/manifest.json`,
    layers,
    labels: `${base}/labels.json`,
    card: `${base}/README.md`,
    ...Object.fromEntries(Object.entries(LINKS).map(([k, f]) => [k, `${base}/${f}`])),
    manifestHash,
    labelsHash,
    model: raw.model,
    revision: raw.revision,
    datasetHash: readString(raw, 'datasetHash') ?? '',
    engine: readString(raw, 'engine') ?? 'mlx-4bit',
    arch: raw.arch,
    val: raw.val,
    promotedAt: Date.now(),
  }
  const latestPath = join(vdir, 'latest.json')
  writeFileSync(latestPath, JSON.stringify(latest, null, 2))
  r2put(`models/${FAMILY}/latest.json`, latestPath, POINTER_CC)
  console.log(
    `${FAMILY}: published v${version} (archive v${entry.version}, ${files.length} layers)`,
  )
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : err)
    process.exit(1)
  })
