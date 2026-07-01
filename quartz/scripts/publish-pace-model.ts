import { createHash } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PaceModel, ensureBackend, parseManifest, parseSafetensors } from '../util/pace-model'
import { isRecord, readNumber, readString } from '../util/type-guards'
import {
  IMMUTABLE_CC,
  LEGACY_VAL_SPACE,
  POINTER_CC,
  fetchLiveLatest,
  livePointerVal,
  r2put,
  regressesLiveMae,
} from './pace-r2'

const FAMILIES = ['pace', 'hr'] as const
const ARCHIVE = process.env.PACE_ARCHIVE ?? '.models-archive'
const GOLDEN_TOL = 1e-4
const REGRESS_TOL = 0.15

function latestArchiveEntry(dir: string): { path: string; version: number } | null {
  if (!existsSync(dir)) return null
  let best: { path: string; version: number } | null = null
  for (const name of readdirSync(dir)) {
    const m = /-v(\d+)$/.exec(name)
    if (!m) continue
    const version = Number(m[1])
    if (!best || version > best.version) best = { path: join(dir, name), version }
  }
  return best
}

async function publishFamily(family: string): Promise<void> {
  const entry = latestArchiveEntry(join(ARCHIVE, family))
  if (!entry) {
    console.log(`${family}: nothing archived`)
    return
  }
  const vdir = entry.path
  const manifestRaw: unknown = JSON.parse(readFileSync(join(vdir, 'manifest.json'), 'utf8'))
  const manifest = parseManifest(manifestRaw)
  const stBuf = readFileSync(join(vdir, 'model.safetensors'))

  const sha = createHash('sha256').update(stBuf).digest('hex')
  if (sha !== manifest.sha256) throw new Error(`${family}: safetensors sha256 mismatch`)

  const val = isRecord(manifestRaw) && isRecord(manifestRaw.val) ? manifestRaw.val : {}
  if (val.beatsBaseline !== true) {
    console.log(`${family}: ${entry.path} does not beat baseline; not promoting`)
    return
  }

  await ensureBackend(false)
  const ab = stBuf.buffer.slice(stBuf.byteOffset, stBuf.byteOffset + stBuf.byteLength)
  const model = PaceModel.load(manifest, parseSafetensors(ab))
  const gate = await model.checkGolden(GOLDEN_TOL)
  if (!gate.ok) throw new Error(`${family}: golden parity ${gate.maxErr} > ${GOLDEN_TOL}`)

  const datasetHash = readString(manifestRaw as Record<string, unknown>, 'datasetHash') ?? ''
  const newMae = readNumber(val, 'mae') ?? Number.POSITIVE_INFINITY
  const newValSpace = readString(val, 'valSpace') ?? LEGACY_VAL_SPACE
  const live = await fetchLiveLatest(family)
  if (live && live.datasetHash === datasetHash) {
    console.log(`${family}: dataset unchanged (live v${live.version}); skipping`)
    return
  }
  if (live && regressesLiveMae(live, newMae, newValSpace, REGRESS_TOL)) {
    console.log(
      `${family}: val MAE ${newMae.toFixed(4)} regresses vs ${live.valMae.toFixed(4)}; not promoting`,
    )
    return
  }

  const version = (live?.version ?? 0) + 1
  const stamped = { ...(isRecord(manifestRaw) ? manifestRaw : {}), version }
  const manifestPath = join(vdir, 'manifest.json')
  writeFileSync(manifestPath, JSON.stringify(stamped, null, 2))

  r2put(
    `models/${family}/v${version}/model.safetensors`,
    join(vdir, 'model.safetensors'),
    IMMUTABLE_CC,
  )
  r2put(`models/${family}/v${version}/manifest.json`, manifestPath, IMMUTABLE_CC)

  const latest = {
    version,
    manifest: `models/${family}/v${version}/manifest.json`,
    weights: `models/${family}/v${version}/model.safetensors`,
    sha256: manifest.sha256,
    datasetHash,
    val: livePointerVal(val, newMae),
    promotedAt: Date.now(),
  }
  const latestPath = join(vdir, 'latest.json')
  writeFileSync(latestPath, JSON.stringify(latest, null, 2))
  r2put(`models/${family}/latest.json`, latestPath, POINTER_CC)
  console.log(
    `${family}: published v${version} (mae ${newMae.toFixed(4)}, golden ${gate.maxErr.toExponential(1)})`,
  )
}

async function main(): Promise<void> {
  for (const family of FAMILIES) await publishFamily(family)
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : err)
    process.exit(1)
  })
