import { writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { isRecord, readNumber, readString } from '../util/type-guards'
import { POINTER_CC, PUBLIC_BASE, r2put } from './pace-r2'

async function main(): Promise<void> {
  const [family, versionArg] = process.argv.slice(2)
  if ((family !== 'pace' && family !== 'hr') || !versionArg) {
    console.error('usage: revert-pace-model <pace|hr> <version>')
    process.exit(1)
  }
  const version = Number(versionArg)
  const manifestUrl = `${PUBLIC_BASE}/models/${family}/v${version}/manifest.json`
  const res = await fetch(manifestUrl)
  if (!res.ok) {
    console.error(`${family} v${version} not in R2 (${manifestUrl} -> ${res.status})`)
    process.exit(1)
  }
  const manifest: unknown = await res.json()
  if (!isRecord(manifest)) {
    console.error('manifest malformed')
    process.exit(1)
  }
  const val = isRecord(manifest.val) ? manifest.val : {}
  const latest = {
    version,
    manifest: `models/${family}/v${version}/manifest.json`,
    weights: `models/${family}/v${version}/model.safetensors`,
    sha256: readString(manifest, 'sha256') ?? '',
    datasetHash: readString(manifest, 'datasetHash') ?? '',
    val: {
      mae: readNumber(val, 'mae') ?? 0,
      nll: readNumber(val, 'nll') ?? 0,
      coverage90: readNumber(val, 'coverage90') ?? 0,
    },
    promotedAt: Date.now(),
    revertedTo: version,
  }
  const tmp = join(tmpdir(), `pace-revert-${family}-v${version}.json`)
  writeFileSync(tmp, JSON.stringify(latest, null, 2))
  r2put(`models/${family}/latest.json`, tmp, POINTER_CC)
  console.log(`${family}: latest.json repointed to v${version}`)
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : err)
    process.exit(1)
  })
