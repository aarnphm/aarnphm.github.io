import { execFileSync } from 'node:child_process'
import { isRecord, readNumber, readString } from '../util/type-guards'

export const BUCKET = 'sites'
export const PUBLIC_BASE = process.env.PUBLIC_BASE_URL ?? 'https://aarnphm.xyz'
export const DRY = process.env.PACE_PUBLISH_DRY === '1'
const R2_SCOPE = process.env.PACE_R2_LOCAL === '1' ? '--local' : '--remote'
export const IMMUTABLE_CC = 'public, max-age=31536000, immutable'
export const POINTER_CC = 'public, max-age=60, must-revalidate'

export interface LiveLatest {
  version: number
  datasetHash: string
  valMae: number
}

export async function fetchLiveLatest(family: string): Promise<LiveLatest | null> {
  try {
    const res = await fetch(`${PUBLIC_BASE}/models/${family}/latest.json`)
    if (!res.ok) return null
    const json: unknown = await res.json()
    if (!isRecord(json)) return null
    const val = isRecord(json.val) ? json.val : {}
    return {
      version: readNumber(json, 'version') ?? 0,
      datasetHash: readString(json, 'datasetHash') ?? '',
      valMae: readNumber(val, 'mae') ?? Number.POSITIVE_INFINITY,
    }
  } catch {
    return null
  }
}

export function r2put(key: string, file: string, cacheControl: string): void {
  const args = [
    'dlx',
    'wrangler',
    'r2',
    'object',
    'put',
    `${BUCKET}/${key}`,
    '--file',
    file,
    '--cache-control',
    cacheControl,
    R2_SCOPE,
  ]
  if (DRY) {
    console.log(`  [dry] pnpm ${args.join(' ')}`)
    return
  }
  execFileSync('pnpm', args, { stdio: 'inherit' })
}
