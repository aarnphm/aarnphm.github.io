import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { dirname } from 'node:path'
import type { StravaRawCache } from '../plugins/stores/strava'
import { isRecord } from './type-guards'

function isMissingFile(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT'
}

function parseStravaCache(source: string, path: string): StravaRawCache {
  try {
    const cache: StravaRawCache = JSON.parse(source)
    return cache
  } catch (error) {
    throw new Error(`Strava cache at ${path} is invalid`, { cause: error })
  }
}

export async function readStravaCacheFile(path: string): Promise<StravaRawCache | null> {
  let source: string
  try {
    source = await fsp.readFile(path, 'utf8')
  } catch (error) {
    if (isMissingFile(error)) return null
    throw error
  }
  return parseStravaCache(source, path)
}

export function readStravaCacheFileSync(path: string): StravaRawCache | null {
  let source: string
  try {
    source = fs.readFileSync(path, 'utf8')
  } catch (error) {
    if (isMissingFile(error)) return null
    throw error
  }
  return parseStravaCache(source, path)
}

export async function writeStravaCacheFile(path: string, cache: StravaRawCache): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`
  await fsp.mkdir(dirname(path), { recursive: true })
  try {
    await fsp.writeFile(temporary, JSON.stringify(cache, null, 2))
    await fsp.rename(temporary, path)
  } finally {
    await fsp.rm(temporary, { force: true })
  }
}
