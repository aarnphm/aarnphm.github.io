import { existsSync, readdirSync } from 'node:fs'
import path from 'node:path'

export interface ModelArchiveEntry {
  name: string
  path: string
  version: number
}

export function latestModelArchiveEntry(directory: string): ModelArchiveEntry | null {
  if (!existsSync(directory)) return null
  let best: ModelArchiveEntry | null = null
  for (const name of readdirSync(directory)) {
    const match = /-v(\d+)$/.exec(name)
    if (!match) continue
    const version = Number(match[1])
    if (!best || version > best.version) best = { name, path: path.join(directory, name), version }
  }
  return best
}
