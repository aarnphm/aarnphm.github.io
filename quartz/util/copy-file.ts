import { randomUUID } from 'node:crypto'
import { constants } from 'node:fs'
import fs from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import type { FilePath } from './path'

const ensuredDirs = new Map<string, Promise<void>>()

async function ensureDir(dir: string): Promise<void> {
  const existing = ensuredDirs.get(dir)
  if (existing) return existing
  const pending = fs
    .mkdir(dir, { recursive: true })
    .then(() => undefined)
    .catch(error => {
      ensuredDirs.delete(dir)
      throw error
    })
  ensuredDirs.set(dir, pending)
  return pending
}

export function resetCopyFileCache(): void {
  ensuredDirs.clear()
}

export async function copyFile(
  src: FilePath,
  dest: FilePath,
  outputRoot?: string,
): Promise<FilePath> {
  await ensureDir(dirname(dest))
  if (outputRoot) {
    const temporary = join(dirname(resolve(outputRoot)), `.quartz-copy-${randomUUID()}`)
    try {
      await fs.copyFile(src, temporary, constants.COPYFILE_FICLONE)
      await fs.rename(temporary, dest)
    } finally {
      await fs.rm(temporary, { force: true })
    }
  } else {
    await fs.copyFile(src, dest, constants.COPYFILE_FICLONE)
  }
  return dest
}
