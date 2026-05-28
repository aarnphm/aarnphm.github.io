import { constants } from 'node:fs'
import fs from 'node:fs/promises'
import { dirname, relative, resolve } from 'node:path'
import type { Argv } from './ctx'
import type { FilePath } from './path'

const fallbackLinkErrorCodes = new Set(['EXDEV', 'EPERM', 'EACCES', 'ENOTSUP', 'EMLINK', 'EEXIST'])
const ensuredDirs = new Map<string, Promise<void>>()

type LinkOrCopyOptions = { hardLink?: boolean; symlink?: boolean }

function errorCode(error: unknown): string | undefined {
  if (!(error instanceof Error) || !('code' in error)) return undefined
  const code = error.code
  return typeof code === 'string' ? code : undefined
}

function isFallbackLinkError(error: unknown): boolean {
  const code = errorCode(error)
  return code !== undefined && fallbackLinkErrorCodes.has(code)
}

function isNotFoundError(error: unknown): boolean {
  return errorCode(error) === 'ENOENT'
}

async function isAlreadyLinked(src: FilePath, dest: FilePath): Promise<boolean> {
  try {
    const [srcStat, destStat] = await Promise.all([fs.stat(src), fs.stat(dest)])
    return srcStat.dev === destStat.dev && srcStat.ino === destStat.ino
  } catch (error) {
    if (isNotFoundError(error)) return false
    throw error
  }
}

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

async function copyFile(src: FilePath, dest: FilePath): Promise<void> {
  await fs.copyFile(src, dest, constants.COPYFILE_FICLONE)
}

export async function symlinkPath(src: string, dest: string): Promise<void> {
  const target = relative(dirname(dest), resolve(src))
  try {
    await fs.symlink(target, dest)
  } catch (error) {
    if (errorCode(error) !== 'EEXIST') throw error
    await fs.rm(dest, { force: true })
    await fs.symlink(target, dest)
  }
}

export function resetLinkOrCopyFileCache(): void {
  ensuredDirs.clear()
}

export function useLocalDevLinks(argv: Pick<Argv, 'watch'>): boolean {
  return argv.watch && process.env.CF_PAGES !== '1'
}

export async function linkOrCopyFile(
  src: FilePath,
  dest: FilePath,
  options: LinkOrCopyOptions = {},
): Promise<FilePath> {
  await ensureDir(dirname(dest))
  if (options.symlink) {
    await symlinkPath(src, dest)
    return dest
  }
  if (!options.hardLink) {
    await copyFile(src, dest)
    return dest
  }

  try {
    await fs.link(src, dest)
  } catch (error) {
    if (!isFallbackLinkError(error)) {
      throw error
    }
    if (errorCode(error) === 'EEXIST') {
      if (await isAlreadyLinked(src, dest)) return dest
      await fs.rm(dest, { force: true })
      try {
        await fs.link(src, dest)
        return dest
      } catch (retryError) {
        if (!isFallbackLinkError(retryError)) {
          throw retryError
        }
      }
    }
    await copyFile(src, dest)
  }
  return dest
}
