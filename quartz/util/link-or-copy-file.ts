import fs from 'node:fs/promises'
import { dirname } from 'node:path'
import type { FilePath } from './path'

const fallbackLinkErrorCodes = new Set(['EXDEV', 'EPERM', 'EACCES', 'ENOTSUP', 'EMLINK', 'EEXIST'])
const ensuredDirs = new Set<string>()

type LinkOrCopyOptions = { hardLink?: boolean }

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

export async function linkOrCopyFile(
  src: FilePath,
  dest: FilePath,
  options: LinkOrCopyOptions = {},
): Promise<FilePath> {
  const dir = dirname(dest)
  if (!ensuredDirs.has(dir)) {
    await fs.mkdir(dir, { recursive: true })
    ensuredDirs.add(dir)
  }
  if (!options.hardLink) {
    await fs.copyFile(src, dest)
    return dest
  }

  if (await isAlreadyLinked(src, dest)) return dest

  try {
    await fs.link(src, dest)
  } catch (error) {
    if (!isFallbackLinkError(error)) {
      throw error
    }
    if (errorCode(error) === 'EEXIST') {
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
    await fs.copyFile(src, dest)
  }
  return dest
}
