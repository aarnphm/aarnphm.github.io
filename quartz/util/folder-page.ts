import path from 'node:path'
import { slugifyFilePath, type FilePath, type FullSlug } from './path'

const folderPageSourceExtensions = new Set(['.md', '.base', '.canvas', '.ipynb', '.pdf'])

export function isFolderPageSourcePath(fp: FilePath): boolean {
  return folderPageSourceExtensions.has(path.extname(fp).toLowerCase())
}

export function folderPageSourceSlug(fp: FilePath): FullSlug {
  return slugifyFilePath(fp, path.extname(fp).toLowerCase() === '.ipynb')
}
