import { globby } from 'globby'
import path from 'path'
import { FilePath } from './path'

export function toPosixPath(fp: string): string {
  return fp.split(path.sep).join('/')
}

export async function glob(
  pattern: string,
  cwd: string,
  ignorePatterns: string[],
  gitignore = true,
): Promise<FilePath[]> {
  const fps = (await globby(pattern, { cwd, ignore: ignorePatterns, gitignore })).map(toPosixPath)
  return fps as FilePath[]
}
