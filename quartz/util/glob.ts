import path from "path"
import { FilePath } from "./path"
import { globby } from "globby"

export function toPosixPath(fp: string): string {
  return fp.split(path.sep).join("/")
}

export async function glob(
  pattern: string,
  cwd: string,
  ignorePatterns: string[],
): Promise<FilePath[]> {
  const fps = (
    await globby(pattern, {
      cwd,
      ignore: ignorePatterns,
      gitignore: true,
    })
  ).map(toPosixPath)
  return fps as FilePath[]
}

export function random(min: number, max: number) {
  if (max == null) {
    max = min
    min = 0
  }
  return min + Math.floor(Math.random() * (max - min + 1))
}

// Fisher-Yates shuffle
export function sample(object: any[], n: number) {
  const sample = [...object]
  var length = sample.length
  n = Math.max(Math.min(n, length), 0)
  var last = length - 1
  for (var index = 0; index < n; index++) {
    var rand = random(index, last)
    var temp = object[index]
    sample[index] = sample[rand]
    sample[rand] = temp
  }
  return sample.slice(0, n)
}
