import { open, readdir, stat } from 'node:fs/promises'
import path from 'node:path'

const imageExtensions = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.gif',
  '.avif',
  '.heic',
  '.svg',
  '.ico',
])
const maxImageBytes = 25 * 1024 * 1024
const roots = process.argv.slice(2)
if (roots.length === 0) roots.push('content/thoughts/images', 'content/hinterland/app')

async function* files(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name)
    if (entry.isDirectory()) yield* files(filename)
    else if (entry.isFile()) yield filename
  }
}

let images = 0
let largest = null
const failures = []
try {
  for (const root of roots) {
    for await (const filename of files(root)) {
      const extension = path.extname(filename).toLowerCase()
      const image = imageExtensions.has(extension)
      if (!image && extension !== '.md') continue
      const info = await stat(filename)
      if (image) {
        images++
        if (!largest || info.size > largest.bytes) largest = { file: filename, bytes: info.size }
        if (info.size >= maxImageBytes)
          failures.push(`${filename}: ${info.size} bytes must be below 25 MiB`)
      }
      const handle = await open(filename, 'r')
      try {
        const prefix = Buffer.alloc(200)
        const { bytesRead } = await handle.read(prefix, 0, prefix.length, 0)
        if (
          prefix
            .subarray(0, bytesRead)
            .toString()
            .startsWith('version https://git-lfs.github.com/spec/v1\n')
        ) {
          failures.push(`${filename}: unresolved Git LFS pointer`)
        }
      } finally {
        await handle.close()
      }
    }
  }
} catch (error) {
  failures.push(error instanceof Error ? error.message : String(error))
}

if (failures.length > 0) {
  process.stderr.write(failures.join('\n') + '\n')
  process.exitCode = 1
} else {
  process.stdout.write(
    JSON.stringify({ status: 'passed', roots, images, largest, maxImageBytes }) + '\n',
  )
}
