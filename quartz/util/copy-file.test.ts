import assert from 'node:assert/strict'
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { FilePath } from './path'
import { copyFile, resetCopyFileCache } from './copy-file'

test('resetCopyFileCache forgets removed output directories', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'quartz-copy-file-'))
  try {
    const src = path.join(root, 'source.txt') as FilePath
    const output = path.join(root, 'public')
    const dest = path.join(output, 'nested', 'source.txt') as FilePath
    await writeFile(src, 'first')

    await copyFile(src, dest)
    await stat(dest)

    await rm(output, { recursive: true, force: true })
    resetCopyFileCache()
    await copyFile(src, dest)

    assert.equal((await stat(dest)).isFile(), true)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
