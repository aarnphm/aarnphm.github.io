import assert from 'node:assert/strict'
import { mkdtemp, open, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { writeRefreshToken } from './auth-oura'

const hostile = "space \"quote\" 'single' `backtick` $dollar $$ $& $' $`"

test('Oura refresh tokens are written literally through the env owner', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'garden-oura-auth-'))
  const envFile = path.join(directory, `.env ${hostile}`)
  const refreshToken = `token ${hostile}`

  try {
    await writeFile(envFile, 'KEEP=value\nOURA_REFRESH_TOKEN=old\n')
    await writeRefreshToken(refreshToken, envFile)

    const handle = await open(envFile, 'r')
    try {
      const size = (await handle.stat()).size
      const content = Buffer.alloc(size)
      await handle.read(content, 0, size, 0)
      assert.equal(content.toString(), `KEEP=value\nOURA_REFRESH_TOKEN=${refreshToken}\n`)
    } finally {
      await handle.close()
    }
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
