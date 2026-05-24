import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

test('keeps postscript as a script orchestrator during watch builds', async () => {
  const source = await readFile(
    new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
    'utf8',
  )

  assert.match(source, /contentHashSlug\('static\/scripts\/script', content\)/)
  assert.doesNotMatch(source, /postscript:\s*await joinScripts\(scripts\)/)
  assert.doesNotMatch(source, /if \(!shouldHashAssets\(ctx\)\)/)
})
