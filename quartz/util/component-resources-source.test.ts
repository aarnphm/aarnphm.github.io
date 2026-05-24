import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

test('keeps postscript as a script orchestrator during watch builds', async () => {
  const source = await readFile(
    new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
    'utf8',
  )
  const start = source.indexOf('async function writeAfterDomLoadedScripts')
  const end = source.indexOf('async function writeNotebookRuntimeAssets')
  const implementation = source.slice(start, end)

  assert.match(implementation, /contentHashSlug\('static\/scripts\/script', content\)/)
  assert.match(
    implementation,
    /entries\.map\(\(\{ filename \}\) => `await import\("\.\/\$\{filename\}"\);`\)\.join\('\\n'\)/,
  )
  assert.doesNotMatch(implementation, /postscript:\s*await joinScripts\(scripts\)/)
  assert.doesNotMatch(implementation, /if \(!shouldHashAssets\(ctx\)\)/)
  assert.doesNotMatch(implementation, /Promise\.all\(\[/)
})
