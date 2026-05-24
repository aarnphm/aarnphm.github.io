import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import test from 'node:test'

test('keeps postscript as a script orchestrator during watch builds', async () => {
  const source = await readFile(
    new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
    'utf8',
  )
  const start = source.indexOf('async function writeAfterDomLoadedScripts')
  const end = source.indexOf('async function notebookPyrightTypeshedJson')
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

test('loads extracted client assets relative to their script chunks', async () => {
  const notebookRuntimeInline = await readFile(
    new URL('../components/scripts/notebook-runtime.inline.ts', import.meta.url),
    'utf8',
  )
  const collaborativeCommentsInline = await readFile(
    new URL('../components/scripts/collaborative-comments.inline.ts', import.meta.url),
    'utf8',
  )
  const semanticInline = await readFile(
    new URL('../components/scripts/semantic.inline.ts', import.meta.url),
    'utf8',
  )

  assert.match(notebookRuntimeInline, /new URL\(name, import\.meta\.url\)/)
  assert.match(collaborativeCommentsInline, /new URL\(name, import\.meta\.url\)/)
  assert.match(semanticInline, /new URL\('semantic\.worker\.js', import\.meta\.url\)/)
  assert.doesNotMatch(notebookRuntimeInline, /static\/scripts\/\$\{name\}/)
  assert.doesNotMatch(collaborativeCommentsInline, /static\/scripts\/\$\{name\}/)
  assert.doesNotMatch(semanticInline, /\/semantic\.worker\.js/)
})

test('emits semantic worker as a split static script asset', async () => {
  const source = await readFile(
    new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
    'utf8',
  )

  assert.match(source, /const semanticWorkerEntry = 'quartz\/workers\/semantic\.worker\.ts'/)
  assert.match(source, /const semanticWorkerPath = 'static\/scripts\/semantic\.worker\.js'/)
  assert.match(source, /entryPoints: \{ 'semantic\.worker': semanticWorkerEntry \}/)
  assert.match(source, /splitting: true/)
  assert.match(source, /assetBasename\(ctx, semanticWorkerPath\)/)
  assert.match(source, /\.filter\(\s*src => src !== semanticWorkerEntry,\s*\)/)
  assert.match(source, /if \(changeEvent\.path === semanticWorkerEntry\) continue/)
})

test('keeps base and component styles in the quartz layer', async () => {
  const source = await readFile(
    new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
    'utf8',
  )
  const customStyles = await readFile(new URL('../styles/custom.scss', import.meta.url), 'utf8')

  assert.match(source, /import baseStyles from '\.\.\/\.\.\/styles\/base\.scss'/)
  assert.match(source, /import customStyles from '\.\.\/\.\.\/styles\/custom\.scss'/)
  assert.match(source, /const quartzBase = joinStyles\(/)
  assert.match(
    source,
    /const stylesheet = `@layer quartz-base \{\\n\$\{quartzBase\}\\n\}\\n\$\{customStyles\}`/,
  )
  assert.match(
    source,
    /minifyStylesheet\('component\.css', `@layer quartz-base \{\\n\$\{stylesheet\}\\n\}`\)/,
  )
  assert.doesNotMatch(customStyles, /@use ['"]\.\/base\.scss['"]/)

  const pageStyles = await readdir(new URL('../styles/pages/', import.meta.url))
  for (const pageStyle of pageStyles.filter(file => file.endsWith('.scss'))) {
    const stylesheet = await readFile(
      new URL(`../styles/pages/${pageStyle}`, import.meta.url),
      'utf8',
    )
    assert.doesNotMatch(stylesheet, /@use ['"]\.\.\/base\.scss['"]/)
  }
})
