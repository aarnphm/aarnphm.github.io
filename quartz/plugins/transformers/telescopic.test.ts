import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import test from 'node:test'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import { isFullSlug } from '../../util/path'

test('telescopic fence marks register rendered blocks through Obsidian parsing', async t => {
  const hooks = registerHooks({
    resolve(specifier, context, next) {
      return next(specifier.endsWith('.inline') ? `${specifier}.ts` : specifier, context)
    },
    load(url, context, next) {
      if (/\.(?:scss|css|inline\.ts)$/.test(url)) {
        return {
          format: 'module',
          source: `export default ${JSON.stringify(readFileSync(new URL(url), 'utf8'))}`,
          shortCircuit: true,
        }
      }
      return next(url, context)
    },
  })
  t.after(() => hooks.deregister())
  const { TelescopicText } = await import('./telescopic')
  const { ObsidianFlavoredMarkdown } = await import('./ofm')
  const ctx: BuildCtx = {
    buildId: 'telescopic-test',
    argv: {
      directory: 'content',
      output: 'public',
      verbose: false,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: false,
    },
    get cfg(): BuildCtx['cfg'] {
      throw new Error('Unexpected config access')
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
  const plugins = [TelescopicText(), ObsidianFlavoredMarkdown()]
  const processor = unified().use(remarkParse)
  for (const plugin of plugins) processor.use(plugin.markdownPlugins?.(ctx) ?? [])
  processor.use(remarkRehype, { allowDangerousHtml: true })
  for (const plugin of plugins) processor.use(plugin.htmlPlugins?.(ctx) ?? [])
  processor.use(rehypeStringify)
  const source =
    '```telescopic id="thoughts" marks="telexample"\n* small\n  * expanded text\n```\n\nOrdinary text ^ordinary\n'
  const file = new VFile(source)
  const slug = 'fixture'
  assert.ok(isFullSlug(slug))
  file.data.slug = slug
  const result = await processor.process(file)
  assert.equal(result.data.blocks?.telexample?.tagName, 'div')
  assert.deepEqual(result.data.blocks?.telexample?.properties.className, ['telescopic-container'])
  assert.equal(result.data.blocks?.ordinary?.tagName, 'p')
  assert.match(String(result), /id="telexample"/)
  assert.doesNotMatch(String(result), /\^telexample/)
  const directory = 'docs/verification/telescopic-marks/artifacts'
  await mkdir(directory, { recursive: true })
  await writeFile(`${directory}/source.md`, source)
  await writeFile(`${directory}/rendered.html`, String(result))
})
