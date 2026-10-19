// Runs both notes through Quartz's Codeblock transformer (the real jsx-fence entry point) and
// reports the FSMDecoding figures it renders, so replay.mjs can check the embeds parse as written.
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { repo } from '../figure-harness.mjs'

const entry = `
import { readFileSync } from 'node:fs'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import { unified } from 'unified'
import { VFile } from 'vfile'
import { Codeblock } from './quartz/plugins/transformers/codeblock'
export function run(note) {
  const plugin = Codeblock({})
  const p = unified().use(remarkParse)
  for (const m of plugin.markdownPlugins({})) p.use(m)
  p.use(remarkRehype, { allowDangerousHtml: true })
  for (const h of plugin.htmlPlugins({})) p.use(h)
  p.use(rehypeStringify, { allowDangerousHtml: true })
  return String(p.processSync(new VFile({ value: readFileSync(note, 'utf8'), path: note })))
}
`

export async function renderNotes(notes) {
  const bundle = await esbuild.build({
    stdin: { contents: entry, resolveDir: repo, loader: 'ts' },
    write: false,
    bundle: true,
    platform: 'node',
    format: 'esm',
    jsx: 'automatic',
    jsxImportSource: 'preact',
    packages: 'external',
    loader: { '.py': 'text', '.inline.ts': 'text' },
    plugins: [sassPlugin({ type: 'css-text', cssImports: true })],
    logLevel: 'error',
  })
  const tmp = path.join(repo, `.fsm-pipeline-${process.pid}.mjs`)
  await writeFile(tmp, bundle.outputFiles[0].text)
  try {
    const { run } = await import(pathToFileURL(tmp).href)
    return notes.map(n => run(path.join(repo, n)))
  } finally {
    await rm(tmp, { force: true })
  }
}
