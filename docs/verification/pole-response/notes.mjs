// Runs each note's PoleResponse fences through Quartz's Codeblock transformer (the real embed path)
// and checks the rendered figure carries the props the fence asks for.
// Writes artifacts/notes.json. Usage: node docs/verification/pole-response/notes.mjs
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '../../..')
const art = path.join(here, 'artifacts')
await mkdir(art, { recursive: true })

const inlineLoader = {
  name: 'inline-script-loader',
  setup(build) {
    build.onLoad({ filter: /\.inline\.(ts|js)$/ }, () => ({ contents: '', loader: 'text' }))
  },
}
const entry = `
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import { unified } from 'unified'
import { Codeblock } from './quartz/plugins/transformers/codeblock.ts'
export const run = md => {
  const plugin = Codeblock()
  const proc = unified().use(remarkParse)
  for (const p of plugin.markdownPlugins()) proc.use(p)
  proc.use(remarkRehype, { allowDangerousHtml: true })
  for (const p of plugin.htmlPlugins()) proc.use(p)
  proc.use(rehypeStringify, { allowDangerousHtml: true })
  return String(proc.processSync({ value: md, path: 'note.md' }))
}
`
const bundle = await esbuild.build({
  stdin: { contents: entry, resolveDir: repo, loader: 'ts' },
  write: false,
  bundle: true,
  platform: 'node',
  format: 'esm',
  jsx: 'automatic',
  jsxImportSource: 'preact',
  packages: 'external',
  loader: { '.py': 'text' },
  plugins: [sassPlugin({ type: 'css-text', cssImports: true }), inlineLoader],
  logLevel: 'error',
})
const tmp = path.join(repo, `.prs-notes-${process.pid}.mjs`)
await writeFile(tmp, bundle.outputFiles[0].text)
let run
try {
  ;({ run } = await import(pathToFileURL(tmp).href))
} finally {
  await rm(tmp, { force: true })
}

const NOTES = {
  'thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response.md': [
    { mode: 'first', state: '{"kind":"first","p":-1}' },
    { mode: 'second', lock: 'sigma', state: '"re":0,"im":3.14159' },
    { mode: 'second', radios: 'data-prs-case-input', regions: true },
    { mode: 'second', state: '"re":-1,"im":2.828' },
    { mode: 'second', radios: 'data-prs-lock-input', state: '"re":-1,"im":3' },
  ],
  'thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response.md': [
    { mode: 'first', state: '{"kind":"first","p":-1}' },
    { mode: 'second', radios: 'data-prs-case-input' },
  ],
  'thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Root locus control.md': [
    { mode: 'locus', comp: 'none' },
    { mode: 'locus', comp: 'integrator', radios: 'data-prs-comp-input' },
    { mode: 'locus', comp: 'pi', radios: 'data-prs-comp-input' },
  ],
  'thoughts/Z-transform.md': [{ mode: 'second', regions: true, state: '"re":-2,"im":2' }],
}

const results = { checks: [] }
const check = (name, ok, detail) => results.checks.push({ name, ok: Boolean(ok), detail })
const unescape = s => s.replace(/&quot;|&#x22;/g, '"').replace(/&#x26;|&amp;/g, '&')

for (const [note, want] of Object.entries(NOTES)) {
  const md = await readFile(path.join(repo, 'content', note), 'utf8')
  const fences = [...md.matchAll(/```jsx imports=\{Zoomable,PoleResponse\}\n[\s\S]*?\n```/g)].map(
    m => m[0],
  )
  check(`${note}: ${want.length} PoleResponse fences`, fences.length === want.length, fences.length)
  check(
    `${note}: no replaced image left`,
    !/!\[\[[^\]]*(time-response-freq-domain|undamped-natural-freq|sec-order-impulse-response|peak-graph-time-response|relations-to-poles|same-envelope|same-frequency|same-overshoot|time constant|second-order-system|transient-response-root-locus|ideal-integral-compensator|zero-add-compensator|s-plane-stability)\.webp/.test(
      md,
    ),
  )
  fences.forEach((fence, i) => {
    const html = unescape(run(fence))
    const w = want[i] ?? {}
    const label = `${note} #${i}`
    check(`${label}: figure rendered`, html.includes('data-pole-response'), html.slice(0, 200))
    check(`${label}: mode ${w.mode}`, html.includes(`data-prs-mode="${w.mode}"`))
    if (w.lock) check(`${label}: lock ${w.lock}`, html.includes(`data-prs-lock="${w.lock}"`))
    if (w.comp) check(`${label}: comp ${w.comp}`, html.includes(`data-prs-comp="${w.comp}"`))
    if (w.state) check(`${label}: initial state`, html.includes(w.state), w.state)
    if (w.radios) check(`${label}: ${w.radios} controls`, html.includes(w.radios))
    check(
      `${label}: regions ${Boolean(w.regions)}`,
      html.includes('prs-regions') === Boolean(w.regions),
    )
    check(`${label}: caption math rendered`, /<figcaption[^>]*>[\s\S]*class="katex"/.test(html))
    check(`${label}: no raw dollar math in caption`, !/<figcaption[^>]*>[^<]*\$/.test(html))
  })
}

results.passed = results.checks.every(c => c.ok)
await writeFile(path.join(art, 'notes.json'), JSON.stringify(results, null, 2))
console.log(
  `${results.checks.filter(c => c.ok).length}/${results.checks.length} note checks passed`,
)
for (const c of results.checks.filter(c => !c.ok))
  console.log('FAIL', c.name, JSON.stringify(c.detail))
process.exit(results.passed ? 0 : 1)
