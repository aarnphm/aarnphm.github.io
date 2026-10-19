import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
// Renders KVBudgetAllocation through Quartz's esbuild, Sass and inline-script loaders, then drives
// the page in Chromium. Writes screenshots and results.json to ./artifacts.
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '../../..')
const out = path.join(here, 'artifacts')
const executablePath = process.env.CHROMIUM ?? '/opt/pw-browsers/chromium'
await mkdir(out, { recursive: true })

const inlineLoader = {
  name: 'inline-script-loader',
  setup(build) {
    build.onLoad({ filter: /\.inline\.(ts|js)$/ }, async args => {
      const res = await esbuild.build({
        entryPoints: [args.path],
        write: false,
        bundle: true,
        platform: 'browser',
        format: 'iife',
      })
      return { contents: res.outputFiles[0].text, loader: 'text' }
    })
  },
}

const entry = `
import { h } from 'preact'
import { render } from 'preact-render-to-string'
import { KVBudgetAllocation } from './quartz/components/mdx/KVBudgetAllocation.tsx'
export const html = [
  render(h(KVBudgetAllocation, { policy: 'pyramidkv', budget: 4, caption: 'Instance one.' })),
  render(h(KVBudgetAllocation, { policy: 'adakv', budget: 4, caption: 'Instance two.' })),
].join('\\n')
export const css = KVBudgetAllocation.css
export const script = KVBudgetAllocation.afterDOMLoaded
`
const bundle = await esbuild.build({
  stdin: { contents: entry, resolveDir: repo, loader: 'tsx' },
  write: false,
  bundle: true,
  platform: 'node',
  format: 'esm',
  jsx: 'automatic',
  jsxImportSource: 'preact',
  packages: 'external',
  plugins: [sassPlugin({ type: 'css-text', cssImports: true }), inlineLoader],
})
// The bundle keeps npm imports bare, so it must resolve them from the repository.
const tmp = path.join(repo, '.kvb-render.mjs')
await writeFile(tmp, bundle.outputFiles[0].text)
const { html, css, script } = await import(tmp)
await rm(tmp)

const katexCss = await readFile(path.join(repo, 'node_modules/katex/dist/katex.min.css'), 'utf8')
const theme = (light, lightgray, gray, darkgray, dark) =>
  `--light:${light};--lightgray:${lightgray};--gray:${gray};--darkgray:${darkgray};--dark:${dark};--secondary:rgb(205,213,151);--tertiary:rgb(252,193,146)`
const page = `<!doctype html><html saved-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>${katexCss.replace(/url\(fonts\//g, 'url(https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/')}</style>
<style>
:root{${theme('rgb(255,252,240)', 'rgb(230,228,217)', 'rgb(183,181,172)', 'rgb(111,110,105)', 'rgb(16,15,15)')};--radius-none:0;--codeFont:'Berkeley Mono',ui-monospace,monospace}
:root[saved-theme='dark']{${theme('rgb(16,15,15)', 'rgb(40,39,38)', 'rgb(87,86,83)', 'rgb(135,133,128)', 'rgb(206,205,195)')}}
body{margin:0;padding:16px;background:var(--light);color:var(--dark);font-family:Cambria,Georgia,serif;max-width:46rem}
${css}
</style></head><body>${html}
<script>window.addCleanup=fn=>(window.__cleanups=window.__cleanups||[]).push(fn)</script>
<script>${script}</script>
<script>document.dispatchEvent(new Event('nav'))</script>
</body></html>`
const pagePath = path.join(out, 'page.html')
await writeFile(pagePath, page)

const browser = await chromium.launch({ executablePath })
const results = { checks: [], screenshots: [] }
const check = (name, ok, detail) => results.checks.push({ name, ok, detail })
const ctx = await browser.newContext({ viewport: { width: 900, height: 1400 } })
const pg = await ctx.newPage()
await pg.goto(`file://${pagePath}`)

const snapshot = idx =>
  pg.evaluate(i => {
    const root = document.querySelectorAll('[data-kv-budget-allocation]')[i]
    const cells = [...root.querySelectorAll('[data-kvb-cell]')]
    const kept = cells.filter(c => c.getAttribute('data-kvb-kept') === 'true').length
    const rows = [...root.querySelectorAll('[data-kvb-row]')].map(t => Number(t.textContent))
    const out = Object.fromEntries(
      [...root.querySelectorAll('[data-kvb-out]')].map(d => [d.dataset.kvbOut, d.textContent]),
    )
    const rule = [...root.querySelectorAll('[data-kvb-rule]')].find(
      p => getComputedStyle(p).display !== 'none',
    )?.dataset.kvbRule
    return { kept, rows, out, rule, policy: root.dataset.kvbPolicy }
  }, idx)

const ssr = await snapshot(0)
check('SSR PyramidKV k=4 per-layer budgets', ssr.out.layers === '7 · 5 · 3 · 1', ssr.out.layers)
check('SSR drawn kept cells equal readout', `${ssr.kept} / 192` === ssr.out.kept, ssr)
check(
  'Row counts follow layer budgets',
  JSON.stringify(ssr.rows) === JSON.stringify([7, 7, 7, 5, 5, 5, 3, 3, 3, 1, 1, 1]),
  ssr.rows,
)
await pg
  .locator('[data-kv-budget-allocation]')
  .nth(0)
  .screenshot({ path: path.join(out, 'light-pyramidkv-k4.png') })
results.screenshots.push('light-pyramidkv-k4.png')

const states = {}
for (const policy of ['full', 'streaming', 'snapkv', 'pyramidkv', 'adakv']) {
  for (const k of ['2', '4', '6']) {
    const root = pg.locator('[data-kv-budget-allocation]').nth(0)
    await root.locator(`input[value="${policy}"]`).check({ force: true })
    await root.locator(`input[data-kvb-budget-input][value="${k}"]`).check({ force: true })
    const s = await snapshot(0)
    states[`${policy}:${k}`] = s
    check(`${policy}:${k} drawn cells match readout`, `${s.kept} / 192` === s.out.kept, s.out.kept)
    check(`${policy}:${k} rule shown`, s.rule === policy, s.rule)
  }
}
results.states = Object.fromEntries(Object.entries(states).map(([k, s]) => [k, s.out]))
check('Full keeps everything', states['full:4'].out.kept === '192 / 192', states['full:4'].out.kept)
for (const k of ['2', '4', '6']) {
  const budget = 12 * Number(k)
  for (const p of ['streaming', 'snapkv', 'pyramidkv', 'adakv'])
    check(
      `${p}:${k} spends exactly 12k`,
      states[`${p}:${k}`].kept === budget,
      states[`${p}:${k}`].kept,
    )
}
const second = await snapshot(1)
check(
  'Second instance is independent',
  second.policy === 'adakv' && second.rule === 'adakv',
  second.policy,
)

// Keyboard: arrow keys move within the native radio group.
const root0 = pg.locator('[data-kv-budget-allocation]').nth(0)
await root0.locator('input[value="snapkv"]').check({ force: true })
await root0.locator('input[value="snapkv"]').focus()
await pg.keyboard.press('ArrowRight')
const afterKey = await snapshot(0)
check('ArrowRight selects next policy', afterKey.policy === 'pyramidkv', afterKey.policy)
const focusOutline = await pg.evaluate(() => {
  const label = document.activeElement?.closest('label')
  return label ? getComputedStyle(label).outlineStyle : null
})
check('Focused segment shows an outline', focusOutline === 'solid', focusOutline)
await root0.screenshot({ path: path.join(out, 'light-focus-keyboard.png') })
results.screenshots.push('light-focus-keyboard.png')

await root0.locator('input[value="adakv"]').check({ force: true })
await root0.locator('input[data-kvb-budget-input][value="4"]').check({ force: true })
await root0.screenshot({ path: path.join(out, 'light-adakv-k4.png') })
results.screenshots.push('light-adakv-k4.png')

await pg.evaluate(() => document.documentElement.setAttribute('saved-theme', 'dark'))
await root0.screenshot({ path: path.join(out, 'dark-adakv-k4.png') })
results.screenshots.push('dark-adakv-k4.png')

await pg.setViewportSize({ width: 320, height: 1400 })
await pg.evaluate(() => document.documentElement.setAttribute('saved-theme', 'light'))
const overflow = await pg.evaluate(() => document.documentElement.scrollWidth - innerWidth)
check('No horizontal page scroll at 320px', overflow <= 0, overflow)
await root0.screenshot({ path: path.join(out, 'light-320px.png') })
results.screenshots.push('light-320px.png')

await pg.evaluate(() => window.__cleanups.forEach(fn => fn()))
await root0.locator('input[value="full"]').check({ force: true })
const afterCleanup = await snapshot(0)
check('Cleanup removes the change listener', afterCleanup.policy === 'adakv', afterCleanup.policy)

await browser.close()
results.passed = results.checks.every(c => c.ok)
await writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2))
console.log(`${results.checks.filter(c => c.ok).length}/${results.checks.length} checks passed`)
for (const c of results.checks.filter(c => !c.ok))
  console.log('FAIL', c.name, JSON.stringify(c.detail))
process.exit(results.passed ? 0 : 1)
