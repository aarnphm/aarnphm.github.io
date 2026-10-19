// Shared harness for Garden figures. Renders registered MDX components through Quartz's esbuild,
// Sass and inline-script loaders, optionally inside Zoomable, and opens the page in Chromium.
//
// Library: import { renderFigurePage, openFigurePage, checker } from '../figure-harness.mjs'
// CLI:     node docs/verification/figure-harness.mjs <Component> '<props json>' <out-dir>
//          writes page.html, light.png, dark.png, 320.png and results.json to <out-dir>.
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { chromium } from 'playwright-core'

const here = path.dirname(fileURLToPath(import.meta.url))
export const repo = path.resolve(here, '../..')
const executablePath = process.env.CHROMIUM ?? '/opt/pw-browsers/chromium'

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

const theme = (light, lightgray, gray, darkgray, dark) =>
  `--light:${light};--lightgray:${lightgray};--gray:${gray};--darkgray:${darkgray};--dark:${dark};--secondary:rgb(205,213,151);--tertiary:rgb(252,193,146)`

/**
 * @param {{ component: string, instances: object[], out: string, zoomable?: boolean, label?: string }} opts
 * Each instance is the component's props. Returns the path of the written page.
 */
export async function renderFigurePage({ component, instances, out, zoomable = true, label }) {
  await mkdir(out, { recursive: true })
  const entry = `
import { h } from 'preact'
import { render } from 'preact-render-to-string'
import * as mdx from './quartz/components/mdx/index.ts'
const C = mdx[${JSON.stringify(component)}]
const Z = mdx.Zoomable
const instances = ${JSON.stringify(instances)}
export const html = instances
  .map(p => render(${zoomable} ? h(Z, { label: ${JSON.stringify(label ?? component)} }, h(C, p)) : h(C, p)))
  .join('\\n')
const parts = ${zoomable} ? [Z, C] : [C]
export const css = parts.map(p => p.css ?? '').join('\\n')
export const scripts = parts.map(p => p.afterDOMLoaded ?? '')
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
    loader: { '.py': 'text' },
    plugins: [
      sassPlugin({ type: 'css-text', cssImports: true }),
      sassPlugin({ filter: /\.inline\.scss$/, type: 'css', cssImports: true }),
      inlineLoader,
    ],
  })
  // The bundle keeps npm imports bare, so it must resolve them from the repository root.
  const tmp = path.join(repo, `.figure-render-${process.pid}-${Date.now()}.mjs`)
  await writeFile(tmp, bundle.outputFiles[0].text)
  let mod
  try {
    mod = await import(pathToFileURL(tmp).href)
  } finally {
    await rm(tmp, { force: true })
  }
  const katexCss = await readFile(path.join(repo, 'node_modules/katex/dist/katex.min.css'), 'utf8')
  const page = `<!doctype html><html saved-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>${katexCss.replace(/url\(fonts\//g, 'url(https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/')}</style>
<style>
:root{${theme('rgb(255,252,240)', 'rgb(230,228,217)', 'rgb(183,181,172)', 'rgb(111,110,105)', 'rgb(16,15,15)')};--radius-none:0;--codeFont:'Berkeley Mono',ui-monospace,monospace}
:root[saved-theme='dark']{${theme('rgb(16,15,15)', 'rgb(40,39,38)', 'rgb(87,86,83)', 'rgb(135,133,128)', 'rgb(206,205,195)')}}
body{margin:0;padding:16px;background:var(--light);color:var(--dark);font-family:Cambria,Georgia,serif;max-width:46rem}
${mod.css}
</style></head><body>${mod.html}
<script>window.addCleanup=fn=>(window.__cleanups=window.__cleanups||[]).push(fn)</script>
${mod.scripts.map(s => `<script>${s}</script>`).join('\n')}
<script>document.dispatchEvent(new Event('nav'))</script>
</body></html>`
  const pagePath = path.join(out, 'page.html')
  await writeFile(pagePath, page)
  return pagePath
}

/** Opens a written page. Returns { browser, page }; close the browser when done. */
export async function openFigurePage(pagePath, viewport = { width: 900, height: 1400 }) {
  const browser = await chromium.launch({ executablePath })
  const page = await (await browser.newContext({ viewport })).newPage()
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  page.on('console', m => m.type() === 'error' && errors.push(m.text()))
  await page.goto(pathToFileURL(pagePath).href)
  return { browser, page, errors }
}

/** Collects named checks and writes results.json. */
export function checker(out) {
  const results = { checks: [], screenshots: [] }
  return {
    results,
    check(name, ok, detail) {
      results.checks.push({ name, ok: Boolean(ok), detail })
    },
    async shoot(locator, file) {
      await locator.screenshot({ path: path.join(out, file) })
      results.screenshots.push(file)
    },
    async finish() {
      results.passed = results.checks.every(c => c.ok)
      await writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2))
      console.log(
        `${results.checks.filter(c => c.ok).length}/${results.checks.length} checks passed`,
      )
      for (const c of results.checks.filter(c => !c.ok))
        console.log('FAIL', c.name, JSON.stringify(c.detail))
      return results.passed
    },
  }
}

/** Light, dark and 320px screenshots of every figure root plus page-level checks. */
export async function standardChecks(page, errors, c, rootSelector = 'figure') {
  const roots = page.locator(rootSelector)
  const n = await roots.count()
  c.check('figure rendered', n > 0, n)
  for (let i = 0; i < n; i++) await c.shoot(roots.nth(i), `light-${i}.png`)
  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'dark'))
  for (let i = 0; i < n; i++) await c.shoot(roots.nth(i), `dark-${i}.png`)
  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'light'))
  await page.setViewportSize({ width: 320, height: 1400 })
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)
  c.check('no horizontal page scroll at 320px', overflow <= 0, overflow)
  for (let i = 0; i < n; i++) await c.shoot(roots.nth(i), `320-${i}.png`)
  await page.setViewportSize({ width: 900, height: 1400 })
  c.check('no page errors', errors.length === 0, errors)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const [component, props = '{}', outArg] = process.argv.slice(2)
  if (!component || !outArg) {
    console.error("usage: node figure-harness.mjs <Component> '<props json>' <out-dir>")
    process.exit(2)
  }
  const out = path.resolve(outArg)
  const parsed = JSON.parse(props)
  const pagePath = await renderFigurePage({
    component,
    instances: Array.isArray(parsed) ? parsed : [parsed],
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  process.exit((await c.finish()) ? 0 : 1)
}
