import { mkdir, writeFile, readFile, rm } from 'node:fs/promises'
import path from 'node:path'
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { fileURLToPath, pathToFileURL } from 'node:url'

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const components = [
  ['FlashInferPageLayout', {}],
  ['PacketSniffer', {}],
  ['MTPModules', { depth: 2 }],
  ['ParamDecomposition', { components: 6 }],
  ['SafetyLevels', {}],
  ['VllmEngineLayout', {}],
]

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

let failed = false
for (const [component, props] of components) {
  const out = path.join(repo, 'docs/verification/image-to-react-port/artifacts', component)
  await mkdir(out, { recursive: true })
  const entry = `
import { h } from 'preact'
import { render } from 'preact-render-to-string'
import * as mdx from './quartz/components/mdx/index.ts'
const C = mdx[${JSON.stringify(component)}]
const Z = mdx.Zoomable
const instances = ${JSON.stringify([props])}
export const html = instances
  .map(p => render(h(Z, { label: ${JSON.stringify(component)} }, h(C, p))))
  .join('\\n')
export const css = [Z, C].map(p => p.css ?? '').join('\\n')
export const scripts = [Z, C].map(p => p.afterDOMLoaded ?? '')
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
  const tmp = path.join(repo, `.figure-render-${process.pid}-${component}.mjs`)
  await writeFile(tmp, bundle.outputFiles[0].text)
  try {
    const mod = await import(pathToFileURL(tmp).href + `?t=${Date.now()}`)
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
    await writeFile(path.join(out, 'page.html'), page)
    console.log('OK', component)
  } catch (e) {
    failed = true
    console.error('FAIL', component, e)
  } finally {
    await rm(tmp, { force: true })
  }
}
process.exit(failed ? 1 : 0)
