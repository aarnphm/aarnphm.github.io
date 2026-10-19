// Renders the DataChart fences exactly as written in the four notes, through Quartz's Codeblock
// transformer (MDX parse, prop parsing, htmlToJsx), then drives them in Chromium.
// Run: node docs/verification/charts-tables/replay.mjs   (writes ./artifacts)
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { checker, openFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '../../..')
const out = path.join(here, 'artifacts')
await mkdir(out, { recursive: true })

const NOTES = [
  'content/posts/structured decoding.md',
  'content/thoughts/sparse crosscoders.md',
  'content/thoughts/economies of scale.md',
]
const fences = []
for (const note of NOTES) {
  const src = await readFile(path.join(repo, note), 'utf8')
  for (const m of src.matchAll(/```jsx imports=\{[^}]*DataChart[^}]*\}\n[\s\S]*?\n```/g))
    fences.push({ note, md: m[0] })
}

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
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import { Codeblock } from './quartz/plugins/transformers/codeblock.ts'
import { DataChart, Zoomable } from './quartz/components/mdx/index.ts'
const plugin = Codeblock()
export const render = md =>
  String(
    unified()
      .use(remarkParse)
      .use(plugin.markdownPlugins())
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(plugin.htmlPlugins())
      .use(rehypeStringify, { allowDangerousHtml: true })
      .processSync(md),
  )
export const css = [Zoomable.css, DataChart.css].join('\\n')
export const scripts = [Zoomable.afterDOMLoaded, DataChart.afterDOMLoaded]
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
  plugins: [sassPlugin({ type: 'css-text', cssImports: true }), inlineLoader],
})
const tmp = path.join(repo, `.dch-render-${process.pid}.mjs`)
await writeFile(tmp, bundle.outputFiles[0].text)
let mod
try {
  mod = await import(pathToFileURL(tmp).href)
} finally {
  await rm(tmp, { force: true })
}
const html = fences.map(f => mod.render(f.md)).join('\n')
await writeFile(
  path.join(out, 'fences.md'),
  fences.map(f => `<!-- ${f.note} -->\n${f.md}`).join('\n\n'),
)

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
${mod.css}
</style></head><body>${html}
<script>window.addCleanup=fn=>(window.__cleanups=window.__cleanups||[]).push(fn)</script>
${mod.scripts.map(s => `<script>${s}</script>`).join('\n')}
<script>document.dispatchEvent(new Event('nav'))</script>
</body></html>`
const pagePath = path.join(out, 'page.html')
await writeFile(pagePath, page)

// Values as printed on the source figures (vLLM) and as read from the Anthropic figure.
const EXPECT = [
  {
    kind: 'bar',
    rows: ['1', '5', '10', '15', '20', '25'],
    cols: [
      ['75', '68', '80', '118', '220', '446'],
      ['10507', '139455', '144131', '204710', '192325', '244884'],
    ],
  },
  {
    kind: 'bar',
    rows: ['1', '5', '10', '15', '20', '25'],
    cols: [
      ['8.4', '9.6', '12.1', '16.1', '26.5', '32.0'],
      ['7.9', '13.4', '19.6', '73.8', '68.6', '152.3'],
    ],
  },
  { kind: 'line', positions: 11, points: [10, 11] },
  { kind: 'line', positions: 23, points: [10, 13] },
  { kind: 'probe' },
]

const { browser, page: pg, errors } = await openFigurePage(pagePath)
const c = checker(out)
const charts = pg.locator('.data-chart')
c.check('every note fence rendered a chart', (await charts.count()) === EXPECT.length, {
  fences: fences.length,
  charts: await charts.count(),
})

// Geometry read back from the SVG: fit the y scale from gridlines and tick labels, then compare
// every drawn mark with the value in the chart's own table.
const geometry = i =>
  charts.nth(i).evaluate(root => {
    const grid = [...root.querySelectorAll('.dch-grid line')].filter(
      l => l.getAttribute('y1') === l.getAttribute('y2'),
    )
    const ticks = [...root.querySelectorAll('.dch-tick')][0]
    const labels = ticks ? [...ticks.querySelectorAll('text')].map(t => t.textContent) : []
    const parse = s => {
      const m = /^([\d.]+)([kMG]?)$/.exec(s)
      return m ? Number(m[1]) * { '': 1, k: 1e3, M: 1e6, G: 1e9 }[m[2]] : NaN
    }
    const tickYs = grid.map(l => Number(l.getAttribute('y1')))
    const tickVs = labels.map(parse)
    const rows = [...root.querySelectorAll('.dch-table tbody tr')].map(tr =>
      [...tr.children].map(td => td.textContent.trim()),
    )
    const bars = [...root.querySelectorAll('.dch-marks .dch-bar')].map(r => ({
      cls: r.getAttribute('class'),
      top: Number(r.getAttribute('y')),
      bottom: Number(r.getAttribute('y')) + Number(r.getAttribute('height')),
      x: Number(r.getAttribute('x')),
      w: Number(r.getAttribute('width')),
    }))
    const markers = [...root.querySelectorAll('.dch-marks .dch-marker')].map(m => ({
      cls: m.getAttribute('class'),
      y:
        m.tagName === 'circle'
          ? Number(m.getAttribute('cy'))
          : Number(m.getAttribute('y')) + Number(m.getAttribute('height')) / 2,
    }))
    const log = root.querySelector('.dch-axis')?.textContent.includes('log')
    return { tickYs, tickVs, rows, bars, markers, log }
  })

const fitScale = g => {
  const f = g.log ? Math.log10 : v => v
  const [a, b] = [0, g.tickVs.length - 1]
  const k = (g.tickYs[b] - g.tickYs[a]) / (f(g.tickVs[b]) - f(g.tickVs[a]))
  return v => g.tickYs[a] + k * (f(v) - f(g.tickVs[a]))
}

const readout = i =>
  charts
    .nth(i)
    .locator('.dch-readout:not([hidden])')
    .evaluate(dl => [...dl.querySelectorAll('dd')].map(d => d.textContent.trim()))

const activeIndex = i =>
  charts
    .nth(i)
    .evaluate(root =>
      [...root.querySelectorAll('[data-dch-pos]')].findIndex(
        o => o.getAttribute('aria-selected') === 'true',
      ),
    )

for (let i = 0; i < EXPECT.length; i++) {
  const e = EXPECT[i]
  const root = charts.nth(i)
  if (e.kind === 'probe') continue
  const g = await geometry(i)
  const y = fitScale(g)
  if (e.kind === 'bar') {
    c.check(
      `chart ${i}: table equals the printed values`,
      JSON.stringify(g.rows) ===
        JSON.stringify(e.rows.map((r, j) => [r, e.cols[0][j], e.cols[1][j]])),
      g.rows,
    )
    const s0 = g.bars.filter(b => b.cls.includes('dch-s0'))
    const s1 = g.bars.filter(b => b.cls.includes('dch-s1'))
    const err = [
      ...s0.map((b, j) => b.top - y(Number(e.cols[0][j]))),
      ...s1.map((b, j) => b.top - y(Number(e.cols[1][j]))),
    ]
    c.check(
      `chart ${i}: every bar top sits on its value (scale fitted from tick labels)`,
      err.length === 12 && err.every(d => Math.abs(d) < 0.6),
      err.map(d => d.toFixed(2)),
    )
    c.check(`chart ${i}: bars share one baseline`, new Set(g.bars.map(b => b.bottom)).size === 1, [
      ...new Set(g.bars.map(b => b.bottom)),
    ])
    c.check(
      `chart ${i}: bars within a group keep a 2-unit gap, at most 24 thick`,
      s0.every((b, j) => Math.abs(s1[j].x - (b.x + b.w) - 2) < 0.15) &&
        g.bars.every(b => b.w <= 24),
      s0.map((b, j) => (s1[j].x - b.x - b.w).toFixed(2)),
    )
  } else {
    const n0 = g.markers.filter(m => m.cls.includes('dch-s0')).length
    const n1 = g.markers.filter(m => m.cls.includes('dch-s1')).length
    c.check(`chart ${i}: marker counts per series`, n0 === e.points[0] && n1 === e.points[1], [
      n0,
      n1,
    ])
    c.check(
      `chart ${i}: one table row per x position`,
      g.rows.length === e.positions,
      g.rows.length,
    )
    const values = g.rows
      .flatMap(r => r.slice(1))
      .filter(v => v !== '–')
      .map(Number)
    const drawn = g.markers.map(m => m.y).sort((a, b) => a - b)
    const expected = values.map(y).sort((a, b) => a - b)
    c.check(
      `chart ${i}: every marker sits on a table value`,
      drawn.length === expected.length && drawn.every((d, j) => Math.abs(d - expected[j]) < 0.6),
      drawn.map((d, j) => (d - expected[j]).toFixed(2)),
    )
  }

  // Pointer: hover each position and compare the readout with that table row.
  const opts = root.locator('[data-dch-pos]')
  const n = await opts.count()
  let hoverOk = true
  const hoverFail = []
  for (let j = 0; j < n; j++) {
    const box = await opts.nth(j).boundingBox()
    const cx = await opts.nth(j).getAttribute('data-dch-cx')
    const svgBox = await root.locator('[data-dch-svg]').boundingBox()
    const px = svgBox.x + (Number(cx) / 440) * svgBox.width
    await pg.mouse.move(px, box.y + box.height / 2)
    const r = await readout(i)
    const row = g.rows[j]
    const want = row.slice(1).map(v => (v === '–' ? '–' : v))
    const got = r.slice(1, 1 + want.length).map(v => v.replace(/ ms$/, ''))
    const ok = (await activeIndex(i)) === j && JSON.stringify(got) === JSON.stringify(want)
    if (!ok) {
      hoverOk = false
      hoverFail.push({ j, got, want })
    }
  }
  c.check(`chart ${i}: hovering each of ${n} positions shows its table row`, hoverOk, hoverFail)

  // Keyboard: Tab reaches the selected option; arrows, Home and End move selection and focus.
  await opts.nth(n - 1).evaluate(o => o.closest('.data-chart').scrollIntoView())
  await root.evaluate(r => r.querySelector('[data-dch-pos][tabindex="0"]').focus())
  await pg.keyboard.press('Home')
  const a0 = await activeIndex(i)
  await pg.keyboard.press('ArrowRight')
  const a1 = await activeIndex(i)
  await pg.keyboard.press('End')
  const aEnd = await activeIndex(i)
  await pg.keyboard.press('ArrowRight')
  const aClamp = await activeIndex(i)
  await pg.keyboard.press('ArrowLeft')
  const aLeft = await activeIndex(i)
  const focusOk = await root.evaluate(
    r =>
      document.activeElement?.getAttribute('aria-selected') === 'true' &&
      r.contains(document.activeElement),
  )
  const tabStops = await root.evaluate(
    r =>
      [...r.querySelectorAll('[data-dch-pos]')].filter(o => o.getAttribute('tabindex') === '0')
        .length,
  )
  c.check(
    `chart ${i}: Home, ArrowRight, End, clamp, ArrowLeft`,
    a0 === 0 &&
      a1 === 1 &&
      aEnd === n - 1 &&
      aClamp === n - 1 &&
      aLeft === n - 2 &&
      focusOk &&
      tabStops === 1,
    { a0, a1, aEnd, aClamp, aLeft, focusOk, tabStops },
  )
  const ring = await root.evaluate(() => {
    const s = getComputedStyle(document.activeElement)
    return { stroke: s.stroke, width: s.strokeWidth }
  })
  c.check(
    `chart ${i}: focused position shows a 2px ink stroke`,
    ring.width === '2px' && ring.stroke !== 'none',
    ring,
  )
  const band = await root.evaluate(r =>
    [...r.querySelectorAll('[data-dch-band]')].map(b => b.hasAttribute('data-dch-on')),
  )
  c.check(
    `chart ${i}: exactly the selected band is lit`,
    band.filter(Boolean).length === 1 && band[n - 2],
    band.indexOf(true),
  )
  await pg.mouse.move(0, 0)
}

// Probe: the slider, the drag and the readout agree with LRAC(Q) = 36/Q + 2 + Q/4.
{
  const i = 4
  const root = charts.nth(i)
  const lrac = q => 36 / q + 2 + q / 4
  const xPx = q => 46 + (q / 40) * (428 - 46)
  const yPx = v => 228 - (v / 16) * (228 - 30)
  const region = q =>
    Math.abs(q - 12) <= 0.25
      ? 'minimum efficient scale'
      : q < 12
        ? 'economies of scale'
        : 'diseconomies of scale'
  const state = () =>
    root.evaluate(r => {
      const dot = r.querySelector('[data-dch-dot]')
      const dl = r.querySelector('.dch-readout:not([hidden])')
      return {
        cx: Number(dot.getAttribute('cx')),
        cy: Number(dot.getAttribute('cy')),
        d: r.querySelector('[data-dch-drop]').getAttribute('d'),
        dd: [...dl.querySelectorAll('dd')].map(d => d.textContent.trim()),
        value: r.querySelector('[data-dch-probe-input]').value,
        valuetext: r.querySelector('[data-dch-probe-input]').getAttribute('aria-valuetext'),
      }
    })
  const agrees = (s, q) =>
    Math.abs(s.cx - xPx(q)) < 0.06 &&
    Math.abs(s.cy - yPx(lrac(q))) < 0.06 &&
    s.dd[0] === q.toFixed(1) &&
    s.dd[1] === lrac(q).toFixed(2) &&
    s.dd[2] === region(q) &&
    s.d === `M46 ${s.cy}H${s.cx}V228` &&
    s.valuetext === `${q.toFixed(1)}, ${lrac(q).toFixed(2)}, ${region(q)}`
  const s0 = await state()
  c.check('probe: SSR state at Q = 8', agrees(s0, 8), s0)
  const slider = root.locator('[data-dch-probe-input]')
  await slider.focus()
  const seen = []
  let ok = true
  for (let step = 0; step < 12; step++) {
    await pg.keyboard.press('ArrowRight')
    const q = 8 + 0.5 * (step + 1)
    const s = await state()
    seen.push([q, s.dd])
    if (!agrees(s, q)) ok = false
  }
  c.check('probe: ArrowRight walks Q from 8.5 to 14 in 0.5 steps, through the minimum', ok, seen)
  await pg.keyboard.press('Home')
  const sHome = await state()
  await pg.keyboard.press('End')
  const sEnd = await state()
  c.check('probe: Home and End reach Q = 3 and Q = 38', agrees(sHome, 3) && agrees(sEnd, 38), [
    sHome.dd,
    sEnd.dd,
  ])
  const focus = await slider.evaluate(
    el => getComputedStyle(el).outlineStyle + ' ' + getComputedStyle(el).outlineWidth,
  )
  c.check('probe: slider shows a visible focus outline', focus.startsWith('solid 2px'), focus)
  await c.shoot(root, 'probe-end.png')

  // Drag: press at Q = 20 and move to Q = 30 on the plot.
  const svgBox = await root.locator('[data-dch-svg]').boundingBox()
  const toPage = (q, v) => [
    svgBox.x + (xPx(q) / 440) * svgBox.width,
    svgBox.y + (yPx(v) / 270) * svgBox.height,
  ]
  const [ax, ay] = toPage(20, 6)
  const [bx] = toPage(30, 6)
  await pg.mouse.move(ax, ay)
  await pg.mouse.down()
  const sDown = await state()
  await pg.mouse.move(bx, ay, { steps: 6 })
  const sMove = await state()
  await pg.mouse.up()
  c.check(
    'probe: pointer drag moves the point along the curve',
    agrees(sDown, 20) && agrees(sMove, 30),
    [sDown.dd, sMove.dd],
  )
  const marks = await root.evaluate(r =>
    [...r.querySelectorAll('.dch-ref rect')].map(m => [
      Number(m.getAttribute('x')) + 3.75,
      Number(m.getAttribute('y')) + 3.75,
    ]),
  )
  c.check(
    'probe: sage marks sit on the curve at Q = 4 and Q = 12',
    marks.length === 2 &&
      Math.abs(marks[0][0] - xPx(4)) < 0.06 &&
      Math.abs(marks[0][1] - yPx(lrac(4))) < 0.06 &&
      Math.abs(marks[1][1] - yPx(8)) < 0.06,
    marks,
  )
}

// SPA navigation: run cleanups, fire nav again, and check one key press still moves one step.
await pg.evaluate(() => {
  for (const fn of window.__cleanups ?? []) fn()
  window.__cleanups = []
  document.dispatchEvent(new Event('nav'))
})
await charts.nth(0).evaluate(r => r.querySelector('[data-dch-pos][tabindex="0"]').focus())
await pg.keyboard.press('Home')
await pg.keyboard.press('ArrowRight')
c.check(
  'after cleanup and a second nav, one ArrowRight moves one position',
  (await activeIndex(0)) === 1,
  await activeIndex(0),
)
const slider = charts.nth(4).locator('[data-dch-probe-input]')
await slider.focus()
await pg.keyboard.press('Home')
await pg.keyboard.press('ArrowRight')
const afterNav = await charts
  .nth(4)
  .evaluate(r => r.querySelector('.dch-readout:not([hidden]) dd').textContent)
c.check('after a second nav the probe still updates once per key', afterNav === '3.5', afterNav)

// Table fallback opens with the native disclosure.
await charts.nth(0).locator('summary').click()
c.check(
  'table disclosure opens',
  await charts
    .nth(0)
    .locator('details')
    .evaluate(d => d.open),
  null,
)
await c.shoot(charts.nth(0), 'table-open.png')
await charts.nth(0).locator('summary').click()

await standardChecks(pg, errors, c, '.zoomable')
await browser.close()
process.exit((await c.finish()) ? 0 : 1)
