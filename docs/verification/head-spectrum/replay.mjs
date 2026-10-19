// Replay: node docs/verification/head-spectrum/replay.mjs
// Renders HeadSpectrum twice (the note's embed and a second instance), drives every control by
// pointer and keyboard, and checks the drawing and readouts against the exported JSON.
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  checker,
  openFigurePage,
  renderFigurePage,
  repo,
  standardChecks,
} from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(here, 'artifacts')
const data = JSON.parse(
  await readFile(path.join(repo, 'quartz/components/mdx/qwen3-head-spectra.json'), 'utf8'),
)
const note = await readFile(path.join(repo, 'content/lectures/412/notes.md'), 'utf8')
const caption = note.match(/<HeadSpectrum caption="([^"]*)"/)?.[1]

// Independent copies of the plot geometry, so a scale bug in the component cannot hide itself.
const LAYOUT = {
  wide: { x0: 58, x1: 548, sTop: 22, sBot: 146, eTop: 192, eBot: 300 },
  narrow: { x0: 40, x1: 292, sTop: 22, sBot: 126, eTop: 172, eBot: 272 },
}
const xE = (g, k) => g.x0 + ((k - 1) / 127) * (g.x1 - g.x0)
const yE = (g, e) => g.eBot - e * (g.eBot - g.eTop)
const xS = (g, i) => g.x0 + ((i - 1) / 1023) * (g.x1 - g.x0)
const yS = (g, s) => g.sBot - ((Math.log10(Math.max(s, 1e-10)) + 10) / 11) * (g.sBot - g.sTop)
const fmtSigma = s => (s >= 0.01 ? s.toFixed(4) : s.toExponential(2))
const near = (a, b, tol = 0.15) => Math.abs(Number(a) - Number(b)) <= tol
const headOf = id => data.heads.find(h => `${h.layer}-${h.head}` === id)

const instances = [
  { caption },
  { caption: 'second instance', layer: 3, head: 3, prompt: 'plain', k: 64 },
]
const pagePath = await renderFigurePage({
  component: 'HeadSpectrum',
  instances,
  out,
  label: 'W_OV spectrum and attention per head',
})
const { browser, page, errors } = await openFigurePage(pagePath)
const c = checker(out)
c.check('note caption found', Boolean(caption), caption?.slice(0, 60))

const roots = page.locator('[data-head-spectrum]')
const root0 = roots.nth(0)
const root1 = roots.nth(1)

const snapshot = root =>
  root.evaluate(el => {
    const vis = e => e && getComputedStyle(e).display !== 'none' && e.getClientRects().length > 0
    const svg = [...el.querySelectorAll('[data-hs-layout]')].find(vis)
    const attr = (sel, a) => svg.querySelector(sel)?.getAttribute(a)
    const prompt = el.getAttribute('data-hs-prompt')
    const panel = el.querySelector(`[data-hs-prompt-panel="${prompt}"]`)
    // The per-head rows use display: contents, so they have no box of their own.
    const activeRows = [...el.querySelectorAll('.hs-head-rows')].filter(
      r => getComputedStyle(r).display !== 'none',
    )
    return {
      head: el.getAttribute('data-hs-head'),
      prompt,
      layout: svg.getAttribute('data-hs-layout'),
      k: Number(el.querySelector('[data-hs-k-input]').value),
      outK: el.querySelector('[data-hs-out="k"]').textContent,
      outSigma: el.querySelector('[data-hs-out="sigmaK"]').textContent,
      outEnergy: el.querySelector('[data-hs-out="energyK"]').textContent,
      reff: activeRows.map(r => r.querySelector('[data-hs-reff]').textContent),
      sigma1: activeRows.map(r => r.querySelectorAll('dd')[0].textContent),
      dotE: [attr('[data-hs-dot-e]', 'cx'), attr('[data-hs-dot-e]', 'cy')],
      dotS: [attr('[data-hs-dot-s]', 'cx'), attr('[data-hs-dot-s]', 'cy')],
      markerE: attr('[data-hs-marker-e]', 'x1'),
      band: attr('[data-hs-kept-band]', 'width'),
      areaPoints: attr('[data-hs-kept-area]', 'points').trim().split(/\s+/).length,
      activeCurves: [...svg.querySelectorAll('.hs-curve[data-active="true"]')].map(p =>
        p.getAttribute('data-hs-for'),
      ),
      reffX: [...svg.querySelectorAll('.hs-reff')]
        .filter(vis)
        .map(g => g.querySelector('line').getAttribute('x1')),
      visiblePanels: [...el.querySelectorAll('[data-hs-prompt-panel]')]
        .filter(vis)
        .map(p => p.getAttribute('data-hs-prompt-panel')),
      cells: [...panel.querySelectorAll('[data-hs-cell]')].map(r => [
        r.getAttribute('data-hs-cell'),
        r.style.getPropertyValue('--hs-a'),
      ]),
      masked: panel.querySelectorAll('.hs-cell--masked').length,
      cellReadout: panel.querySelector('[data-hs-cell-readout]').textContent,
    }
  })

const verify = async (root, label) => {
  const s = await snapshot(root)
  const h = headOf(s.head)
  const g = LAYOUT[s.layout]
  const pi = s.prompt === 'plain' ? 0 : 1
  const n = data.prompts[pi].tokens.length
  const e = h.energy[s.k - 1] ?? 1
  const problems = []
  if (s.outK !== String(s.k)) problems.push(['k readout', s.outK, s.k])
  if (s.outSigma !== fmtSigma(h.sigma[s.k - 1]))
    problems.push(['sigma_k', s.outSigma, h.sigma[s.k - 1]])
  // The figure computes E_k from the 5-digit sigma; the export computed it in float64 from float32.
  if (!near(s.outEnergy, e, 2e-4)) problems.push(['E_k', s.outEnergy, e])
  if (s.reff.length !== 1 || !near(s.reff[0], h.rEff, 0.006))
    problems.push(['r_eff', s.reff, h.rEff])
  if (s.sigma1[0] !== fmtSigma(h.sigma[0])) problems.push(['sigma_1', s.sigma1, h.sigma[0]])
  if (!near(s.dotE[0], xE(g, s.k)) || !near(s.dotE[1], yE(g, e), 0.2))
    problems.push(['E dot', s.dotE, [xE(g, s.k), yE(g, e)]])
  if (!near(s.dotS[0], xS(g, s.k)) || !near(s.dotS[1], yS(g, h.sigma[s.k - 1]), 0.2))
    problems.push(['sigma dot', s.dotS, [xS(g, s.k), yS(g, h.sigma[s.k - 1])]])
  if (!near(s.markerE, xE(g, s.k))) problems.push(['E marker', s.markerE])
  if (!near(s.band, xS(g, s.k) - xS(g, 1), 0.2)) problems.push(['kept band', s.band])
  if (s.areaPoints !== s.k + 2) problems.push(['kept area points', s.areaPoints, s.k + 2])
  if (s.activeCurves.length !== 2 || s.activeCurves.some(id => id !== s.head))
    problems.push(['active curves', s.activeCurves])
  if (s.reffX.length !== 1 || !near(s.reffX[0], xE(g, h.rEff)))
    problems.push(['r_eff rule', s.reffX])
  if (s.visiblePanels.join() !== s.prompt) problems.push(['prompt panel', s.visiblePanels])
  if (s.cells.length !== (n * (n + 1)) / 2 || s.masked !== (n * (n - 1)) / 2)
    problems.push(['cell counts', s.cells.length, s.masked])
  for (const [rc, a] of s.cells) {
    const [r, col] = rc.split('-').map(Number)
    if (!near(a, h.attention[pi][r][col], 5e-5))
      problems.push(['cell', rc, a, h.attention[pi][r][col]])
  }
  c.check(
    `${label}: drawing and readouts match data (head ${s.head}, ${s.prompt}, k ${s.k}, ${s.layout})`,
    problems.length === 0,
    problems.slice(0, 5),
  )
  return s
}

// Initial SSR state of both instances.
await verify(root0, 'initial instance 0')
const s1 = await verify(root1, 'initial instance 1')
c.check(
  'instance 1 starts from its props',
  s1.head === '3-3' && s1.prompt === 'plain' && s1.k === 64,
  s1,
)

// Head segments by click.
for (const h of data.heads) {
  await root0.locator('.hs-segment', { hasText: new RegExp(`^L${h.layer} H${h.head}$`) }).click()
  await verify(root0, `click head L${h.layer} H${h.head}`)
}
// Head segments by keyboard: arrows move a native radio group.
await root0.locator('.hs-segment', { hasText: /^L3 H3$/ }).click()
await root0.locator('[data-hs-head-input]:checked').focus()
await page.keyboard.press('ArrowRight')
let s = await verify(root0, 'keyboard ArrowRight on heads')
c.check('ArrowRight selects the next head', s.head === '3-6', s.head)
await page.keyboard.press('ArrowLeft')
s = await snapshot(root0)
c.check('ArrowLeft returns to the previous head', s.head === '3-3', s.head)
await c.shoot(root0.locator('.hs-segments--heads'), 'focus-head-segment.png')

// Prompt segments by click and keyboard.
await root0.locator('.hs-segment', { hasText: /^no comma$/ }).click()
s = await verify(root0, 'click prompt no comma')
c.check('no-comma prompt shows 10 tokens', s.cells.length === 55, s.cells.length)
await root0.locator('[data-hs-prompt-input]:checked').focus()
await page.keyboard.press('ArrowRight')
s = await verify(root0, 'keyboard prompt comma')
c.check('comma prompt shows 11 tokens', s.prompt === 'comma' && s.cells.length === 66, s.prompt)

// k slider by keyboard.
await root0.locator('.hs-segment', { hasText: /^L12 H8$/ }).click()
const slider0 = root0.locator('[data-hs-k-input]')
await slider0.focus()
for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowRight')
s = await verify(root0, 'slider ArrowRight x5')
c.check('ArrowRight x5 moves k from 32 to 37', s.k === 37, s.k)
await page.keyboard.press('Home')
s = await verify(root0, 'slider Home')
c.check('Home sets k = 1 and E_1 = 0.1120', s.k === 1 && s.outEnergy === '0.1120', s)
await page.keyboard.press('End')
s = await verify(root0, 'slider End')
c.check('End sets k = 128 and E_128 = 1.0000', s.k === 128 && s.outEnergy === '1.0000', s)
const valuetext = await slider0.getAttribute('aria-valuetext')
c.check('slider aria-valuetext carries E_k', valuetext === 'k 128, E_k 1.0000', valuetext)
await c.shoot(root0.locator('.hs-plot'), 'focus-slider.png')

// k slider by pointer drag, and the thumb centre lines up with the drawn marker.
const thumbAligned = async (root, label) => {
  const geo = await root.evaluate(el => {
    const input = el.querySelector('[data-hs-k-input]')
    const svg = [...el.querySelectorAll('[data-hs-layout]')].find(
      e => getComputedStyle(e).display !== 'none',
    )
    const dot = svg.querySelector('[data-hs-dot-e]').getBoundingClientRect()
    const box = input.getBoundingClientRect()
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize)
    const k = Number(input.value)
    return {
      k,
      thumb: box.left + rem / 2 + ((k - 1) / 127) * (box.width - rem),
      dot: dot.left + dot.width / 2,
    }
  })
  c.check(
    `${label}: slider thumb centre sits on the k marker`,
    Math.abs(geo.thumb - geo.dot) < 1.5,
    geo,
  )
}
const box = await slider0.boundingBox()
await page.mouse.move(box.x + box.width - 8, box.y + box.height / 2)
await page.mouse.down()
const target = 96
const rem = 16
const tx = box.x + rem / 2 + ((target - 1) / 127) * (box.width - rem)
await page.mouse.move(tx, box.y + box.height / 2, { steps: 8 })
await page.mouse.up()
s = await verify(root0, 'slider drag')
c.check('drag moves k to about 96', Math.abs(s.k - target) <= 1, s.k)
await thumbAligned(root0, 'wide')

// Head change keeps k.
await root0.locator('.hs-segment', { hasText: /^L22 H4$/ }).click()
s = await verify(root0, 'head change after drag')
c.check('head change keeps k', Math.abs(s.k - target) <= 1, s.k)

// Heatmap: hover and keyboard read the cell weight.
await root0.locator('.hs-segment', { hasText: /^L12 H8$/ }).click()
const panel = root0.locator('[data-hs-prompt-panel="comma"]')
const tokens = data.prompts[1].tokens.map(t => t.trim() || '␣')
const att = headOf('12-8').attention[1]
await panel.locator('[data-hs-cell="4-3"]').hover()
s = await snapshot(root0)
c.check(
  'hover reads target life, source of',
  s.cellReadout === `${tokens[4]} → ${tokens[3]}: ${att[4][3].toFixed(4)}`,
  s.cellReadout,
)
await panel.locator('[data-hs-cells]').focus()
await page.keyboard.press('ArrowDown')
await page.keyboard.press('ArrowLeft')
s = await snapshot(root0)
c.check(
  'arrow keys move to row 5, column 2',
  s.cellReadout === `${tokens[5]} → ${tokens[2]}: ${att[5][2].toFixed(4)}`,
  s.cellReadout,
)
const markOnCell = await panel.evaluate(el => {
  const m = el.querySelector('[data-hs-active]').getBoundingClientRect()
  const cell = el.querySelector('[data-hs-cell="5-2"]').getBoundingClientRect()
  const frame = getComputedStyle(el.querySelector('.hs-frame')).display
  return {
    frame,
    dx: Math.abs(m.left + m.width / 2 - (cell.left + cell.width / 2)),
    dy: Math.abs(m.top + m.height / 2 - (cell.top + cell.height / 2)),
  }
})
c.check(
  'salmon mark sits on the active cell',
  markOnCell.dx < 0.5 && markOnCell.dy < 0.5,
  markOnCell,
)
c.check('keyboard focus shows the dark grid frame', markOnCell.frame !== 'none', markOnCell)
await page.keyboard.press('End')
s = await snapshot(root0)
c.check(
  'End moves to the diagonal',
  s.cellReadout === `${tokens[5]} → ${tokens[5]}: ${att[5][5].toFixed(4)}`,
  s.cellReadout,
)
await page.keyboard.press('ArrowRight')
s = await snapshot(root0)
c.check(
  'ArrowRight stops at the causal mask',
  s.cellReadout.startsWith(`${tokens[5]} → ${tokens[5]}:`),
  s.cellReadout,
)
await c.shoot(panel, 'focus-heat.png')

// Ramp: a heavier weight draws darker ink in light theme and brighter in dark theme.
const lum = async sel =>
  panel.locator(sel).evaluate(el => {
    const [r, g, b] = getComputedStyle(el)
      .fill.match(/[\d.]+/g)
      .map(Number)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  })
const heavy = '[data-hs-cell="0-0"]'
const light = '[data-hs-cell="10-9"]'
await page.waitForTimeout(400)
const lightTheme = [await lum(heavy), await lum(light)]
await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'dark'))
// The fill eases over 160ms; read it after the transition ends.
await page.waitForTimeout(400)
const darkTheme = [await lum(heavy), await lum(light)]
await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'light'))
c.check('ramp: heavy cell darker on paper', lightTheme[0] < lightTheme[1], lightTheme)
c.check('ramp: heavy cell brighter in dark theme', darkTheme[0] > darkTheme[1], darkTheme)

// Instance 1 is untouched by everything above.
const s1b = await snapshot(root1)
c.check('instance 1 unchanged', s1b.head === '3-3' && s1b.prompt === 'plain' && s1b.k === 64, s1b)
const names = await page.evaluate(() => [
  ...new Set(
    [...document.querySelectorAll('[data-head-spectrum] input[type=radio]')].map(i => i.name),
  ),
])
c.check('radio names are instance-unique', names.length === 4, names)

// Reset instance 0 to the note's default for the screenshots.
await root0.locator('.hs-segment', { hasText: /^L12 H8$/ }).click()
await root0.locator('.hs-segment', { hasText: /^comma$/ }).click()
await slider0.fill('32')
await slider0.dispatchEvent('input')

// Narrow layout: the phone-width plot is drawn and the slider still lines up.
await page.setViewportSize({ width: 320, height: 1400 })
s = await verify(root0, 'narrow')
c.check('narrow layout active at 320px', s.layout === 'narrow', s.layout)
await thumbAligned(root0, 'narrow')
await slider0.focus()
for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowLeft')
await verify(root0, 'narrow after keys')
await thumbAligned(root0, 'narrow after keys')
await slider0.fill('32')
await slider0.dispatchEvent('input')
await page.setViewportSize({ width: 900, height: 1400 })

// Reduced motion stops transitions.
await page.emulateMedia({ reducedMotion: 'reduce' })
const tr = await panel.locator(heavy).evaluate(el => getComputedStyle(el).transitionDuration)
c.check('reduced motion removes cell transitions', tr === '0s', tr)
await page.emulateMedia({ reducedMotion: 'no-preference' })

await page.evaluate(() => document.activeElement?.blur())
await standardChecks(page, errors, c, '.zoomable')
await browser.close()
process.exit((await c.finish()) ? 0 : 1)
