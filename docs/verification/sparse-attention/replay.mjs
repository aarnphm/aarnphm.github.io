// Renders SparseAttentionMask (two instances, as DS32 embeds it) and MLAAttentionModes through
// the shared figure harness, drives every control by click and keyboard, and checks that the
// drawn state agrees with the readout tables. Writes screenshots and results.json per figure.
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const MODES = ['cmp', 'slc', 'win', 'nsa', 'dsa']
let ok = true

// Props come from the DS32 embeds, so the replay renders what the note renders.
const note = await readFile(path.resolve(here, '../../../content/thoughts/DS32.md'), 'utf8')
const embeds = name =>
  [...note.matchAll(new RegExp(`<${name}([^>]*?)/>`, 'gs'))].map(([, attrs]) => {
    const props = {}
    for (const [, k, v] of attrs.matchAll(/(\w+)="([^"]*)"/g)) props[k] = v
    for (const [, k, v] of attrs.matchAll(/(\w+)=\{(\d+)\}/g)) props[k] = Number(v)
    return props
  })

// --- SparseAttentionMask ---------------------------------------------------------------------
{
  const out = path.join(here, 'artifacts', 'sparse-attention-mask')
  const pagePath = await renderFigurePage({
    component: 'SparseAttentionMask',
    label: 'Sparse attention mask',
    out,
    instances: embeds('SparseAttentionMask'),
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  // Fills ease over 160ms; reduced motion turns that off so computed fills are final at once.
  await page.emulateMedia({ reducedMotion: 'reduce' })
  const cleanupsAtLoad = await page.evaluate(() => window.__cleanups.length)
  const roots = page.locator('[data-sparse-attention-mask]')
  c.check('two instances, as in DS32', (await roots.count()) === 2)
  c.check(
    'initial modes from props',
    (await roots.nth(0).getAttribute('data-sam-mode')) === 'dsa' &&
      (await roots.nth(1).getAttribute('data-sam-mode')) === 'nsa',
  )
  const ids = await page.$$eval('[data-sam-query-input]', els => els.map(e => e.id))
  c.check('instance-unique control ids', new Set(ids).size === ids.length, ids)

  // Reads what instance i draws for its current query and mode, and its toy readout column.
  const snapshot = i =>
    roots.nth(i).evaluate(root => {
      const svg = root.querySelector('svg')
      const probe = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      probe.style.fill = 'var(--sam-lit)'
      svg.appendChild(probe)
      const lit = getComputedStyle(probe).fill
      probe.remove()
      const t = Number(root.querySelector('[data-sam-query-input]').value)
      const row = root.querySelector(`[data-sam-row="${t}"]`)
      const keyOf = r => Math.round((Number(r.getAttribute('x')) - 40) / 6)
      const litCells = [...row.querySelectorAll('.sam-cell')]
        .filter(r => getComputedStyle(r).fill === lit)
        .map(keyOf)
      const cmpCells = [...row.querySelectorAll('.sam-cmp')].filter(
        r => getComputedStyle(r).display !== 'none' && getComputedStyle(r).fill === lit,
      ).length
      const mode = root.getAttribute('data-sam-mode')
      const visible = sel =>
        [...root.querySelectorAll(sel)].filter(e => getComputedStyle(e).display !== 'none')
      const strip = visible('[data-sam-strip]').map(e => e.getAttribute('data-sam-strip'))
      const rule = visible('[data-sam-rule]').map(e => e.getAttribute('data-sam-rule'))
      const spansOn = visible(
        `[data-sam-strip="${mode}"] [data-sam-cmp-span][data-on="true"]`,
      ).length
      const blocks = [...root.querySelectorAll('[data-sam-block]')]
        .filter(b => b.getAttribute('data-shown') === 'true')
        .map(b => ({
          j: Number(b.getAttribute('data-sam-block')),
          on: b.getAttribute('data-on') === 'true',
          tag: b.querySelector('[data-sam-tag]').textContent,
          h: Number(b.querySelector('[data-sam-score]').getAttribute('height')),
        }))
      const zero = Number(root.querySelector('.sam-zero').getAttribute('y1'))
      const bars = [...root.querySelectorAll('[data-sam-bar]')]
        .filter(b => b.getAttribute('data-shown') === 'true')
        .map(b => {
          const y = Number(b.getAttribute('y'))
          const h = Number(b.getAttribute('height'))
          return {
            s: Number(b.getAttribute('data-sam-bar')),
            on: b.getAttribute('data-on') === 'true',
            v: y < zero ? h : -h,
          }
        })
      const readout = Object.fromEntries(
        [...root.querySelectorAll('[data-sam-out]')].map(e => [
          e.getAttribute('data-sam-out'),
          e.textContent,
        ]),
      )
      const frame = root.querySelector('[data-sam-row-frame]')
      const highlighted = [...root.querySelectorAll('tr[data-sam-out-row]')]
        .filter(
          tr =>
            getComputedStyle(tr.cells[1]).backgroundColor !==
            getComputedStyle(tr.cells[1].closest('table').querySelector('thead th'))
              .backgroundColor,
        )
        .map(tr => tr.getAttribute('data-sam-out-row'))
      return {
        t,
        mode,
        litCells,
        cmpCells,
        strip,
        rule,
        spansOn,
        blocks,
        bars,
        readout,
        frameY: Number(frame.getAttribute('y')),
        litRows: [...root.querySelectorAll('[data-sam-row][data-lit="true"]')].map(r =>
          Number(r.getAttribute('data-sam-row')),
        ),
        head: root.querySelector('[data-sam-head]').textContent,
        valueText: root.querySelector('[data-sam-query-value]').textContent,
        highlighted,
      }
    })

  const num = s => Number(String(s).replace(/,/g, ''))
  const verify = (label, s) => {
    const r = s.readout
    const T = s.t + 1
    c.check(`${label}: one lit row at t`, s.litRows.length === 1 && s.litRows[0] === s.t, s.litRows)
    c.check(`${label}: frame on row t`, s.frameY === 82 + s.t * 6 - 1, s.frameY)
    c.check(
      `${label}: header and value show t`,
      s.head === `t=${s.t}` && s.valueText === String(s.t),
    )
    c.check(
      `${label}: one strip and one rule for mode`,
      s.strip.join() === s.mode && s.rule.join() === s.mode,
      s,
    )
    c.check(`${label}: dense = t+1`, num(r.dense) === T, r.dense)
    c.check(
      `${label}: N_t = cmp + slc + win`,
      num(r.nsa) === num(r.cmp) + num(r.slc) + num(r.win),
      r,
    )
    if (s.mode === 'cmp') {
      c.check(`${label}: lit wide cells = compressed count`, s.cmpCells === num(r.cmp), [
        s.cmpCells,
        r.cmp,
      ])
      c.check(`${label}: strip spans on = compressed count`, s.spansOn === num(r.cmp), [
        s.spansOn,
        r.cmp,
      ])
      c.check(`${label}: no fine-grained cell lit`, s.litCells.length === 0, s.litCells)
    }
    if (s.mode === 'slc' || s.mode === 'nsa') {
      const on = s.blocks.filter(b => b.on)
      const keys = new Set(on.flatMap(b => Array.from({ length: 8 }, (_, k) => b.j * 8 + k)))
      const slcKeys = [...keys].filter(k => k <= s.t)
      c.check(
        `${label}: kept blocks = min(n, blocks)`,
        on.length === Math.min(5, s.blocks.length),
        on,
      )
      c.check(`${label}: selected keys = readout`, slcKeys.length === num(r.slc), [
        slcKeys.length,
        r.slc,
      ])
      const forced = s.blocks.filter(b => b.tag)
      c.check(
        `${label}: forced blocks are kept`,
        forced.every(b => b.on) && forced.some(b => b.j === 0),
        forced,
      )
      const free = s.blocks.filter(b => !b.tag)
      const minKept = Math.min(...free.filter(b => b.on).map(b => b.h), Infinity)
      const maxDropped = Math.max(...free.filter(b => !b.on).map(b => b.h), -Infinity)
      c.check(`${label}: free blocks are the top scores`, minKept >= maxDropped, [
        minKept,
        maxDropped,
      ])
      const win = Array.from({ length: num(r.win) }, (_, k) => s.t - k)
      const expected = s.mode === 'slc' ? slcKeys : [...new Set([...slcKeys, ...win])]
      c.check(
        `${label}: lit cells = drawn ${s.mode === 'slc' ? 'blocks' : 'blocks + window'}`,
        expected.sort((a, b) => a - b).join() === s.litCells.join(),
        [expected.length, s.litCells.length],
      )
    }
    if (s.mode === 'win') {
      const exp = Array.from({ length: Math.min(24, T) }, (_, k) => s.t - k).sort((a, b) => a - b)
      c.check(
        `${label}: lit cells = last w keys`,
        exp.join() === s.litCells.join() && num(r.win) === exp.length,
        s.litCells,
      )
    }
    if (s.mode === 'dsa') {
      const top = s.bars.filter(b => b.on)
      c.check(
        `${label}: top-k bars = min(k, t+1) = readout`,
        top.length === Math.min(16, T) && num(r.dsa) === top.length,
        [top.length, r.dsa],
      )
      c.check(`${label}: lit cells = top-k keys`, top.map(b => b.s).join() === s.litCells.join(), [
        top.map(b => b.s),
        s.litCells,
      ])
      const minTop = Math.min(...top.map(b => b.v))
      const maxRest = Math.max(...s.bars.filter(b => !b.on).map(b => b.v), -Infinity)
      c.check(`${label}: top-k bars are the highest index scores`, minTop >= maxRest, [
        minTop,
        maxRest,
      ])
      c.check(`${label}: indexer scores every earlier key`, num(r.idx) === T && s.bars.length === T)
    }
    c.check(`${label}: readout rows for mode highlighted`, s.highlighted.length > 0, s.highlighted)
  }

  // Every mode by click, at several queries set by keyboard, on instance 0.
  const r0 = roots.nth(0)
  const slider = r0.locator('[data-sam-query-input]')
  for (const m of MODES) {
    await r0.locator(`label.sam-segment:has(input[value="${m}"])`).click()
    for (const key of ['End', 'ArrowLeft', 'PageDown', 'Home', 'ArrowRight']) {
      await slider.focus()
      await slider.press(key)
      verify(`${m} after ${key}`, await snapshot(0))
    }
    for (const t of [3, 7, 8, 23, 40, 55, 63]) {
      await slider.fill(String(t))
      verify(`${m} t=${t}`, await snapshot(0))
    }
    await c.shoot(r0, `mode-${m}.png`)
  }
  // Every query in every mode, by input events.
  let sweep = true
  for (const m of MODES) {
    await r0.locator(`label.sam-segment:has(input[value="${m}"])`).click()
    for (let t = 0; t < 64; t++) {
      await slider.fill(String(t))
      const before = c.results.checks.length
      verify(`sweep ${m} t=${t}`, await snapshot(0))
      const added = c.results.checks.splice(before)
      if (added.some(x => !x.ok)) {
        sweep = false
        c.results.checks.push(...added.filter(x => !x.ok))
      }
    }
  }
  c.check('sweep: all 5 modes x 64 queries agree with the readout', sweep)

  // Radio keyboard: arrow keys move the selection inside the native group.
  await r0.locator('input[value="nsa"]').focus()
  await page.keyboard.press('ArrowRight')
  c.check(
    'ArrowRight on radios selects next mode',
    (await r0.getAttribute('data-sam-mode')) === 'dsa',
  )
  const focusRing = await r0.evaluate(root => {
    const label = root.querySelector('input[value="dsa"]').closest('label')
    return getComputedStyle(label).outlineStyle
  })
  c.check('focused segment shows an outline', focusRing === 'solid', focusRing)
  await slider.focus()
  const sliderRing = await slider.evaluate(e => getComputedStyle(e).outlineStyle)
  c.check('focused slider shows an outline', sliderRing === 'solid', sliderRing)

  c.check('instance two unaffected', (await roots.nth(1).getAttribute('data-sam-mode')) === 'nsa')
  await roots.nth(1).locator('label.sam-segment:has(input[value="slc"])').click()
  await roots.nth(1).locator('[data-sam-query-input]').fill('30')
  verify('instance two slc t=30', await snapshot(1))
  c.check('instance one keeps its state', (await snapshot(0)).mode === 'dsa')

  // SPA navigation: cleanup, then a new nav must leave exactly one live handler.
  await page.evaluate(() => {
    for (const fn of window.__cleanups ?? []) fn()
    window.__cleanups = []
    document.dispatchEvent(new Event('nav'))
  })
  await r0.locator('label.sam-segment:has(input[value="win"])').click()
  await slider.fill('12')
  verify('after cleanup and nav', await snapshot(0))
  const cleanupsAfterNav = await page.evaluate(() => window.__cleanups.length)
  c.check('same cleanup count after nav as at load', cleanupsAfterNav === cleanupsAtLoad, [
    cleanupsAtLoad,
    cleanupsAfterNav,
  ])

  // Restore the embedded states before the standard screenshots.
  await r0.locator('label.sam-segment:has(input[value="dsa"])').click()
  await slider.fill('63')
  await roots.nth(1).locator('label.sam-segment:has(input[value="nsa"])').click()
  await roots.nth(1).locator('[data-sam-query-input]').fill('55')
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  ok = (await c.finish()) && ok
}

// --- MLAAttentionModes -----------------------------------------------------------------------
{
  const out = path.join(here, 'artifacts', 'mla-attention-modes')
  const pagePath = await renderFigurePage({
    component: 'MLAAttentionModes',
    label: 'MLA attention modes',
    out,
    instances: embeds('MLAAttentionModes'),
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  const root = page.locator('[data-mla-attention-modes]')

  const snapshot = () =>
    root.evaluate(root => {
      const visible = sel =>
        [...root.querySelectorAll(sel)].filter(e => getComputedStyle(e).display !== 'none')
      const panel = visible('[data-mam-panel]')
      const rows = panel.flatMap(p =>
        [...p.querySelectorAll('.mam-row')].map(row => {
          const track = row.querySelector('.mam-track').getBoundingClientRect().width
          const segs = [...row.querySelectorAll('.mam-seg')].map(s => ({
            dims: Number(s.getAttribute('data-dims')),
            w: s.getBoundingClientRect().width,
            kind: s.className.replace(/.*mam-seg--/, ''),
          }))
          return {
            label: row.querySelector('.mam-label').textContent,
            track,
            segs,
            dims: row.querySelector('.mam-dims').textContent,
          }
        }),
      )
      const table = Object.fromEntries(
        [...root.querySelectorAll('tbody tr')].map(tr => [
          tr.cells[0].textContent,
          {
            mha: tr.cells[1].textContent,
            mqa: tr.cells[2].textContent,
            bg: [
              getComputedStyle(tr.cells[1]).backgroundColor,
              getComputedStyle(tr.cells[2]).backgroundColor,
            ],
          },
        ]),
      )
      return {
        mode: root.getAttribute('data-mam-mode'),
        panels: panel.map(p => p.getAttribute('data-mam-panel')),
        rules: visible('[data-mam-rule]').map(p => p.getAttribute('data-mam-rule')),
        rows,
        table,
      }
    })

  const num = s => Number(String(s).replace(/[^0-9.]/g, ''))
  const verify = (label, s) => {
    c.check(
      `${label}: one panel and rule`,
      s.panels.join() === s.mode && s.rules.join() === s.mode,
      s,
    )
    // Widths: every bar scales with its dims against the widest vector (576 dims = full track).
    const unit = s.rows[0].track / 576
    const bad = s.rows.flatMap(r => r.segs.filter(g => Math.abs(g.w - g.dims * unit) > 1.5))
    c.check(`${label}: segment widths proportional to dims`, bad.length === 0, bad)
    const sumOk = s.rows.every(
      r => num(r.dims.split('=').pop()) === r.segs.reduce((a, g) => a + g.dims, 0),
    )
    c.check(`${label}: dims labels equal drawn segments`, sumOk)
    // The first two rows are the cache; the rest are what attention sees.
    const seen = s.rows.slice(2)
    const q = seen.find(r => r.label.startsWith('q'))
    const v = seen.find(r => r.label.startsWith('v'))
    const k = seen.find(r => r.label.startsWith('k'))
    const col = s.mode
    c.check(
      `${label}: score width = drawn query`,
      num(s.table['score width per head'][col]) === q.segs.reduce((a, g) => a + g.dims, 0),
    )
    c.check(
      `${label}: value width = drawn value`,
      num(s.table['value width per head'][col]) === v.segs.reduce((a, g) => a + g.dims, 0),
    )
    c.check(
      `${label}: FLOPs per key = 2 (score + value)`,
      num(s.table['FLOPs per key per head'][col]) ===
        2 * (q.segs.reduce((a, g) => a + g.dims, 0) + v.segs.reduce((a, g) => a + g.dims, 0)),
    )
    const cache = s.rows.filter(r => r.segs.every(g => g.kind === 'cache')).slice(0, 2)
    const cached = cache.reduce((a, r) => a + r.segs[0].dims, 0)
    const [vals, bytes] = s.table['cached per token per layer, values · bytes'][col]
      .split('·')
      .map(num)
    c.check(
      `${label}: cache = drawn c_KV + k_R, 2 B each`,
      vals === cached && bytes === 2 * cached,
      [vals, bytes, cached],
    )
    const kv = s.table['K and V per token seen by attention, values · bytes'][col]
      .split('·')
      .map(num)
    const perHead = k.segs.reduce((a, g) => a + g.dims, 0) + v.segs.reduce((a, g) => a + g.dims, 0)
    const expected = col === 'mha' ? 128 * perHead : k.segs.reduce((a, g) => a + g.dims, 0)
    c.check(
      `${label}: K,V per token = drawn ${col === 'mha' ? '128 heads x (k + v)' : 'shared k (v is its c_KV part)'}`,
      kv[0] === expected && kv[1] === 2 * expected,
      [kv, expected],
    )
    const built = s.rows.flatMap(r => r.segs).filter(g => g.kind === 'built').length
    c.check(`${label}: salmon marks per-head built vectors`, built > 0)
    const sel = Object.values(s.table)[0].bg
    c.check(
      `${label}: selected column highlighted`,
      col === 'mha' ? sel[0] !== sel[1] : sel[1] !== sel[0],
    )
  }

  verify('initial mha', await snapshot())
  await c.shoot(root, 'mode-mha.png')
  await root.locator('label.mam-segment:has(input[value="mqa"])').click()
  verify('click mqa', await snapshot())
  await c.shoot(root, 'mode-mqa.png')
  await root.locator('input[value="mqa"]').focus()
  await page.keyboard.press('ArrowLeft')
  verify('ArrowLeft back to mha', await snapshot())
  const ring = await root.evaluate(
    r => getComputedStyle(r.querySelector('input[value="mha"]').closest('label')).outlineStyle,
  )
  c.check('focused segment shows an outline', ring === 'solid', ring)
  await page.keyboard.press('ArrowRight')
  verify('ArrowRight to mqa', await snapshot())
  await page.evaluate(() => {
    for (const fn of window.__cleanups ?? []) fn()
    window.__cleanups = []
    document.dispatchEvent(new Event('nav'))
  })
  c.check('nav keeps the checked mode', (await snapshot()).mode === 'mqa')
  await root.locator('label.mam-segment:has(input[value="mha"])').click()
  verify('after cleanup and nav', await snapshot())
  await standardChecks(page, errors, c, '.zoomable')
  // The narrow-width MQA panel, where labels sit above their bars.
  await page.setViewportSize({ width: 320, height: 1400 })
  await root.locator('label.mam-segment:has(input[value="mqa"])').click()
  await c.shoot(page.locator('.zoomable'), '320-mqa.png')
  await browser.close()
  ok = (await c.finish()) && ok
}

process.exit(ok ? 0 : 1)
