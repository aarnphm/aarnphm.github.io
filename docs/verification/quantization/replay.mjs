// Renders NumberFormatBits, MXBlockScale and MXComputeFlow through the shared figure harness,
// drives every control by click and keyboard, and checks the drawing against the readouts and
// against known encodings. Writes pages, screenshots and results.json under ./artifacts/<figure>.
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const art = name => path.join(here, 'artifacts', name)
const noMotion = '*,*::after{transition:none!important}'
let allPassed = true

const f32hex = x => {
  const dv = new DataView(new ArrayBuffer(4))
  dv.setFloat32(0, x)
  return `0x${dv.getUint32(0).toString(16).toUpperCase().padStart(8, '0')}`
}

async function finish(c, page, errors, browser) {
  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  allPassed = (await c.finish()) && allPassed
}

// ---------------------------------------------------------------- NumberFormatBits
{
  const out = art('number-format-bits')
  const pagePath = await renderFigurePage({
    component: 'NumberFormatBits',
    instances: [
      { caption: 'Instance one.', value: 'pi', format: 'fp16' },
      { caption: 'Instance two.', value: 'tiny', format: 'e5m2' },
    ],
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  // Computed colours are read right after each change, so transitions are switched off.
  await page.addStyleTag({ content: noMotion })
  const c = checker(out)
  const root = page.locator('[data-number-format-bits]').nth(0)
  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-number-format-bits]')[i]
      const out = Object.fromEntries(
        [...r.querySelectorAll('[data-nfb-out]')].map(d => [
          d.dataset.nfbOut,
          d.textContent.trim(),
        ]),
      )
      const sel = r.querySelector('[data-nfb-format-input]:checked').value
      const row = r.querySelector(`[data-nfb-row="${sel}"]`)
      const bits = [...row.querySelectorAll('.nfb-bit')].map(b => b.dataset.nfbOn).join('')
      const lit = [...r.querySelectorAll('.nfb-name')].filter(
        n => getComputedStyle(n).backgroundColor !== 'rgba(0, 0, 0, 0)',
      ).length
      return { out, sel, bits, lit, value: r.querySelector('[data-nfb-value-input]:checked').value }
    }, i)

  // Known encodings: IEEE binary16 and bfloat16 tables, OCP FP8/FP4 code points, and the note's
  // int8 example (0.3 -> 38 with S = 1/127). FP32 rows are checked against DataView below.
  const known = {
    p03: { bf16: '0x3E9A', fp16: '0x34CD', e5m2: '0x35', e4m3: '0x2A', e2m1: '0x1', int8: '0x26' },
    pi: { bf16: '0x4049', fp16: '0x4248', e5m2: '0x42', e4m3: '0x45', e2m1: '0x5', int8: '0x7F' },
    m300: { bf16: '0xC396', fp16: '0xDCB0', e5m2: '0xDD', e4m3: '0xF9', e2m1: '0xF', int8: '0x81' },
    k1: { bf16: '0x447A', fp16: '0x63D0', e5m2: '0x64', e4m3: '0x7E', e2m1: '0x7', int8: '0x7F' },
    k70: { bf16: '0x4789', fp16: '0x7C00', e5m2: '0x7B', e4m3: '0x7E', e2m1: '0x7', int8: '0x7F' },
    tiny: { bf16: '0x3728', fp16: '0x00A8', e5m2: '0x01', e4m3: '0x00', e2m1: '0x0', int8: '0x00' },
  }
  const inputs = { p03: 0.3, pi: Math.PI, m300: -300, k1: 1000, k70: 70000, tiny: 1e-5 }
  const ranges = {
    fp32: ['3.4028×1038', '1.1755×10−38', '1.4013×10−45'],
    bf16: ['3.3895×1038', '1.1755×10−38', '9.1835×10−41'],
    fp16: ['65504', '6.1035×10−5', '5.9605×10−8'],
    e5m2: ['57344', '6.1035×10−5', '1.5259×10−5'],
    e4m3: ['448', '0.015625', '0.0019531'],
    e2m1: ['6', '1', '0.5'],
  }
  const s0 = await snap(0)
  c.check('SSR: pi in FP16 is 0x4248', s0.out.hex === '0x4248', s0.out)
  c.check('SSR: one salmon row name', s0.lit === 1, s0.lit)
  const results = {}
  for (const [vid, x] of Object.entries(inputs)) {
    await root.locator(`[data-nfb-value-input][value="${vid}"]`).check({ force: true })
    for (const fid of ['fp32', 'bf16', 'fp16', 'e5m2', 'e4m3', 'e2m1', 'int8']) {
      await root.locator(`[data-nfb-row="${fid}"]`).click()
      const s = await snap(0)
      const expected = fid === 'fp32' ? f32hex(x) : known[vid][fid]
      results[`${vid}:${fid}`] = {
        hex: s.out.hex,
        dec: s.out.dec,
        rel: s.out.rel,
        outcome: s.out.outcome,
      }
      c.check(`${vid} ${fid} encodes to ${expected}`, s.out.hex === expected, s.out.hex)
      c.check(
        `${vid} ${fid} drawn bits equal stored bits`,
        parseInt(s.bits, 2) === parseInt(s.out.hex, 16) &&
          s.bits.length === (s.out.hex.length - 2) * 4,
        { bits: s.bits, hex: s.out.hex },
      )
      if (ranges[fid] && vid === 'pi')
        c.check(
          `${fid} range readout`,
          [s.out.max, s.out.minN, s.out.minS].every((v, k) => v === ranges[fid][k]),
          [s.out.max, s.out.minN, s.out.minS],
        )
    }
  }
  c.results.encodings = results
  c.check(
    'FP16 overflow of 70000 is infinity',
    results['k70:fp16'].outcome.includes('infinity'),
    results['k70:fp16'],
  )
  c.check('E4M3 saturates 1000 to 448', results['k1:e4m3'].dec === '448', results['k1:e4m3'])
  c.check(
    'E4M3 flushes 1e-5 to zero',
    results['tiny:e4m3'].outcome.includes('zero'),
    results['tiny:e4m3'],
  )
  c.check(
    'FP16 keeps 1e-5 as a subnormal',
    results['tiny:fp16'].outcome.includes('subnormal'),
    results['tiny:fp16'],
  )

  // Keyboard: arrow keys inside each native radio group.
  await root.locator('[data-nfb-value-input][value="pi"]').check({ force: true })
  await root.locator('[data-nfb-value-input][value="pi"]').focus()
  await page.keyboard.press('ArrowRight')
  let k = await snap(0)
  c.check('ArrowRight moves the input to -300', k.value === 'm300', k.value)
  await root.locator('[data-nfb-row="fp16"]').click()
  await root.locator('[data-nfb-format-input][value="fp16"]').focus()
  await page.keyboard.press('ArrowDown')
  k = await snap(0)
  c.check('ArrowDown moves the decoded row to E5M2', k.sel === 'e5m2' && k.out.hex === '0xDD', k)
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused row shows an outline', outline === 'solid', outline)
  await c.shoot(root, 'focus-keyboard.png')
  const second = await snap(1)
  c.check(
    'Second instance keeps its own state',
    second.value === 'tiny' && second.sel === 'e5m2',
    second,
  )
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await root.locator('[data-nfb-value-input][value="k70"]').check({ force: true })
  const after = await snap(0)
  c.check('Cleanup removes the change listener', after.out.hex === '0xDD', after.out.hex)
  await page.reload()
  await page.addStyleTag({ content: noMotion })
  await root.locator('[data-nfb-value-input][value="pi"]').check({ force: true })
  await root.locator('[data-nfb-row="fp16"]').click()
  await finish(c, page, errors, browser)
}

// ---------------------------------------------------------------- MXBlockScale
{
  const out = art('mx-block-scale')
  const pagePath = await renderFigurePage({
    component: 'MXBlockScale',
    instances: [
      { caption: 'Instance one.', format: 'e2m1', block: 'max7' },
      { caption: 'Instance two.', format: 'e4m3', block: 'spike' },
    ],
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  // Computed colours are read right after each change, so transitions are switched off.
  await page.addStyleTag({ content: noMotion })
  const c = checker(out)
  const root = page.locator('[data-mx-block-scale]').nth(0)
  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-mx-block-scale]')[i]
      const out = Object.fromEntries(
        [...r.querySelectorAll('[data-mxb-out]')].map(d => [
          d.dataset.mxbOut,
          d.textContent.trim(),
        ]),
      )
      const cols = [...r.querySelectorAll('[data-mxb-col]')].map(g => ({
        orig: +g.querySelector('.mxb-orig').getAttribute('height'),
        origY: +g.querySelector('.mxb-orig').getAttribute('y'),
        deq: +g.querySelector('.mxb-deq').getAttribute('height'),
        deqY: +g.querySelector('.mxb-deq').getAttribute('y'),
        max: g.dataset.mxbMax === 'true',
      }))
      const ticks = [
        ...r
          .querySelector('[data-mxb-ticks]')
          .getAttribute('d')
          .matchAll(/M[\d.]+ ([\d.]+)/g),
      ].map(m => +m[1])
      return {
        out,
        cols,
        ticks,
        format: r.querySelector('[data-mxb-format-input]:checked').value,
        block: r.querySelector('[data-mxb-block-input]:checked').value,
      }
    }, i)
  const zeroY = await page.evaluate(() => +document.querySelector('.mxb-zero').getAttribute('y1'))
  const states = {}
  for (const block of ['normal', 'max7', 'spike'])
    for (const format of ['e2m1', 'e2m3', 'e3m2', 'e4m3']) {
      await root.locator(`[data-mxb-block-input][value="${block}"]`).check({ force: true })
      await root.locator(`[data-mxb-format-input][value="${format}"]`).check({ force: true })
      const s = await snap(0)
      states[`${block}:${format}`] = s.out
      const flushedDrawn = s.cols.filter(col => col.orig > 0 && col.deq === 0).length
      c.check(
        `${block}:${format} flushed cells equal readout`,
        s.out.flushed.startsWith(`${flushedDrawn}/32`),
        { drawn: flushedDrawn, out: s.out.flushed },
      )
      // Every dequantized bar ends on a drawn level tick.
      const ends = s.cols
        .filter(col => col.deq > 0)
        .map(col => (col.deqY < zeroY - 0.01 ? col.deqY : col.deqY + col.deq))
      const off = ends.filter(y => !s.ticks.some(t => Math.abs(t - y) < 0.02))
      c.check(`${block}:${format} dequantized bars end on level ticks`, off.length === 0, off)
      c.check(
        `${block}:${format} exactly one block maximum lit`,
        s.cols.filter(col => col.max).length === 1,
      )
    }
  c.results.states = states
  c.check(
    'max 7 in E2M1: S = 1, one value clamped to 6',
    states['max7:e2m1'].scale === '20=1' &&
      states['max7:e2m1'].clipped.startsWith('1/32') &&
      states['max7:e2m1'].top.includes('6S=6'),
    states['max7:e2m1'],
  )
  c.check(
    'outlier 24 in E2M1: shared exponent 4 - 2 = 2, E8M0 code 129',
    states['spike:e2m1'].shared.endsWith('=4−2=2') && states['spike:e2m1'].code.endsWith('=129'),
    states['spike:e2m1'],
  )
  c.check(
    'outlier 24 in E4M3: shared exponent 4 - 8 = -4',
    states['spike:e4m3'].shared.endsWith('=4−8=−4'),
    states['spike:e4m3'],
  )
  c.check(
    'MXFP4 stores 4.25 bits per element',
    states['normal:e2m1'].bits.includes('=4.25'),
    states['normal:e2m1'].bits,
  )
  c.check(
    'MXFP8 stores 8.25 bits per element',
    states['normal:e4m3'].bits.includes('=8.25'),
    states['normal:e4m3'].bits,
  )
  {
    await root.locator('[data-mxb-block-input][value="max7"]').check({ force: true })
    await root.locator('[data-mxb-format-input][value="e2m1"]').check({ force: true })
    const s = await snap(0)
    const m = s.cols.find(col => col.max)
    c.check(
      'max 7 drawn: dequantized bar is 6/7 of the input bar',
      Math.abs(m.deq / m.orig - 6 / 7) < 0.002,
      m,
    )
  }
  await root.locator('[data-mxb-format-input][value="e2m1"]').focus()
  await page.keyboard.press('ArrowRight')
  const k = await snap(0)
  c.check('ArrowRight moves the format to E2M3', k.format === 'e2m3', k.format)
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused segment shows an outline', outline === 'solid', outline)
  await c.shoot(root, 'focus-keyboard.png')
  const second = await snap(1)
  c.check(
    'Second instance keeps its own state',
    second.format === 'e4m3' && second.block === 'spike',
    second.format,
  )
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await root.locator('[data-mxb-block-input][value="spike"]').check({ force: true })
  const after = await snap(0)
  c.check('Cleanup removes the change listener', after.out.amax === k.out.amax, after.out.amax)
  await page.reload()
  await page.addStyleTag({ content: noMotion })
  await finish(c, page, errors, browser)
}

// ---------------------------------------------------------------- MXComputeFlow
{
  const out = art('mx-compute-flow')
  const pagePath = await renderFigurePage({
    component: 'MXComputeFlow',
    instances: [
      { caption: 'Instance one.', pass: 'forward' },
      { caption: 'Instance two.', pass: 'dgrad' },
    ],
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  // Computed colours are read right after each change, so transitions are switched off.
  await page.addStyleTag({ content: noMotion })
  const c = checker(out)
  const root = page.locator('[data-mx-compute-flow]').nth(0)
  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-mx-compute-flow]')[i]
      const panels = [...r.querySelectorAll('g[data-mxf-panel]')].filter(
        g => getComputedStyle(g).display !== 'none',
      )
      const g = panels[0]
      const blocks = [...g.querySelectorAll('.mxf-block')].map(b => ({
        w: +b.getAttribute('width'),
        h: +b.getAttribute('height'),
        lit: b.classList.contains('mxf-block--lit'),
      }))
      const readouts = [...r.querySelectorAll('.mxf-readout')].filter(
        d => getComputedStyle(d).display !== 'none',
      )
      const out = Object.fromEntries(
        [...readouts[0].querySelectorAll('div')].map(d => [
          d.querySelector('dt').textContent,
          d.querySelector('dd').textContent.trim(),
        ]),
      )
      const used = [...r.querySelectorAll('tr[data-mxf-use]')]
        .filter(
          tr =>
            getComputedStyle(tr.querySelector('td')).backgroundColor !==
            getComputedStyle(r.querySelector('thead th')).backgroundColor,
        )
        .map(tr => tr.dataset.mxfUse)
      const differ = [...r.querySelectorAll('.mxf-num')].map(td => td.textContent)
      return {
        pass: r.dataset.mxfPass,
        visible: panels.map(p => p.dataset.mxfPanel),
        readouts: readouts.length,
        horizontal: blocks.filter(b => b.w > b.h),
        vertical: blocks.filter(b => b.h > b.w),
        out,
        used,
        differ,
      }
    }, i)
  // Operand shapes and reduction length per product, from Figure 2 of Rouhani et al. (2023).
  const expect = {
    forward: { R: 64, leftRows: 32, rightCols: 32 },
    wgrad: { R: 32, leftRows: 64, rightCols: 32 },
    dgrad: { R: 32, leftRows: 32, rightCols: 64 },
  }
  const passes = {}
  for (const [id, e] of Object.entries(expect)) {
    await root.locator(`[data-mxf-pass-input][value="${id}"]`).check({ force: true })
    const s = await snap(0)
    passes[id] = s.out
    const perLine = e.R / 32
    c.check(
      `${id}: only its panel and readout show`,
      s.visible.join() === id && s.readouts === 1,
      s.visible,
    )
    c.check(
      `${id}: left operand has ${e.leftRows} rows of ${perLine} blocks along the reduction`,
      s.horizontal.length === e.leftRows * perLine,
      s.horizontal.length,
    )
    c.check(
      `${id}: right operand has ${e.rightCols} columns of ${perLine} blocks`,
      s.vertical.length === e.rightCols * perLine,
      s.vertical.length,
    )
    c.check(
      `${id}: one lit row and one lit column`,
      s.horizontal.filter(b => b.lit).length === perLine &&
        s.vertical.filter(b => b.lit).length === perLine,
    )
    c.check(
      `${id}: block run is 32 elements`,
      [...s.horizontal.map(b => b.w), ...s.vertical.map(b => b.h)].every(v => v === 32 * 3 - 1.5),
    )
    c.check(
      `${id}: reduction length readout`,
      s.out['Reduction length'].endsWith(`=${e.R}`),
      s.out['Reduction length'],
    )
    c.check(
      `${id}: two MX copies highlighted, both used by this product`,
      s.used.length === 2 && s.used.every(u => u === id),
      s.used,
    )
  }
  c.results.passes = passes
  const s = await snap(0)
  c.results.differ = s.differ
  c.check(
    'Each tensor has two MX copies that differ',
    s.differ.every(d => +d.split(' / ')[0] > 0),
    s.differ,
  )
  await root.locator('[data-mxf-pass-input][value="forward"]').check({ force: true })
  await root.locator('[data-mxf-pass-input][value="forward"]').focus()
  await page.keyboard.press('ArrowRight')
  const k = await snap(0)
  c.check('ArrowRight selects the weight gradient', k.pass === 'wgrad', k.pass)
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused segment shows an outline', outline === 'solid', outline)
  await c.shoot(root, 'focus-keyboard.png')
  const second = await snap(1)
  c.check('Second instance keeps its own state', second.pass === 'dgrad', second.pass)
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await root.locator('[data-mxf-pass-input][value="dgrad"]').check({ force: true })
  const after = await snap(0)
  c.check('Cleanup removes the change listener', after.pass === 'wgrad', after.pass)
  await page.reload()
  await page.addStyleTag({ content: noMotion })
  await finish(c, page, errors, browser)
}

process.exit(allPassed ? 0 : 1)
