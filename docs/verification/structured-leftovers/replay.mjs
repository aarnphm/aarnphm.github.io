// Replay: node docs/verification/structured-leftovers/replay.mjs
// Renders TokenBoundaryShift, GrammarBitmaskBatch and LogitsProcessorTrace with the props the notes
// give them, drives every control by click and keyboard, and checks each drawn state against
// oracles built here from the exported model data, the note's JSON language and (when present)
// the vLLM v0.5.0 source. Writes pages, screenshots and results.json under ./artifacts.
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  checker,
  openFigurePage,
  renderFigurePage,
  repo,
  standardChecks,
} from '../figure-harness.mjs'
import { captions } from './parse-captions.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const art = path.join(here, 'artifacts')
const data = JSON.parse(
  readFileSync(path.join(repo, 'quartz/components/mdx/jump-forward-boundaries.json'), 'utf8'),
)
// vLLM 0.5.0 sdist from PyPI, unpacked; source-line checks are skipped when it is absent.
const VLLM_SRC = process.env.VLLM_SRC
const embeds = captions()
const propsOf = name => embeds.filter(e => e.component === name).map(e => e.props)
const near = (a, b, tol) => Math.abs(a - b) <= tol
const pct = p => (p >= 0.01 ? `${(p * 100).toFixed(1)}%` : `${(p * 100).toPrecision(2)}%`)
const JSON_LANG = ['Paul', 'John'].flatMap(n => ['20', '30'].map(a => `{"name":"${n}","age":${a}}`))
let passed = true

// ---------- TokenBoundaryShift ----------
{
  const out = path.join(art, 'token-boundary-shift')
  const instances = propsOf('TokenBoundaryShift')
  const pagePath = await renderFigurePage({
    component: 'TokenBoundaryShift',
    out,
    label: 'Variance in sampling distribution',
    instances,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  c.check('one embed in the notes', instances.length === 1, instances.length)

  // Oracle: renormalize the exported probabilities and recompute every readout here.
  const order = ['Paul', 'Pa', 'P', 'John', 'Jo', 'J']
  const idx = order.map(t => data.allowed.findIndex(a => a.text === t))
  const norm = b => {
    const s = b.probs.reduce((a, p) => a + p, 0)
    return b.probs.map(p => p / s)
  }
  const nat = data.boundaries.find(b => b.natural)
  const pn = norm(nat)
  const expect = b => {
    const q = norm(b)
    const kl = pn.reduce((a, p, i) => a + (p > 0 ? p * Math.log(p / q[i]) : 0), 0)
    const tv = 0.5 * pn.reduce((a, p, i) => a + Math.abs(p - q[i]), 0)
    const paul = idx.slice(0, 3).reduce((a, i) => a + q[i], 0)
    return { q, kl, tv, paul }
  }

  c.check(
    'natural split is the tokenizer split of the forced text',
    JSON.stringify(nat.pieces) === JSON.stringify(['{"', 'name', '":"']),
    nat.pieces,
  )
  c.check(
    'every allowed token keeps the output inside the language',
    data.allowed.every(a => JSON_LANG.some(w => w.startsWith(data.forced + a.text))),
    data.allowed.map(a => a.text),
  )
  c.check(
    'all eight segmentations of name and the char split are present',
    data.boundaries.length === 9 &&
      data.boundaries.every(b => b.pieces.join('') === data.forced) &&
      new Set(data.boundaries.map(b => b.id)).size === 9,
    data.boundaries.map(b => b.id),
  )

  const root = page.locator('[data-token-boundary-shift]').first()
  const snap = () =>
    page.evaluate(() => {
      const r = document.querySelector('[data-token-boundary-shift]')
      const visible = el => el.getBoundingClientRect().width > 0
      const groups = [...r.querySelectorAll('[data-tbs-boundary]')].filter(visible)
      const tape = groups
        .flatMap(g => [...g.querySelectorAll('.tbs-piece')])
        .map(t => t.textContent)
      const bars = Object.fromEntries(
        groups.flatMap(g =>
          [...g.querySelectorAll('[data-tbs-bar]')].map(b => [
            b.dataset.tbsBar,
            Number(b.getAttribute('height')),
          ]),
        ),
      )
      const outs = Object.fromEntries(
        [...r.querySelectorAll('[data-tbs-out]')].map(d => [d.dataset.tbsOut, d.textContent]),
      )
      const tokens = [...r.querySelectorAll('[data-tbs-token]')].map(td => td.textContent)
      const checked = r.querySelector('[data-tbs-input]:checked').value
      // One visible group for the tape and one per token column.
      return { tape, bars, outs, tokens, checked, groups: groups.length }
    })

  const PLOT_H = 150
  const Y_MAX = 0.8
  for (const b of data.boundaries) {
    await root.locator(`input[value="${b.id}"]`).check({ force: true })
    const s = await snap()
    const e = expect(b)
    c.check(`${b.id}: radio checked`, s.checked === b.id, s.checked)
    c.check(`${b.id}: one tape and six bar groups shown`, s.groups === 7, s.groups)
    c.check(
      `${b.id}: tape spells the boundary`,
      JSON.stringify(s.tape) === JSON.stringify(b.pieces),
      s.tape,
    )
    c.check(
      `${b.id}: salmon bar heights equal renormalized probabilities`,
      order.every((t, k) => near(s.bars[t], (PLOT_H * Math.min(e.q[idx[k]], Y_MAX)) / Y_MAX, 0.01)),
      s.bars,
    )
    c.check(
      `${b.id}: table equals model`,
      JSON.stringify(s.tokens) === JSON.stringify(idx.map(i => pct(e.q[i]))),
      s.tokens,
    )
    c.check(
      `${b.id}: readouts equal oracle`,
      s.outs.forced === `3 → ${b.pieces.length}` &&
        s.outs.mass === `${pct(nat.allowedMass)} → ${pct(b.allowedMass)}` &&
        s.outs.paul === `${pct(expect(nat).paul)} → ${pct(e.paul)}` &&
        s.outs.kl === `${e.kl.toFixed(4)} nats` &&
        s.outs.tv === e.tv.toFixed(3),
      s.outs,
    )
    c.check(
      `${b.id}: KL and TV agree with the extraction script`,
      near(e.kl, b.kl, 1e-4 + 1e-3 * b.kl) && near(e.tv, b.tv, 1e-4),
      { page: [e.kl, e.tv], script: [b.kl, b.tv] },
    )
  }
  const shift = data.boundaries.map(b => expect(b).paul)
  c.check(
    'caption values of the Paul branch match the data',
    pct(expect(nat).paul) === '21.2%' &&
      pct(Math.min(...shift)) === '20.3%' &&
      pct(Math.max(...shift)) === '40.4%',
    shift,
  )

  // Keyboard: the radio group moves with the arrows and updates the figure.
  await root.locator('input[value="name"]').check({ force: true })
  await root.locator('input[value="name"]').focus()
  await page.keyboard.press('ArrowRight')
  const k1 = await snap()
  c.check(
    'ArrowRight selects the next segmentation',
    k1.checked === 'n|a|m|e' && k1.outs.forced === '3 → 6',
    k1,
  )
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('focused segment shows an outline', outline === 'solid', outline)
  await c.shoot(page.locator('.zoomable').first(), 'focus.png')
  await root.locator('input[value="chars"]').check({ force: true })
  await c.shoot(page.locator('.zoomable').first(), 'chars.png')
  await root.locator(`input[value="${instances[0].boundary}"]`).check({ force: true })

  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  passed = (await c.finish()) && passed
}

// ---------- GrammarBitmaskBatch ----------
{
  const out = path.join(art, 'grammar-bitmask-batch')
  const instances = propsOf('GrammarBitmaskBatch')
  const pagePath = await renderFigurePage({
    component: 'GrammarBitmaskBatch',
    out,
    label: 'Scheduler broadcast bitmask',
    instances: [...instances, { request: 'r1', workers: 4, caption: 'Second instance.' }],
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  // Reduced motion drops the fill transitions, so computed colours are final when read.
  await page.emulateMedia({ reducedMotion: 'reduce' })
  const c = checker(out)
  c.check('one embed in the notes', instances.length === 1, instances.length)
  c.check(
    'bitmask width is ceil(V / 32)',
    data.bitmaskWords === Math.ceil(data.vocab / 32),
    data.bitmaskWords,
  )

  const stateOf = { r1: 0, r2: 9, r3: 22 }
  const allowedOracle = st => {
    const s = data.states.find(x => x.state === st)
    return { s, ok: s.allowed.every(a => JSON_LANG.some(w => w.startsWith(s.consumed + a.text))) }
  }
  for (const st of [0, 9, 22]) {
    const { s, ok } = allowedOracle(st)
    c.check(
      `state ${st}: allowed tokens are viable continuations`,
      ok && s.allowed.length > 0,
      s.allowed,
    )
  }

  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-grammar-bitmask-batch]')[i]
      const visible = el => el.getBoundingClientRect().width > 0
      const req = r.querySelector('[data-gbb-input="request"]:checked').value
      const n = r.querySelector('[data-gbb-input="workers"]:checked').value
      const litRow = r.querySelector(`.gbb-req[data-gbb-req="${req}"]`)
      const lit = [...litRow.querySelectorAll('.gbb-bit')].map(b => Number(b.dataset.gbbBin))
      const fill = getComputedStyle(litRow.querySelector('.gbb-bit') ?? litRow).fill
      const set = [...r.querySelectorAll('[data-gbb-n]')].find(visible)
      const boxes = [...set.querySelectorAll('.gbb-box')].map(b => ({
        x: Number(b.getAttribute('x')),
        y: Number(b.getAttribute('y')),
        w: Number(b.getAttribute('width')),
      }))
      const wires = [...set.querySelectorAll('.gbb-wire')].map(l =>
        ['x1', 'x2', 'y1', 'y2'].map(a => Number(l.getAttribute(a))),
      )
      const litLogits = [...set.querySelectorAll('.gbb-logit[data-on="true"]')].length
      const detail = [...r.querySelectorAll('.gbb-detail')].filter(visible)
      const dd = detail.flatMap(d => [...d.querySelectorAll('dd')].map(x => x.textContent))
      const rows = detail.flatMap(d =>
        [...d.querySelectorAll('tbody tr')].map(tr => [...tr.children].map(x => x.textContent)),
      )
      return {
        req,
        n,
        lit,
        fill,
        boxes,
        wires,
        litLogits,
        details: detail.length,
        dd,
        rows,
        attr: r.dataset,
      }
    }, i)

  const root = page.locator('[data-grammar-bitmask-batch]').first()
  for (const req of ['r0', 'r1', 'r2', 'r3']) {
    for (const n of ['1', '2', '4']) {
      await root.locator(`input[value="${req}"]`).check({ force: true })
      await root.locator(`input[data-gbb-input="workers"][value="${n}"]`).check({ force: true })
      const s = await snap(0)
      const tag = `${req}, N=${n}`
      c.check(
        `${tag}: controls and root agree`,
        s.req === req && s.n === n && s.attr.gbbRequest === req,
        s.attr,
      )
      c.check(`${tag}: N worker boxes`, s.boxes.length === Number(n), s.boxes.length)
      c.check(`${tag}: lit logits row in every worker`, s.litLogits === Number(n), s.litLogits)
      c.check(`${tag}: one detail panel`, s.details === 1, s.details)
      const bus = s.wires[0]
      const centers = s.boxes.map(b => b.x + b.w / 2)
      const top = s.boxes[0].y
      const drops = s.wires.filter(w => w[0] === w[1] && w[3] === top)
      c.check(
        `${tag}: bus spans the worker drops and each drop ends on a box`,
        bus[0] <= Math.min(...centers) &&
          bus[1] >= Math.max(...centers) &&
          drops.length === Number(n) &&
          drops.every(w => centers.some(x => near(x, w[0], 0.01)) && w[2] === bus[2]),
        { bus, drops, centers },
      )
      if (n !== '2') continue
      if (req === 'r0') {
        c.check(
          'r0: no bits drawn, row not applied',
          s.lit.length === 0 && s.dd[1] === 'row not applied',
          s,
        )
        const top = data.states[0].top[0]
        c.check(
          'r0: greedy pick is the unmasked argmax',
          s.dd[2] === `${top.text}  ${pct(top.p)}` && s.dd[3] === s.dd[2],
          s.dd,
        )
        continue
      }
      const { s: st } = allowedOracle(stateOf[req])
      const words = [...new Set(st.allowed.map(a => a.id >> 5))]
      const bins = [...new Set(words.map(w => Math.floor(w / 40)))].sort((a, b) => a - b)
      c.check(
        `${req}: lit bins are the words of allowed IDs`,
        JSON.stringify(s.lit.sort((a, b) => a - b)) === JSON.stringify(bins),
        { lit: s.lit, bins },
      )
      c.check(`${req}: lit bins are salmon`, s.fill === 'rgb(253, 178, 162)', s.fill)
      c.check(
        `${req}: word count`,
        s.dd[1] === `${words.length} of ${data.bitmaskWords.toLocaleString('en-US')}`,
        s.dd[1],
      )
      const kept = st.allowed.reduce((a, t) => a + t.p, 0)
      const best = st.allowed.reduce((a, t) => (t.p > a.p ? t : a))
      c.check(
        `${req}: unmasked argmax`,
        s.dd[2] === `${st.top[0].text}  ${pct(st.top[0].p)}`,
        s.dd[2],
      )
      c.check(
        `${req}: greedy pick after mask`,
        s.dd[3] === `${best.text}  ${pct(best.p / kept)}`,
        s.dd[3],
      )
      c.check(
        `${req}: table rows give id, word and bit`,
        s.rows.length === st.allowed.length &&
          s.rows.every(
            (row, k) =>
              row[1] === String(st.allowed[k].id) &&
              row[2] === `${st.allowed[k].id >> 5}, ${st.allowed[k].id & 31}`,
          ),
        s.rows,
      )
    }
  }
  const waiting = await page.evaluate(
    () => document.querySelector('.gbb-req[data-gbb-req="r4"]').textContent,
  )
  c.check(
    'r4 waits for its grammar outside the batch',
    waiting.includes('WAITING_FOR_FSM'),
    waiting,
  )
  c.check('r4 is not selectable', (await root.locator('input[value="r4"]').count()) === 0, null)
  // Second instance keeps its own state.
  const s2 = await snap(1)
  c.check(
    'second instance keeps r1 and N=4',
    s2.req === 'r1' && s2.n === '4' && s2.boxes.length === 4,
    s2.attr,
  )

  // Keyboard on both radio groups.
  await root.locator('input[value="r1"]').check({ force: true })
  await root.locator('input[value="r1"]').focus()
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('Tab')
  await page.keyboard.press('ArrowLeft')
  const k = await snap(0)
  c.check(
    'arrow keys move request and worker selection',
    k.req === 'r2' && k.n === '2' && k.boxes.length === 2,
    k.attr,
  )
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('focused segment shows an outline', outline === 'solid', outline)
  await c.shoot(page.locator('.zoomable').first(), 'focus.png')
  await root.locator('input[value="r1"]').check({ force: true })
  await root.locator('input[data-gbb-input="workers"][value="4"]').check({ force: true })
  await c.shoot(page.locator('.zoomable').first(), 'r1-n4.png')
  await root.locator('input[value="r2"]').check({ force: true })
  await root.locator('input[data-gbb-input="workers"][value="2"]').check({ force: true })

  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  passed = (await c.finish()) && passed
}

// ---------- LogitsProcessorTrace ----------
{
  const out = path.join(art, 'logits-processor-trace')
  const instances = propsOf('LogitsProcessorTrace')
  const pagePath = await renderFigurePage({
    component: 'LogitsProcessorTrace',
    out,
    label: 'Logit processor stack',
    instances,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  const c = checker(out)
  c.check(
    'two embeds in the notes',
    instances.length === 2,
    instances.map(i => i.guided),
  )

  const states = await page.evaluate(() =>
    JSON.parse(document.querySelector('[data-logits-processor-trace]').dataset.lptStates),
  )
  const BEFORE = 15
  const AFTER = 3
  for (const g of ['0', '1', '2', '4']) {
    const steps = states[g]
    const rows = steps.filter(s => s.who === 'LogitsProcessor → JSONLogitsProcessor')
    const forward = steps.findIndex(s => s.code.startsWith('model_executable'))
    const sample = steps.findIndex(s => s.code.startsWith('model.sample'))
    const rowIdx = steps.map((s, i) => (rows.includes(s) ? i : -1)).filter(i => i >= 0)
    c.check(
      `guided ${g}: ${BEFORE + AFTER} fixed calls plus one per guided row`,
      steps.length === BEFORE + AFTER + Number(g) && rows.length === Number(g),
      steps.length,
    )
    c.check(
      `guided ${g}: per-row calls sit after the logits and before sample`,
      rowIdx.every(i => i > forward + 3 && i < sample),
      { forward, sample, rowIdx },
    )
    c.check(
      `guided ${g}: sampler follows sample`,
      steps[sample + 1].who === 'LlamaForCausalLM → Sampler',
      steps[sample + 1],
    )
  }

  if (VLLM_SRC && existsSync(VLLM_SRC)) {
    const all = [...states['4']]
    const bad = []
    for (const s of all) {
      const [, file, a, b] = s.src.match(/^vllm\/(.+):(\d+)(?:-(\d+))?$/)
      const lines = readFileSync(path.join(VLLM_SRC, 'vllm', file), 'utf8').split('\n')
      const span = lines.slice(Number(a) - 1, Number(b ?? a)).join('\n')
      // The called name: the identifier before the first parenthesis.
      const name = s.code
        .replace(/^logits_row = /, '')
        .split('(')[0]
        .split('.')
        .pop()
      if (!span.includes(name)) bad.push({ src: s.src, name, span })
    }
    c.check('every call names code on its cited vLLM v0.5.0 line', bad.length === 0, bad)
  } else c.check('vLLM source check skipped (set VLLM_SRC)', true, VLLM_SRC ?? null)

  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-logits-processor-trace]')[i]
      const visible = el => el.getBoundingClientRect().width > 0
      const set = [...r.querySelectorAll('[data-lpt-set]')].filter(visible)
      const calls = set.flatMap(s => [...s.querySelectorAll('[data-lpt-call]')])
      const on = calls.filter(c => c.dataset.on === 'true').map(c => Number(c.dataset.lptCall))
      const lanes = [...r.querySelectorAll('.lpt-track')].map(l => Number(l.getAttribute('y1')))
      const ends = calls.every(c => {
        const wire = c.querySelector('.lpt-wire')
        if (!wire)
          return lanes.some(
            y => Math.abs(Number(c.querySelector('rect').getAttribute('y')) + 5 - y) < 0.01,
          )
        const [y1, y2] = ['y1', 'y2'].map(a => Number(wire.getAttribute(a)))
        const rects = [...c.querySelectorAll('rect')].map(
          rc => Number(rc.getAttribute('y')) + Number(rc.getAttribute('height')) / 2,
        )
        return lanes.includes(y1) && lanes.includes(y2) && rects.includes(y1) && rects.includes(y2)
      })
      const range = r.querySelector('[data-lpt-step]')
      const outs = Object.fromEntries(
        [...r.querySelectorAll('[data-lpt-out]')].map(d => [d.dataset.lptOut, d.textContent]),
      )
      const litFill = calls[on[0]]
        ? getComputedStyle(calls[on[0]].querySelector('rect')).fill
        : null
      return {
        sets: set.length,
        calls: calls.length,
        on,
        ends,
        value: Number(range.value),
        max: Number(range.max),
        valuetext: range.getAttribute('aria-valuetext'),
        outs,
        litFill,
        guided: r.querySelector('[data-lpt-guided]:checked').value,
      }
    }, i)

  const root = page.locator('[data-logits-processor-trace]').first()
  const range = root.locator('[data-lpt-step]')
  for (const g of ['0', '1', '2', '4']) {
    await root.locator(`input[value="${g}"]`).check({ force: true })
    const steps = states[g]
    for (let k = 1; k <= steps.length; k++) {
      await range.fill(String(k))
      const s = await snap(0)
      const e = steps[k - 1]
      const ok =
        s.sets === 1 &&
        s.calls === steps.length &&
        s.max === steps.length &&
        s.value === k &&
        s.on.length === 1 &&
        s.on[0] === k - 1 &&
        s.ends &&
        s.outs.step === e.step &&
        s.outs.code === e.code &&
        s.outs.src === e.src &&
        s.outs.who === e.who &&
        s.litFill === 'rgb(253, 178, 162)' &&
        s.valuetext === `call ${e.step}: ${e.code}`
      if (!ok || k === 1 || k === steps.length)
        c.check(`guided ${g}, call ${k}: lit call, lines and readout agree`, ok, s)
    }
  }
  // Fewer guided rows clamp the call index.
  await root.locator('input[value="4"]').check({ force: true })
  await range.fill('22')
  await root.locator('input[value="0"]').check({ force: true })
  const clamp = await snap(0)
  c.check(
    'switching to fewer calls clamps the slider',
    clamp.value === 18 && clamp.max === 18 && clamp.on[0] === 17,
    clamp,
  )

  // Keyboard on the slider and the radio group.
  await root.locator('input[value="2"]').check({ force: true })
  await range.focus()
  await page.keyboard.press('Home')
  const h = await snap(0)
  await page.keyboard.press('ArrowRight')
  const r1 = await snap(0)
  await page.keyboard.press('End')
  const e1 = await snap(0)
  c.check(
    'Home, ArrowRight and End walk the calls',
    h.value === 1 && r1.value === 2 && e1.value === 20 && e1.on[0] === 19,
    [h.value, r1.value, e1.value],
  )
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement).outlineStyle)
  c.check('focused slider shows an outline', outline === 'solid', outline)
  await root.locator('input[value="2"]').focus()
  await page.keyboard.press('ArrowRight')
  const k4 = await snap(0)
  c.check(
    'ArrowRight on the radios selects 4 guided rows',
    k4.guided === '4' && k4.calls === 22,
    k4.guided,
  )
  const s2 = await snap(1)
  c.check('second instance is independent', s2.guided === '2' && s2.calls === 20, s2.guided)

  // Leave the first instance at its note state for the shots.
  await root.locator('input[value="2"]').check({ force: true })
  await range.fill('16')
  await c.shoot(page.locator('.zoomable').first(), 'row-call.png')
  await standardChecks(page, errors, c, '.zoomable')
  await browser.close()
  passed = (await c.finish()) && passed
}

process.exit(passed ? 0 : 1)
