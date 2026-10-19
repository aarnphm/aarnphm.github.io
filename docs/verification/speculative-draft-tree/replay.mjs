// Renders SpeculativeDraftTree twice inside Zoomable, drives every drafter and step by click and
// keyboard, and checks the drawn tree, mask, output strip, test log and readout against each other.
// Run: node docs/verification/speculative-draft-tree/replay.mjs
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(here, 'artifacts')
const METHODS = ['small', 'medusa', 'eagle', 'eagle3']
const STEPS = ['d1', 'd2', 'd3', 'verify', 'accept']
const U = 0.7

// Instance one carries the note's own caption, so the rendered caption is checked as well.
const note = await readFile(
  path.join(here, '../../../content/thoughts/Speculative decoding.md'),
  'utf8',
)
const caption = note.match(/<SpeculativeDraftTree[^>]*?caption="([^"]+)"/)[1]
const pagePath = await renderFigurePage({
  component: 'SpeculativeDraftTree',
  label: 'Speculative draft tree',
  out,
  instances: [
    { method: 'eagle', step: 'd1', caption },
    { method: 'medusa', step: 'verify', caption: 'Instance two.' },
  ],
})
const { browser, page, errors } = await openFigurePage(pagePath)
const c = checker(out)
const ROOT = '[data-speculative-draft-tree]'

// Reads the drawn state of one figure: nodes with depth and parent from geometry, mask cells by
// row, output strip, readout and the visible test log.
const snapshot = idx =>
  page.evaluate(
    ([i, sel]) => {
      const root = document.querySelectorAll(sel)[i]
      const panel = [...root.querySelectorAll('.sdt-panels')].find(p => !p.hidden)
      const nodes = [...panel.querySelectorAll('g.sdt-node')].map(g => {
        const [x, y] = g
          .getAttribute('transform')
          .match(/translate\(([-\d.]+) ([-\d.]+)\)/)
          .slice(1)
          .map(Number)
        return {
          id: Number(g.dataset.sdtNode),
          token: g.querySelector('.sdt-node-token').textContent,
          q: g.querySelector('.sdt-node-q').textContent,
          state: g.getAttribute('data-state'),
          x,
          y,
          depth: Math.round((x - 1) / 72),
        }
      })
      for (const e of panel.querySelectorAll('path.sdt-edge')) {
        const [x1, y1] = e
          .getAttribute('d')
          .match(/M ([-\d.]+) ([-\d.]+)/)
          .slice(1)
          .map(Number)
        const child = nodes.find(n => n.id === Number(e.dataset.sdtNode))
        child.parent = nodes.find(
          n => Math.abs(n.x + 54 - x1) < 0.01 && Math.abs(n.y + 14 - y1) < 0.01,
        )?.id
        child.edgeState = e.getAttribute('data-state')
      }
      const maskSvg = panel.querySelector('.sdt-mask')
      const rowLabels = [...maskSvg.querySelectorAll('.sdt-mask-row')].map(t => t.textContent)
      const cells = [...maskSvg.querySelectorAll('.sdt-cell')].map(r =>
        r.getAttribute('data-state'),
      )
      const width = cells.length / rowLabels.length
      const rows = rowLabels.map((t, r) => ({
        token: t,
        cells: cells.slice(r * width, (r + 1) * width).join(''),
      }))
      const step = [...root.querySelectorAll('.sdt-step')].filter(s => !s.hidden)
      const strip = step[0]
        ? [...step[0].querySelectorAll('.sdt-col-token')].map(t => ({
            t: t.textContent,
            k: t.dataset.kind,
          }))
        : []
      const outv = Object.fromEntries(
        [...root.querySelectorAll('[data-sdt-out]')].map(d => [d.dataset.sdtOut, d.textContent]),
      )
      const beta = [...root.querySelectorAll('dd[data-sdt-m]')]
        .filter(d => !d.hidden)
        .map(d => d.textContent)
      const logs = [...root.querySelectorAll('.sdt-log-wrap')].filter(l => !l.hidden)
      const log = logs[0]
        ? [...logs[0].querySelectorAll('tbody tr')].map(tr =>
            [...tr.children].map(td => td.textContent),
          )
        : null
      return {
        method: root.dataset.sdtMethod,
        step: root.dataset.sdtStep,
        visibleSteps: step.map(s => s.dataset.sdtKey),
        visibleLogs: logs.length,
        nodes,
        rows,
        strip,
        out: outv,
        beta,
        log,
      }
    },
    [idx, ROOT],
  )

const pick = async (idx, method, step) => {
  const root = page.locator(ROOT).nth(idx)
  await root.locator(`label:has(input[data-sdt-method-input][value="${method}"])`).click()
  await root.locator(`label:has(input[data-sdt-step-input][value="${step}"])`).click()
}

const ancestors = (nodes, n) => {
  const out = []
  for (let k = n; k; k = nodes.find(x => x.id === k.parent)) out.push(k)
  return out
}

const results = {}
for (const m of METHODS) {
  for (const s of STEPS) {
    await pick(0, m, s)
    const v = await snapshot(0)
    const key = `${m}:${s}`
    results[key] = {
      out: v.out,
      beta: v.beta,
      log: v.log,
      strip: v.strip.map(x => `${x.t}/${x.k}`).join(' '),
    }
    c.check(`${key} root attributes follow the radios`, v.method === m && v.step === s, [
      v.method,
      v.step,
    ])
    c.check(
      `${key} exactly one rule and strip visible`,
      v.visibleSteps.length === 1 && v.visibleSteps[0] === key,
      v.visibleSteps,
    )
    c.check(
      `${key} log visible only at accept`,
      v.visibleLogs === (s === 'accept' ? 1 : 0),
      v.visibleLogs,
    )

    const drafted = v.nodes.filter(n => n.id !== 0)
    const shown = drafted.filter(n => n.state !== 'hidden')
    c.check(`${key} drawn draft nodes equal readout`, String(shown.length) === v.out.drafted, [
      shown.length,
      v.out.drafted,
    ])
    const depth = { d1: 1, d2: 2, d3: 3 }[s] ?? 3
    c.check(
      `${key} lit nodes are the newest depth while drafting`,
      s.startsWith('d')
        ? drafted.every(
            n =>
              (n.state === 'new') === (n.depth === depth) &&
              (n.state === 'hidden') === n.depth > depth,
          )
        : !drafted.some(n => n.state === 'new'),
      drafted.map(n => `${n.token}:${n.depth}:${n.state}`),
    )
    const wantDraft = m === 'medusa' ? 0 : depth
    c.check(`${key} draft passes`, v.out.draft === String(wantDraft), v.out.draft)
    c.check(`${key} target passes`, v.out.target === (s.startsWith('d') ? '0' : '1'), v.out.target)
    c.check(
      `${key} edges share their child's state`,
      drafted.every(n => n.edgeState === n.state),
      null,
    )

    // Tree mask semantics from geometry: a row sees the prefix, its ancestors and itself.
    const order = v.nodes.slice().sort((a, b) => a.depth - b.depth || a.id - b.id)
    c.check(
      `${key} mask rows follow draft order`,
      v.rows.map(r => r.token).join() === order.map(n => n.token).join(),
      null,
    )
    let maskOk = true
    v.rows.forEach((row, r) => {
      const n = order[r]
      const live = n.state !== 'hidden'
      const anc = new Set(
        ancestors(v.nodes, n)
          .map(a => a.id)
          .concat(0),
      )
      const want = order.map(col => (!live || !anc.has(col.id) ? '0' : col.id === 0 ? '1' : null))
      const prefix = row.cells.slice(0, 2)
      if (prefix !== (live ? '11' : '00')) maskOk = false
      row.cells
        .slice(2)
        .split('')
        .forEach((ch, j) => {
          if (want[j] === '0' && ch !== '0') maskOk = false
          if (want[j] === '1' && ch !== '1') maskOk = false
          if (want[j] === null) {
            const lit = s === 'verify' || (s === 'accept' && n.state === 'accepted')
            if (ch !== (lit ? '2' : '1')) maskOk = false
          }
        })
    })
    c.check(`${key} mask equals ancestor rule and lit rows`, maskOk, v.rows)

    if (s === 'accept') {
      const acc = drafted.filter(n => n.state === 'accepted').sort((a, b) => a.depth - b.depth)
      c.check(
        `${key} accepted nodes form a root path`,
        acc.every((n, i) => n.depth === i + 1 && n.parent === (i ? acc[i - 1].id : 0)),
        acc.map(n => n.token),
      )
      c.check(`${key} accepted readout`, v.out.accepted === `${acc.length} of 3`, v.out.accepted)
      const emitted = v.strip.slice(3)
      c.check(
        `${key} output strip is prefix, accepted path, then one target token`,
        v.strip
          .slice(0, 3)
          .map(x => x.t)
          .join(' ') === 'How can I' &&
          emitted
            .slice(0, -1)
            .map(x => x.t)
            .join(' ') === acc.map(n => n.token).join(' ') &&
          emitted.slice(0, -1).every(x => x.k === 'lit') &&
          emitted.at(-1).k === 'target',
        v.strip,
      )
      c.check(`${key} emitted readout`, v.out.emitted === String(emitted.length), v.out.emitted)
      c.check(
        `${key} plain decoding needs one pass per emitted token`,
        v.out.plain === String(emitted.length),
        v.out.plain,
      )
      // Recompute the acceptance rule from the logged p and q.
      const tests = v.log.slice(0, -1)
      let ruleOk = true
      for (const [, p, q, a, res] of tests) {
        const pv = Number.parseFloat(p)
        const want = Math.min(1, pv / Number(q))
        if (Math.abs(want - Number(a)) > 0.015) ruleOk = false
        if ((res === 'keep') !== U <= Number(a)) ruleOk = false
      }
      c.check(`${key} log follows min(1, p/q) with U = ${U}`, ruleOk, tests)
      c.check(
        `${key} kept tests equal accepted nodes`,
        tests.filter(t => t[4] === 'keep').length === acc.length,
        null,
      )
      c.check(
        `${key} last log row names the emitted token`,
        v.log.at(-1).at(-1) === emitted.at(-1).t,
        v.log.at(-1),
      )
      const rejected = drafted.filter(n => n.state === 'rejected').map(n => n.token)
      c.check(
        `${key} rejected nodes are the rejected tests`,
        rejected.sort().join() ===
          tests
            .filter(t => t[4] === 'reject')
            .map(t => t[0])
            .sort()
            .join(),
        rejected,
      )
    }

    if (s === 'd2' && m === 'medusa') {
      const kids = p =>
        drafted
          .filter(n => n.parent === p.id)
          .map(n => `${n.token} ${n.q}`)
          .join()
      const d1 = drafted.filter(n => n.depth === 1)
      c.check(
        'Medusa gives every depth-1 node the same children and q',
        kids(d1[0]) === kids(d1[1]),
        [kids(d1[0]), kids(d1[1])],
      )
    }
    if (s === 'd2' && (m === 'eagle' || m === 'eagle3')) {
      const kids = p =>
        drafted
          .filter(n => n.parent === p.id)
          .map(n => n.token)
          .join()
      const d1 = drafted.filter(n => n.depth === 1)
      c.check(`${m} branches get different children`, kids(d1[0]) !== kids(d1[1]), [
        kids(d1[0]),
        kids(d1[1]),
      ])
    }
    if (m === 'small')
      c.check(
        `${key} small LM drafts a chain`,
        drafted.length === 3 && drafted.every((n, i) => n.depth === i + 1),
        null,
      )
  }
}

// Screenshots of representative states.
const root0 = page.locator(ROOT).nth(0)
const zoom0 = page.locator('.zoomable').nth(0)
for (const [m, s] of [
  ['eagle', 'd2'],
  ['eagle', 'accept'],
  ['medusa', 'd2'],
  ['medusa', 'accept'],
  ['small', 'verify'],
  ['small', 'accept'],
  ['eagle3', 'd2'],
  ['eagle3', 'accept'],
]) {
  await pick(0, m, s)
  // Let the 180ms state transitions settle before the screenshot.
  await page.waitForTimeout(300)
  await c.shoot(zoom0, `state-${m}-${s}.png`)
}

const second = await snapshot(1)
c.check(
  'second instance keeps its own props',
  second.method === 'medusa' && second.step === 'verify',
  [second.method, second.step],
)

// Keyboard: arrow keys move inside each native radio group.
await pick(0, 'small', 'd1')
await root0.locator('input[data-sdt-method-input][value="small"]').focus()
await page.keyboard.press('ArrowRight')
let k = await snapshot(0)
c.check('ArrowRight selects the next drafter', k.method === 'medusa', k.method)
const outline = await page.evaluate(
  () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
)
c.check('focused segment shows an outline', outline === 'solid', outline)
await page.keyboard.press('Tab')
await page.keyboard.press('ArrowRight')
await page.keyboard.press('ArrowRight')
k = await snapshot(0)
await page.waitForTimeout(300)
c.check('Tab then ArrowRight moves through steps', k.step === 'd3' && k.method === 'medusa', [
  k.method,
  k.step,
])
await c.shoot(zoom0, 'keyboard-focus.png')

await pick(0, 'eagle', 'accept')
await page.waitForTimeout(300)
await standardChecks(page, errors, c, '.zoomable')

await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
await pick(0, 'small', 'd1')
const after = await snapshot(0)
c.check(
  'cleanup removes the change listener',
  after.method === 'eagle' && after.step === 'accept',
  [after.method, after.step],
)

c.results.states = results
await browser.close()
process.exit((await c.finish()) ? 0 : 1)
