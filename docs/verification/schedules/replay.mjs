// Replays ScheduleClassifier and PriorityProtocolTimeline with the props the notes embed. The
// props come from the notes' jsx fences, parsed with remark-mdx as the Codeblock transformer does.
// Writes pages, screenshots and results.json under ./artifacts.
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import remarkMdx from 'remark-mdx'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import {
  checker,
  openFigurePage,
  renderFigurePage,
  repo,
  standardChecks,
} from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(here, 'artifacts')
const notes = 'content/thoughts/university/twenty-four-twenty-five'

const propsFromNote = async (file, component) => {
  const md = await readFile(path.join(repo, notes, file), 'utf8')
  const fences = [...md.matchAll(/```jsx imports=\{[^}]*\}\n([\s\S]*?)```/g)].map(m => m[1])
  const found = []
  for (const code of fences) {
    const tree = unified().use(remarkParse).use(remarkMdx).parse(code)
    visit(tree, n => {
      if (n.name !== component) return
      found.push(Object.fromEntries(n.attributes.map(a => [a.name, a.value])))
    })
  }
  return found
}

// --- ScheduleClassifier -----------------------------------------------------------------------

// Regions derived by hand from the note's definitions, independent of the component's classifier.
const EXPECTED_REGION = {
  serial: 'serial',
  strict: 'strict-csr',
  'lost-update': 'strict-ncsr',
  'dirty-write': 'aca-csr',
  'dirty-write-cycle': 'aca-ncsr',
  serialisable: 'rec-csr',
  'dirty-read-cycle': 'rec-ncsr',
  unrecoverable: 'all-csr',
  'unrecoverable-cycle': 'all-ncsr',
}

const sclProps = await propsFromNote('sfwr-3db3/Transaction.md', 'ScheduleClassifier')
const sclOut = path.join(out, 'schedule-classifier')
const sclPage = await renderFigurePage({
  component: 'ScheduleClassifier',
  instances: sclProps,
  out: sclOut,
  label: 'schedule classifier',
})
const s = await openFigurePage(sclPage)
const c1 = checker(sclOut)
c1.check(
  'note embeds four ScheduleClassifier instances',
  sclProps.length === 4,
  sclProps.map(p => p.preset),
)

const sclSnap = (pg, i) =>
  pg.evaluate(i => {
    const root = document.querySelectorAll('[data-schedule-classifier]')[i]
    const region = root.dataset.sclRegion
    const visible = [...root.querySelectorAll('[data-scl-panel]')]
      .filter(p => !p.hidden && getComputedStyle(p).display !== 'none')
      .map(p => p.dataset.sclPanel)
    const cells = [...root.querySelectorAll('.scl-cell')]
    const filled = cells
      .filter(
        c =>
          getComputedStyle(c).fill !== 'transparent' &&
          getComputedStyle(c).fill !== 'rgba(0, 0, 0, 0)',
      )
      .map(c => c.dataset.region)
    const panel = [...root.querySelectorAll('.scl-readout-panel')].find(p => !p.hidden)
    const out = Object.fromEntries(
      [...panel.querySelectorAll('[data-scl-out]')].map(d => [
        d.dataset.sclOut,
        d.textContent.trim(),
      ]),
    )
    const regionValue = panel.querySelector('[data-scl-region-value]').dataset.sclRegionValue
    const table = [...root.querySelectorAll('.scl-timeline')].find(t => !t.hidden)
    const witness = table.querySelectorAll('[data-scl-witness="true"]').length
    const ops = table.querySelectorAll('tbody tr').length
    const graph = [...root.querySelectorAll('.scl-graph-panel')].find(p => !p.hidden)
    const edges = graph.querySelectorAll('.scl-edge').length
    const cycleEdges = graph.querySelectorAll('.scl-edge--cycle').length
    return {
      preset: root.dataset.sclPreset,
      region,
      visible,
      filled,
      out,
      regionValue,
      witness,
      ops,
      edges,
      cycleEdges,
      inputs: root.querySelectorAll('[data-scl-input]').length,
    }
  }, i)

const sclRoot = s.page.locator('[data-schedule-classifier]').nth(0)
const seenRegions = {}
const sclStates = {}
for (const [preset, region] of Object.entries(EXPECTED_REGION)) {
  await sclRoot.locator(`label:has(input[value="${preset}"])`).click()
  // Let the 180ms fill transition settle before reading computed fills.
  await s.page.waitForTimeout(260)
  const v = await sclSnap(s.page, 0)
  sclStates[preset] = v
  seenRegions[v.region] = (seenRegions[v.region] ?? 0) + 1
  c1.check(`${preset}: region is ${region}`, v.region === region, v.region)
  c1.check(`${preset}: readout region equals lit region`, v.regionValue === v.region, v.regionValue)
  c1.check(
    `${preset}: lit Venn cells are exactly the region's cells`,
    v.filled.length > 0 && v.filled.every(r => r === v.region),
    [...new Set(v.filled)],
  )
  c1.check(
    `${preset}: only its own panels show`,
    v.visible.length === 4 && v.visible.every(p => p === preset),
    v.visible,
  )
  const [level, csr] = v.region === 'serial' ? ['serial', 'csr'] : v.region.split('-')
  const rank = { all: 0, rec: 1, aca: 2, strict: 3, serial: 4 }[level]
  c1.check(
    `${preset}: yes/no rows agree with the region`,
    v.out.recoverable === (rank >= 1 ? 'yes' : 'no') &&
      v.out.aca === (rank >= 2 ? 'yes' : 'no') &&
      v.out.strict === (rank >= 3 ? 'yes' : 'no') &&
      v.out.serial === (rank >= 4 ? 'yes' : 'no') &&
      v.out.csr.startsWith(csr === 'csr' ? 'yes' : 'no'),
    v.out,
  )
  c1.check(
    `${preset}: a cycle draws both edges heavy, an acyclic graph none`,
    csr === 'csr' ? v.cycleEdges === 0 : v.edges === 2 && v.cycleEdges === 2,
    { edges: v.edges, cycle: v.cycleEdges },
  )
  c1.check(
    `${preset}: witness cells mark the failure`,
    preset === 'serial' ? v.witness === 0 : v.witness > 0,
    v.witness,
  )
}
c1.check(
  'every Venn region holds exactly one preset',
  Object.keys(seenRegions).length === 9 && Object.values(seenRegions).every(n => n === 1),
  seenRegions,
)
c1.check(
  'slide tables keep their operation counts',
  sclStates.serial.ops === 10 &&
    sclStates.serialisable.ops === 10 &&
    sclStates.unrecoverable.ops === 8,
  {
    serial: sclStates.serial.ops,
    serialisable: sclStates.serialisable.ops,
    unrecoverable: sclStates.unrecoverable.ops,
  },
)

for (const [i, preset] of [
  [1, 'serial'],
  [2, 'serialisable'],
  [3, 'unrecoverable'],
]) {
  const v = await sclSnap(s.page, i)
  c1.check(
    `single view ${i} (${preset}): no picker, region ${EXPECTED_REGION[preset]}`,
    v.inputs === 0 && v.preset === preset && v.region === EXPECTED_REGION[preset],
    v,
  )
}

await sclRoot.locator('input[value="strict"]').check({ force: true })
await sclRoot.locator('input[value="strict"]').focus()
await s.page.keyboard.press('ArrowRight')
const afterKey = await sclSnap(s.page, 0)
c1.check('ArrowRight moves to the next preset', afterKey.preset === 'lost-update', afterKey.preset)
const outline = await s.page.evaluate(
  () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
)
c1.check('focused segment shows a solid outline', outline === 'solid', outline)
await c1.shoot(sclRoot, 'keyboard-focus.png')
for (const preset of ['unrecoverable-cycle', 'dirty-write']) {
  await sclRoot.locator(`input[value="${preset}"]`).check({ force: true })
  await c1.shoot(sclRoot, `light-${preset}.png`)
}
await sclRoot.locator('input[value="serialisable"]').check({ force: true })
await standardChecks(s.page, s.errors, c1, '.zoomable')
await s.page.evaluate(() => window.__cleanups.forEach(fn => fn()))
await sclRoot.locator('input[value="serial"]').check({ force: true })
const afterCleanup = await sclSnap(s.page, 0)
c1.check(
  'cleanup removes the change listener',
  afterCleanup.preset === 'serialisable',
  afterCleanup.preset,
)
await s.browser.close()
const ok1 = await c1.finish()

// --- PriorityProtocolTimeline -----------------------------------------------------------------

// Hand-traced schedules (see README): blocking units and completion time per task, or null for a
// deadlocked task. Total run units per task come from the slide segments.
const RUN = {
  pip: { T1: 9, T2: 3, T3: 3, T4: 12 },
  pcp: { C: 8, B: 11, A: 15 },
  chained: { TH: 7, TM: 6, TL: 6 },
}
const pipFair = { T1: [4, 18], T2: [3, 21], T3: [2, 24], T4: [0, 27] }
const pcpDead = { C: [0, 17], B: null, A: null }
const EXPECTED = {
  'pip:none': {
    tasks: { T1: [10, 24], T2: [0, 12], T3: [0, 15], T4: [0, 27] },
    deadlock: null,
    chained: false,
  },
  'pip:pip': { tasks: pipFair, deadlock: null, chained: false },
  'pip:pcp': { tasks: pipFair, deadlock: null, chained: false },
  'pcp:none': { tasks: pcpDead, deadlock: 20, chained: false },
  'pcp:pip': { tasks: pcpDead, deadlock: 20, chained: false },
  'pcp:pcp': { tasks: { C: [0, 17], B: [11, 34], A: [0, 25] }, deadlock: null, chained: false },
  'chained:none': {
    tasks: { TH: [7, 18], TM: [0, 9], TL: [0, 19] },
    deadlock: null,
    chained: true,
  },
  'chained:pip': {
    tasks: { TH: [6, 17], TM: [3, 18], TL: [0, 19] },
    deadlock: null,
    chained: true,
  },
  'chained:pcp': {
    tasks: { TH: [2, 13], TM: [3, 18], TL: [0, 19] },
    deadlock: null,
    chained: false,
  },
}

const pptProps = await propsFromNote('sfwr-4aa4/PIP and PCP.md', 'PriorityProtocolTimeline')
const pptOut = path.join(out, 'priority-protocol-timeline')
const pptPage = await renderFigurePage({
  component: 'PriorityProtocolTimeline',
  instances: pptProps,
  out: pptOut,
  label: 'priority protocol timeline',
})
const t = await openFigurePage(pptPage)
const c2 = checker(pptOut)
c2.check(
  'note embeds the PIP and PCP instances',
  pptProps.length === 2 && pptProps[0].protocol === 'pip' && pptProps[1].protocol === 'pcp',
  pptProps.map(p => `${p.preset}:${p.protocol}`),
)

const pptSnap = (pg, i) =>
  pg.evaluate(i => {
    const root = document.querySelectorAll('[data-priority-protocol-timeline]')[i]
    const panels = [...root.querySelectorAll('[data-ppt-panel]')].filter(p => !p.hidden)
    const panel = panels[0]
    const rows = Object.fromEntries(
      [...panel.querySelectorAll('tr[data-ppt-task]')].map(r => [
        r.dataset.pptTask,
        {
          blocking: r.querySelector('[data-ppt-blocking]').textContent.trim(),
          finish: r.querySelector('[data-ppt-finish]').textContent.trim(),
        },
      ]),
    )
    const segs = [...panel.querySelectorAll('[data-ppt-seg]')].map(e => {
      const [task, t0, t1, kind] = e.dataset.pptSeg.split(':')
      return { task, t0: Number(t0), t1: Number(t1), kind }
    })
    const out = Object.fromEntries(
      [...panel.querySelectorAll('[data-ppt-out]')].map(d => [
        d.dataset.pptOut,
        d.textContent.trim(),
      ]),
    )
    const deadlockLine = panel.querySelectorAll('.ppt-deadlock').length
    const shownNote = [...root.querySelectorAll('[data-ppt-preset-panel]')]
      .filter(p => !p.hidden)
      .map(p => p.dataset.pptPresetPanel)
    const shownRule = [...root.querySelectorAll('[data-ppt-protocol-panel]')]
      .filter(p => !p.hidden)
      .map(p => p.dataset.pptProtocolPanel)
    return {
      key: root.dataset.pptKey,
      panels: panels.length,
      rows,
      segs,
      out,
      deadlockLine,
      shownNote,
      shownRule,
    }
  }, i)

const pptRoot = t.page.locator('[data-priority-protocol-timeline]').nth(0)
const pptStates = {}
for (const [key, exp] of Object.entries(EXPECTED)) {
  const [preset, protocol] = key.split(':')
  await pptRoot.locator(`label:has(input[data-ppt-preset][value="${preset}"])`).click()
  await pptRoot.locator(`label:has(input[data-ppt-protocol][value="${protocol}"])`).click()
  const v = await pptSnap(t.page, 0)
  pptStates[key] = { rows: v.rows, out: v.out }
  c2.check(
    `${key}: one panel, its note and rule show`,
    v.key === key &&
      v.panels === 1 &&
      v.shownNote.join() === preset &&
      v.shownRule.join() === protocol,
    v,
  )
  for (const [task, want] of Object.entries(exp.tasks)) {
    const row = v.rows[task]
    const ok =
      want === null
        ? row.blocking === '∞' && row.finish === '–'
        : row.blocking === String(want[0]) && row.finish === String(want[1])
    c2.check(`${key}: ${task} blocking and completion`, ok, { row, want })
    const mine = v.segs.filter(sg => sg.task === task)
    const ran = mine.reduce((a, sg) => a + sg.t1 - sg.t0, 0)
    if (want !== null) {
      c2.check(
        `${key}: ${task} drawn bars sum to its run time and end at completion`,
        ran === RUN[preset][task] && Math.max(...mine.map(sg => sg.t1)) === want[1],
        { ran, end: Math.max(...mine.map(sg => sg.t1)) },
      )
    } else {
      c2.check(
        `${key}: ${task} drawn bars stop before the deadlock`,
        mine.every(sg => sg.t1 <= exp.deadlock),
        mine,
      )
    }
  }
  c2.check(
    `${key}: deadlock readout and rule agree`,
    exp.deadlock === null
      ? v.out.deadlock === 'no' && v.deadlockLine === 0
      : v.out.deadlock.includes(`t=${exp.deadlock}`) && v.deadlockLine === 1,
    { out: v.out.deadlock, line: v.deadlockLine },
  )
  c2.check(
    `${key}: chained blocking readout`,
    v.out.chained.startsWith(exp.chained ? 'yes' : 'no'),
    v.out.chained,
  )
  const inherited = v.segs.some(sg => sg.kind === 'inherit')
  c2.check(
    `${key}: salmon inherited bars only under a protocol`,
    inherited === (protocol !== 'none'),
    inherited,
  )
}

const second = await pptSnap(t.page, 1)
c2.check('second instance keeps PCP on the PCP slide', second.key === 'pcp:pcp', second.key)

await pptRoot.locator('input[data-ppt-preset][value="pip"]').check({ force: true })
await pptRoot.locator('input[data-ppt-protocol][value="none"]').check({ force: true })
await pptRoot.locator('input[data-ppt-protocol][value="none"]').focus()
await t.page.keyboard.press('ArrowRight')
const k = await pptSnap(t.page, 0)
c2.check('ArrowRight moves the protocol from none to PIP', k.key === 'pip:pip', k.key)
const outline2 = await t.page.evaluate(
  () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
)
c2.check('focused segment shows a solid outline', outline2 === 'solid', outline2)
await c2.shoot(pptRoot, 'keyboard-focus.png')
for (const key of ['pip:none', 'pcp:pip', 'chained:pip', 'chained:pcp']) {
  const [preset, protocol] = key.split(':')
  await pptRoot.locator(`input[data-ppt-preset][value="${preset}"]`).check({ force: true })
  await pptRoot.locator(`input[data-ppt-protocol][value="${protocol}"]`).check({ force: true })
  await c2.shoot(pptRoot, `light-${key.replace(':', '-')}.png`)
}
await pptRoot.locator('input[data-ppt-preset][value="pip"]').check({ force: true })
await pptRoot.locator('input[data-ppt-protocol][value="pip"]').check({ force: true })
await standardChecks(t.page, t.errors, c2, '.zoomable')
await t.page.evaluate(() => window.__cleanups.forEach(fn => fn()))
await pptRoot.locator('input[data-ppt-protocol][value="pcp"]').check({ force: true })
const afterCleanup2 = await pptSnap(t.page, 0)
c2.check('cleanup removes the change listener', afterCleanup2.key === 'pip:pip', afterCleanup2.key)
c2.results.states = pptStates
await t.browser.close()
const ok2 = await c2.finish()

process.exit(ok1 && ok2 ? 0 : 1)
