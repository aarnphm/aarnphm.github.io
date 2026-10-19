// Renders GraphTraversal, BlockDiagram and PatriciaTrie through the shared figure harness, drives
// every control by click and keyboard, and checks that each drawn state matches its readout and
// the formulas in the notes. Writes pages, screenshots and results.json under ./artifacts.
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const art = path.join(here, 'artifacts')
let passed = true

// standardChecks shoots 320px right after a theme switch, while colour transitions still run;
// these shots wait for them to settle.
const settled320 = async (page, c, selector) => {
  await page.setViewportSize({ width: 320, height: 1400 })
  await page.waitForTimeout(400)
  const roots = page.locator(selector)
  for (let i = 0; i < (await roots.count()); i++)
    await c.shoot(roots.nth(i), `320-settled-${i}.png`)
  await page.setViewportSize({ width: 900, height: 1400 })
}

// ---------- GraphTraversal ----------
{
  const out = path.join(art, 'graph-traversal')
  const page0 = await renderFigurePage({
    component: 'GraphTraversal',
    out,
    label: 'DFS and BFS on the course example graph',
    instances: [
      { mode: 'dfs', start: 0, caption: 'Instance one.' },
      { mode: 'bfs', start: 1, caption: 'Instance two.' },
    ],
  })
  const { browser, page, errors } = await openFigurePage(page0)
  const c = checker(out)
  const root = i => page.locator('[data-graph-traversal]').nth(i)
  const snap = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-graph-traversal]')[i]
      const nodes = [...r.querySelectorAll('[data-gtr-node]')].map(n => n.dataset.state).join('')
      const edges = [...r.querySelectorAll('[data-gtr-edge]')].map(n => n.dataset.state).join('')
      const slots = [...r.querySelectorAll('[data-gtr-slot]')].map(s => s.textContent)
      const cells = Object.fromEntries(
        [...r.querySelectorAll('[data-gtr-cell]')].map(td => [td.dataset.gtrCell, td.textContent]),
      )
      const rows = [...r.querySelectorAll('[data-gtr-row-label]')].map(t => t.textContent)
      const range = r.querySelector('[data-gtr-range]')
      return {
        nodes,
        edges,
        stack: slots.filter(Boolean),
        cells,
        rows,
        step: Number(range.value),
        max: Number(range.max),
        say: r.querySelector('[data-gtr-say]').textContent,
        count: r.querySelector('[data-gtr-count]').textContent,
        reached: r.querySelector('[data-gtr-out="reached"]').textContent,
        mode: r.dataset.gtrMode,
      }
    }, i)
  const table = (s, r) => Array.from({ length: 10 }, (_, i) => s.cells[`${r}:${i}`])

  const s0 = await snap(0)
  c.check('SSR DFS from n0 starts unvisited', s0.nodes === 'uuuuuuuuuu' && s0.step === 0, s0)
  c.check('SSR second instance is BFS from n1', (await snap(1)).mode === 'bfs')

  // Step DFS from n0 with the next button and check invariants at every step.
  const r0 = root(0)
  let ok = true
  const bad = []
  for (let k = 1; k <= s0.max; k++) {
    await r0.locator('[data-gtr-go="next"]').click()
    const s = await snap(0)
    const open = [...s.nodes].filter(x => x === 'o' || x === 'c').length
    const cur = s.nodes.indexOf('c')
    const top = s.stack.at(-1)
    const finished = [...s.nodes].filter(x => x === 'f').length
    const fShown = table(s, 1).filter(x => x !== '–').length
    const dShown = table(s, 0).filter(x => x !== '–').length
    const marked = [...s.nodes].filter(x => x !== 'u').length
    const good =
      s.step === k &&
      s.count === `${k} / ${s0.max}` &&
      // The open vertices are exactly the call stack.
      (open === s.stack.length || (cur >= 0 && open === s.stack.length)) &&
      (cur < 0 || top === `n${cur}`) &&
      fShown === finished &&
      dShown === marked
    if (!good) {
      ok = false
      bad.push({ k, s })
    }
  }
  c.check('DFS n0: every step keeps stack = open vertices, top = salmon, times = states', ok, bad)
  const end = await snap(0)
  const expectD = ['1', '–', '3', '2', '8', '11', '12', '4', '–', '10']
  const expectF = ['16', '–', '6', '7', '9', '14', '13', '5', '–', '15']
  c.check(
    'DFS n0 discovery times (hand trace, ascending Adj)',
    JSON.stringify(table(end, 0)) === JSON.stringify(expectD),
    table(end, 0),
  )
  c.check(
    'DFS n0 finish times',
    JSON.stringify(table(end, 1)) === JSON.stringify(expectF),
    table(end, 1),
  )
  c.check(
    'DFS n0 reaches 8 of 10, tree has 7 edges',
    end.reached === '8 / 10' && [...end.edges].filter(e => e === 't').length === 7,
    end,
  )
  c.check('DFS n0 total steps = 8 entries + 8 returns + 11 marked scans', s0.max === 27, s0.max)
  await c.shoot(r0, 'dfs-n0-final.png')

  // Back to the middle with the slider keyboard, then first.
  await r0.locator('[data-gtr-go="first"]').click()
  await r0.locator('[data-gtr-range]').focus()
  for (let k = 0; k < 9; k++) await page.keyboard.press('ArrowRight')
  const mid = await snap(0)
  c.check('ArrowRight on the step slider advances', mid.step === 9, mid.step)
  c.check(
    'Step 9 of DFS n0 is the return of n7 to n2',
    mid.say.includes('DFS-R(n7) returns') && mid.stack.join(',') === 'n0,n3,n2',
    mid,
  )
  await c.shoot(r0, 'dfs-n0-step9.png')
  await r0.locator('[data-gtr-go="prev"]').click()
  c.check('back button steps back', (await snap(0)).step === 8)

  // BFS from n0 with the mode radio (keyboard), then last.
  await r0.locator('input[data-gtr-mode-input][value="dfs"]').focus()
  await page.keyboard.press('ArrowRight')
  const b0 = await snap(0)
  c.check('ArrowRight on the mode group selects BFS and resets', b0.mode === 'bfs' && b0.step === 0)
  c.check(
    'BFS rows relabel',
    JSON.stringify(b0.rows) === JSON.stringify(['order', 'dist', 'parent']),
    b0.rows,
  )
  const outline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused segment shows a solid outline', outline === 'solid', outline)
  await r0.locator('[data-gtr-go="last"]').click()
  const b1 = await snap(0)
  c.check(
    'BFS n0 distances',
    JSON.stringify(table(b1, 1)) ===
      JSON.stringify(['0', '–', '2', '1', '1', '2', '2', '1', '–', '1']),
    table(b1, 1),
  )
  c.check(
    'BFS n0 parents',
    JSON.stringify(table(b1, 2)) ===
      JSON.stringify(['–', '–', 'n3', 'n0', 'n0', 'n9', 'n9', 'n0', '–', 'n0']),
    table(b1, 2),
  )
  c.check('BFS ends with an empty queue', b1.stack.length === 0 && b1.say.includes('empty'), b1)
  await c.shoot(r0, 'bfs-n0-final.png')

  // Middle of BFS: queue holds the next distance layer.
  await r0.locator('[data-gtr-go="first"]').click()
  for (let k = 0; k < 6; k++) await r0.locator('[data-gtr-go="next"]').click()
  const bq = await snap(0)
  c.check(
    'BFS n0 after scanning Adj(n0) and dequeuing n3: queue n4 n7 n9',
    bq.stack.join(',') === 'n4,n7,n9' && bq.nodes[3] === 'c',
    bq,
  )
  await c.shoot(r0, 'bfs-n0-step6.png')

  // Every source in both modes: final marked count equals readout and tree edges = reached - 1.
  const sweep = []
  for (const mode of ['dfs', 'bfs'])
    for (let s = 0; s < 10; s++) {
      await r0.locator(`input[data-gtr-mode-input][value="${mode}"]`).check({ force: true })
      await r0.locator(`input[data-gtr-start-input][value="${s}"]`).check({ force: true })
      await r0.locator('[data-gtr-go="last"]').click()
      const f = await snap(0)
      const marked = [...f.nodes].filter(x => x !== 'u').length
      const tree = [...f.edges].filter(x => x === 't').length
      const expect = s === 1 || s === 8 ? 2 : 8
      sweep.push({ mode, s, marked, tree, reached: f.reached })
      c.check(
        `${mode} from n${s}: ${expect} marked, ${expect - 1} tree edges, readout agrees`,
        marked === expect && tree === expect - 1 && f.reached === `${expect} / 10`,
        { marked, tree, reached: f.reached },
      )
    }
  c.results.sweep = sweep
  const second = await snap(1)
  c.check('Second instance kept its own state', second.mode === 'bfs' && second.step === 0)

  await standardChecks(page, errors, c, '.zoomable')
  await settled320(page, c, '.zoomable')
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  const before = (await snap(0)).step
  await r0.locator('[data-gtr-go="first"]').click()
  c.check('Cleanup removes listeners', (await snap(0)).step === before, before)
  await browser.close()
  passed = (await c.finish()) && passed
}

// ---------- BlockDiagram ----------
{
  const out = path.join(art, 'block-diagram')
  const configs = [
    'feedback',
    'summing-junction',
    'sampler-cascade',
    'open-loop',
    'sampled-feedback',
    'sensing-device',
    'digital-controller',
  ]
  const pagePath = await renderFigurePage({
    component: 'BlockDiagram',
    out,
    label: 'Block diagram reduction',
    instances: configs.map(config => ({ config, caption: `Config ${config}.` })),
  })
  const { browser, page, errors } = await openFigurePage(pagePath, { width: 900, height: 1400 })
  const c = checker(out)
  const norm = t => t.replace(/\s+/g, '')
  // Final results as the notes write them.
  const expected = {
    'feedback:0': '\\frac{H_1}{1+H_1H_2}',
    'feedback:1': '\\frac{H_1}{1-H_1H_2}',
    'summing-junction:0': 'G(s)R(s)\\mpG(s)X(s)',
    'summing-junction:1': 'G(s)R(s)\\mpX(s)',
    'sampler-cascade:0': 'G_2(z)G_1(z)',
    'sampler-cascade:1': '\\mathcal{P}_T\\{G_1(s)G_2(s)\\}',
    'open-loop:0': 'G(z)D(z)',
    'sampled-feedback:0': '\\frac{G(z)}{1+\\mathcal{P}_T\\{G(s)H(s)\\}}',
    'sensing-device:0': '\\frac{\\mathcal{P}_T\\{G(s)R(s)\\}}{1+\\mathcal{P}_T\\{G(s)H(s)\\}}',
    'digital-controller:0': '\\frac{G_1(z)G_2(z)}{1+G_1(z)\\mathcal{P}_T\\{G_2(s)H(s)\\}}',
  }
  // Intermediate relations the notes state.
  const relations = {
    'sampler-cascade:0': ['A(z)=G_1(z)E(z)', 'C(z)=G_2(z)A(z)', 'C(z)=G_2(z)G_1(z)E(z)'],
    'sampler-cascade:1': ['C(z)=\\mathcal{P}_T\\{G_1(s)G_2(s)\\}E(z)'],
    'open-loop:0': [
      'G(s)=\\frac{1-e^{-sT}}{s}G_p(s)',
      'C(z)=G(z)D(z)E(z)',
      'G(z)=(1-z^{-1})\\mathcal{P}_T\\left\\{\\frac{G_p(s)}{s}\\right\\}',
    ],
    'sampled-feedback:0': ['C(z)=G(z)E(z)', 'E(z)=R(z)-\\mathcal{P}_T\\{G(s)H(s)\\}E(z)'],
    'sensing-device:0': [
      'C(s)=G(s)R(s)-G(s)H(s)C^*(s)',
      'C(z)=\\mathcal{P}_T\\{G(s)R(s)\\}-\\mathcal{P}_T\\{G(s)H(s)\\}C(z)',
    ],
    'digital-controller:0': [
      'U(z)=G_1(z)E(z)',
      'C(z)=G_2(z)U(z)',
      'E(z)=R(z)-\\mathcal{P}_T\\{G_2(s)H(s)\\}U(z)',
    ],
  }
  const visible = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-block-diagram]')[i]
      const vs = [...r.querySelectorAll('[data-bd-variant]')].filter(v => !v.hidden)
      const ss = vs.flatMap(v => [...v.querySelectorAll('[data-bd-step]')].filter(s => !s.hidden))
      const s = ss[0]
      return {
        variants: vs.map(v => v.dataset.bdVariant),
        steps: ss.map(x => x.dataset.bdStep),
        svgs: s ? s.querySelectorAll('svg').length : 0,
        eq: s?.querySelector('[data-bd-eq]')?.dataset.bdEq ?? '',
        outs: s ? [...s.querySelectorAll('[data-bd-out]')].map(d => d.dataset.bdOut) : [],
        lit: s ? s.querySelectorAll('svg [data-state="lit"]').length : 0,
        done: s ? s.querySelectorAll('svg [data-state="done"]').length : 0,
        height: s ? s.querySelector('svg').getBoundingClientRect().height : 0,
      }
    }, i)

  for (const [i, config] of configs.entries()) {
    const r = page.locator('[data-block-diagram]').nth(i)
    const nVariants = Math.max(1, await r.locator('[data-bd-variant-input]').count())
    for (let v = 0; v < nVariants; v++) {
      if (nVariants > 1)
        await r.locator(`[data-bd-variant-input][value="${v}"]`).check({ force: true })
      const panel = r.locator(`[data-bd-variant="${v}"]`)
      const nSteps = await panel.locator('[data-bd-step-input]').count()
      const eqs = []
      let shown = true
      let lit = true
      for (let k = 0; k < nSteps; k++) {
        await panel.locator(`[data-bd-step-input][value="${k}"]`).check({ force: true })
        const s = await visible(i)
        eqs.push(norm(s.eq))
        shown &&= s.variants.join() === String(v) && s.steps.join() === String(k) && s.svgs === 1
        // Every rule step lights the parts it acts on; the diagram step lights nothing.
        lit &&= k === 0 ? s.lit === 0 : s.lit > 0 || (s.done > 0 && config === 'sensing-device')
        if (k === nSteps - 1) {
          const key = `${config}:${v}`
          const outs = s.outs.map(norm)
          c.check(`${key}: result matches the note`, outs.includes(norm(expected[key])), outs)
          if (config === 'sensing-device')
            c.check(`${key}: no C(z)/R(z) transfer function`, outs.includes('\\text{none}'), outs)
          if (config === 'summing-junction')
            c.check(
              `${key}: moved diagram gives the same output`,
              outs.includes('\\text{yes}'),
              outs,
            )
          await c.shoot(panel, `${config}-${v}-final.png`)
        }
        if (k === 1) await c.shoot(panel, `${config}-${v}-step1.png`)
      }
      c.check(`${config}:${v}: one step panel shown per choice`, shown)
      c.check(`${config}:${v}: rule steps light their parts`, lit)
      for (const rel of relations[`${config}:${v}`] ?? [])
        c.check(
          `${config}:${v}: derives ${rel}`,
          eqs.some(e => e.includes(norm(rel))),
          eqs,
        )
    }
  }

  // Keyboard: arrow keys move the step radio group.
  const r0 = page.locator('[data-block-diagram]').nth(0)
  await r0.locator('[data-bd-variant-input][value="0"]').check({ force: true })
  await r0.locator('[data-bd-variant="0"] [data-bd-step-input][value="0"]').check({ force: true })
  await r0.locator('[data-bd-variant="0"] [data-bd-step-input][value="0"]').focus()
  await page.keyboard.press('ArrowRight')
  c.check('ArrowRight moves to step 1', (await visible(0)).steps.join() === '1')
  await r0.locator('[data-bd-variant-input][value="0"]').focus()
  await page.keyboard.press('ArrowRight')
  c.check(
    'ArrowRight on the variant group shows positive feedback',
    (await visible(0)).variants.join() === '1',
  )
  const focus = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused segment shows a solid outline', focus === 'solid', focus)
  const ids = await page.evaluate(() => {
    const all = [...document.querySelectorAll('[id]')].map(e => e.id)
    return all.length === new Set(all).size
  })
  c.check('Marker and control ids are unique across seven instances', ids)

  await standardChecks(page, errors, c, '.zoomable')
  await settled320(page, c, '.zoomable')
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await r0.locator('[data-bd-variant-input][value="0"]').check({ force: true })
  c.check('Cleanup removes the change listener', (await visible(0)).variants.join() === '1')
  await browser.close()
  passed = (await c.finish()) && passed
}

// ---------- PatriciaTrie ----------
{
  const out = path.join(art, 'patricia-trie')
  const pagePath = await renderFigurePage({
    component: 'PatriciaTrie',
    out,
    label: 'Radix tree insertion',
    instances: [{ caption: 'Default keys.' }, { words: ['car', 'cart', 'cat'], step: 3 }],
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  const words = ['romane', 'romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'rubicundus']
  const state = i =>
    page.evaluate(i => {
      const r = document.querySelectorAll('[data-patricia-trie]')[i]
      const s = [...r.querySelectorAll('[data-ptr-state]')].filter(x => !x.hidden)
      const v = s[0]
      return {
        shown: s.map(x => x.dataset.ptrState),
        labels: [...v.querySelectorAll('[data-ptr-edge]')].map(e => e.dataset.ptrEdge).sort(),
        keys: [...v.querySelectorAll('[data-ptr-key]')].map(k => [
          k.dataset.ptrKey,
          k.dataset.ptrSpell,
        ]),
        nodes: v.querySelectorAll('.ptr-node, .ptr-key').length,
        lit: [...v.querySelectorAll('.ptr-label[data-state="lit"]')].map(e => e.dataset.ptrEdge),
        out: Object.fromEntries(
          [...v.querySelectorAll('[data-ptr-out]')].map(d => [d.dataset.ptrOut, d.textContent]),
        ),
        say: v.querySelector('[data-ptr-say]').textContent,
      }
    }, i)
  const prefixes = ws =>
    new Set(['', ...ws.flatMap(w => [...w].map((_, i) => w.slice(0, i + 1)))]).size

  const final = await state(0)
  c.check('SSR shows the full tree', final.shown.join() === '7')
  c.check(
    'Final edge labels equal the Wikimedia figure',
    JSON.stringify(final.labels) ===
      JSON.stringify(
        ['an', 'e', 'e', 'ic', 'ns', 'om', 'on', 'r', 'r', 'ub', 'ulus', 'undus', 'us'].sort(),
      ),
    final.labels,
  )
  const r = page.locator('[data-patricia-trie]').nth(0)
  for (let k = 0; k <= 7; k++) {
    await r.locator(`[data-ptr-step-input][value="${k}"]`).check({ force: true })
    const s = await state(0)
    const ws = words.slice(0, k)
    const spelled = s.keys.every(([key, spell]) => spell === words[Number(key) - 1])
    c.check(`step ${k}: every key node spells its key`, spelled && s.keys.length === k, s.keys)
    c.check(
      `step ${k}: drawn nodes equal readout, trie count equals distinct prefixes`,
      String(s.nodes) === s.out['Radix tree nodes'] &&
        s.out['Uncompressed trie nodes'] === String(prefixes(ws)) &&
        s.out['Symbols on edges, either tree'] === String(prefixes(ws) - 1) &&
        s.out['Keys stored'] === String(k),
      { nodes: s.nodes, out: s.out },
    )
    if (k > 0) c.check(`step ${k}: one new edge lit`, s.lit.length === 1, s.lit)
    c.results[`step${k}`] = { say: s.say, out: s.out, lit: s.lit }
    await c.shoot(r, `step-${k}.png`)
  }
  const s4 = c.results.step4.say
  c.check(
    'Insert rubens splits "rom" into "r" + "om"',
    s4.includes('split “rom” into “r” + “om”'),
    s4,
  )
  c.check(
    'Final counts: 14 radix nodes vs 28 trie nodes',
    c.results.step7.out['Radix tree nodes'] === '14' &&
      c.results.step7.out['Uncompressed trie nodes'] === '28',
  )

  const note = await state(1)
  c.check(
    'Note example car, cart, cat: edges ca, r, t, t with car terminal above cart',
    JSON.stringify(note.labels) === JSON.stringify(['ca', 'r', 't', 't']) &&
      note.keys.some(([k, sp]) => k === '1' && sp === 'car'),
    note,
  )

  await r.locator('[data-ptr-step-input][value="2"]').focus()
  await page.keyboard.press('ArrowRight')
  c.check('ArrowRight selects the next insertion', (await state(0)).shown.join() === '3')
  const focus = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('Focused segment shows a solid outline', focus === 'solid', focus)
  await r.locator('[data-ptr-step-input][value="7"]').check({ force: true })
  await standardChecks(page, errors, c, '.zoomable')
  await settled320(page, c, '.zoomable')
  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await r.locator('[data-ptr-step-input][value="1"]').check({ force: true })
  c.check('Cleanup removes the change listener', (await state(0)).shown.join() === '7')
  await browser.close()
  passed = (await c.finish()) && passed
}

process.exit(passed ? 0 : 1)
