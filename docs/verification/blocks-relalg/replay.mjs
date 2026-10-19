// Renders RelAlgTree through the shared figure harness, selects every node by click and keyboard,
// and checks each drawn state and relation against the Sells and Bars tables of the Relational
// Algebra note. Writes the page, screenshots and results.json under ./artifacts/rel-alg-tree.
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
const out = path.join(here, 'artifacts', 'rel-alg-tree')
const notePath =
  'content/thoughts/university/twenty-four-twenty-five/sfwr-3db3/Relational Algebra.md'
const note = await readFile(path.join(repo, notePath), 'utf8')

// The Markdown table that follows a "Name:" line in the note.
const noteTable = name => {
  const lines = note.split('\n')
  const at = lines.indexOf(`${name}:`)
  const rows = []
  for (let i = at + 2; lines[i]?.startsWith('|'); i++)
    rows.push(
      lines[i]
        .split('|')
        .slice(1, -1)
        .map(c => c.trim()),
    )
  return { attrs: rows[0], rows: rows.slice(2) }
}
const sells = noteTable('Sells')
const barsNote = noteTable('Bars')
// The note's own remark: its Bars.name column was renamed to bar for the join example.
const renameRemark = note.includes('Bars.name has become Bars.bar')
const bars = { attrs: barsNote.attrs.map(a => (a === 'bar' ? 'name' : a)), rows: barsNote.rows }

// The query worked by hand over the note's rows.
const col = (rel, a) => rel.attrs.indexOf(a)
const maple = bars.rows.filter(r => r[col(bars, 'addr')] === 'Maple St.')
const bud = sells.rows.filter(
  r => Number(r[col(sells, 'price')]) < 3 && r[col(sells, 'beer')] === 'Bud',
)
const uniq = rows => [...new Map(rows.map(r => [r.join('|'), r])).values()]
const expected = {
  bars: bars.rows,
  'sigma-addr': maple,
  'pi-name': uniq(maple.map(r => [r[col(bars, 'name')]])),
  sells: sells.rows,
  'sigma-bud': bud,
  'pi-bar': uniq(bud.map(r => [r[col(sells, 'bar')]])),
  rho: uniq(bud.map(r => [r[col(sells, 'bar')]])),
  union: uniq([...maple.map(r => [r[col(bars, 'name')]]), ...bud.map(r => [r[col(sells, 'bar')]])]),
}
const parentOf = {
  bars: 'sigma-addr',
  'sigma-addr': 'pi-name',
  'pi-name': 'union',
  sells: 'sigma-bud',
  'sigma-bud': 'pi-bar',
  'pi-bar': 'rho',
  rho: 'union',
}
const kids = id => Object.keys(parentOf).filter(k => parentOf[k] === id)
const subtree = id => [id, ...kids(id).flatMap(subtree)]
const order = ['bars', 'sigma-addr', 'pi-name', 'sells', 'sigma-bud', 'pi-bar', 'rho', 'union']

const pagePath = await renderFigurePage({
  component: 'RelAlgTree',
  out,
  label: 'Relational algebra expression tree',
  // The first instance carries the caption the note embeds.
  instances: [
    { caption: note.match(/<RelAlgTree\s+caption="([^"]+)"/)[1] },
    { select: 'sigma-bud', caption: 'Second instance.' },
  ],
})
const { browser, page, errors } = await openFigurePage(pagePath)
const c = checker(out)
const root = i => page.locator('[data-rel-alg-tree]').nth(i)

const snap = i =>
  page.evaluate(i => {
    const r = document.querySelectorAll('[data-rel-alg-tree]')[i]
    const table = t =>
      t && {
        attrs: [...t.querySelectorAll('thead th')].map(th => th.textContent.trim()),
        rows: [...t.querySelectorAll('tbody tr')].map(tr =>
          [...tr.children].map(td => td.textContent),
        ),
        kept: [...t.querySelectorAll('tbody tr')].map(tr => tr.dataset.kept),
      }
    const panels = [...r.querySelectorAll('[data-rat-panel]')].filter(p => !p.hidden)
    const p = panels[0]
    const tables = Object.fromEntries(
      [...r.querySelectorAll('[data-rat-table]')].map(t => [t.dataset.ratTable, table(t)]),
    )
    return {
      sel: r.dataset.ratSel,
      checked: r.querySelector('[data-rat-input]:checked')?.value,
      checkedAll: [...r.querySelectorAll('[data-rat-input]:checked')].map(x => x.value),
      panels: panels.map(x => x.dataset.ratPanel),
      nodes: Object.fromEntries(
        [...r.querySelectorAll('[data-rat-node]')].map(n => [n.dataset.ratNode, n.dataset.state]),
      ),
      edges: Object.fromEntries(
        [...r.querySelectorAll('[data-rat-edge]')].map(e => [e.dataset.ratEdge, e.dataset.state]),
      ),
      tables,
      inputs: p ? [...p.querySelectorAll('[data-rat-table]')].map(t => t.dataset.ratTable) : [],
      text: p?.querySelector('.rat-text')?.textContent ?? '',
      aria: r.querySelector('[data-rat-input]:checked')?.getAttribute('aria-label'),
      fits: [...r.querySelectorAll('.rat-svg foreignObject')].every(fo => {
        const k = fo.querySelector('.rat-label').getBoundingClientRect()
        const b = fo.getBoundingClientRect()
        return b.width === 0 || (k.width <= b.width - 4 && k.height <= b.height)
      }),
      shown: [...r.querySelectorAll('.rat-svg')]
        .filter(x => x.getBoundingClientRect().width > 0)
        .map(x => x.classList[1]),
    }
  }, i)

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)

c.check('Note has the Sells and Bars tables', sells.rows.length === 4 && bars.rows.length === 2, {
  sells,
  bars,
})
c.check('Note states Bars.name became Bars.bar', renameRemark)
c.check(
  'Hand-worked answer over the note rows: Joe’s and Sue’s',
  same(expected.union, [['Joe’s'], ['Sue’s']]),
  expected.union,
)

const first = await snap(0)
c.check(
  'SSR selects the union and shows its panel',
  first.sel === 'union' && same(first.panels, ['union']),
)
c.check(
  'Stored relations in the figure equal the note tables',
  same(first.tables['sells:out'].attrs, sells.attrs) &&
    same(first.tables['sells:out'].rows, sells.rows) &&
    same(first.tables['bars:out'].attrs, bars.attrs) &&
    same(first.tables['bars:out'].rows, bars.rows),
  { sells: first.tables['sells:out'], bars: first.tables['bars:out'] },
)
c.check('Every KaTeX node label fits its box (wide tree)', first.fits)
c.check(
  'At 900px the wide tree shows and the outline is hidden',
  same(first.shown, ['rat-svg--wide']),
)

// Click every node; check the lit node, the sage subtree, the edges and the relations.
for (const id of order) {
  await root(0).locator(`.rat-svg--wide [data-rat-node="${id}"] label`).click()
  const s = await snap(0)
  const sub = subtree(id)
  const nodesOk = Object.entries(s.nodes).every(
    ([n, st]) => st === (n === id ? 'lit' : sub.includes(n) ? 'done' : 'idle'),
  )
  const edgesOk = Object.entries(s.edges).every(
    ([child, st]) => st === (child !== id && sub.includes(child) ? 'on' : 'off'),
  )
  c.check(
    `${id}: one panel, salmon node, sage subtree, dark edges inside it`,
    s.sel === id && same(s.checkedAll, [id, id]) && same(s.panels, [id]) && nodesOk && edgesOk,
    s,
  )
  const result = s.tables[`${id}:out`]
  c.check(`${id}: result equals the hand-worked relation`, same(result.rows, expected[id]), result)
  c.check(
    `${id}: radio names the tuple count`,
    s.aria.endsWith(`${expected[id].length} tuple${expected[id].length === 1 ? '' : 's'}`),
    s.aria,
  )
  // Each input table is the result its child node produces.
  const ins = kids(id)
  const inputsOk = ins.every((k, i) => {
    const t = s.tables[`${id}:in${i}`]
    const child = s.tables[`${k}:out`]
    const n = child.attrs.length
    return (
      same(t.attrs.slice(0, n), child.attrs) &&
      same(
        t.rows.map(r => r.slice(0, n)),
        child.rows,
      )
    )
  })
  c.check(`${id}: inputs equal its children's results (${ins.join(', ') || 'leaf'})`, inputsOk)
  if (id.startsWith('sigma')) {
    const t = s.tables[`${id}:in0`]
    const tests = t.attrs.length - (id === 'sigma-bud' ? sells : bars).attrs.length
    const keptOk = t.rows.every(
      (r, i) => t.kept[i] === String(r.slice(-tests).every(v => v === 'true')),
    )
    c.check(
      `${id}: sage rows are exactly those that pass every test, and the result is them`,
      keptOk &&
        same(
          t.rows.filter((_, i) => t.kept[i] === 'true').map(r => r.slice(0, -tests)),
          result.rows,
        ),
      t,
    )
  }
  c.results[id] = { text: s.text, result: result.rows }
  await page.waitForTimeout(250)
  await c.shoot(root(0), `node-${id}.png`)
}
c.check('Rename turns bar into name', c.results.rho.text.includes('bar becomes name'))
c.check(
  'Union reports Joe’s in both inputs',
  c.results.union.text.includes('Joe’s is in both inputs') &&
    c.results.union.text.includes('holds 2 tuples'),
  c.results.union.text,
)
c.check(
  'Price test is numeric: 2.75 < 3 is true, 3.00 < 3 is false',
  same(
    (await snap(0)).tables['sigma-bud:in0'].rows.map(r => r[3]),
    ['true', 'true', 'true', 'false'],
  ),
)

// Keyboard: arrow keys walk the radio group in evaluation order.
await root(0).locator('.rat-svg--wide [data-rat-node="bars"] label').click()
await root(0).locator('.rat-svg--wide [data-rat-node="bars"] input').focus()
const walked = []
for (let k = 0; k < order.length - 1; k++) {
  await page.keyboard.press('ArrowRight')
  walked.push((await snap(0)).sel)
}
c.check('ArrowRight walks the nodes bottom-up', same(walked, order.slice(1)), walked)
const outline = await page.evaluate(
  () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
)
c.check('Focused node shows a solid outline', outline === 'solid', outline)
await c.shoot(root(0), 'keyboard-union-focus.png')

const second = await snap(1)
c.check(
  'Second instance keeps its own selection',
  second.sel === 'sigma-bud' && same(second.panels, ['sigma-bud']),
  second.sel,
)
c.check(
  'Ids and radio names are unique per instance',
  await page.evaluate(() => {
    const names = [...document.querySelectorAll('[data-rel-alg-tree]')].map(
      r => r.querySelector('[data-rat-input]').name,
    )
    return new Set(names).size === names.length
  }),
)

await root(0).locator('.rat-svg--wide [data-rat-node="union"] label').click()
await standardChecks(page, errors, c, '.zoomable')
// standardChecks shoots 320px right after a theme switch; these wait for colours to settle.
await page.setViewportSize({ width: 320, height: 1400 })
await page.waitForTimeout(400)
for (let i = 0; i < 2; i++) await c.shoot(page.locator('.zoomable').nth(i), `320-settled-${i}.png`)
// At 320px the outline replaces the tree; it drives the same selection by click and keyboard.
const narrow = await snap(0)
c.check(
  'At 320px the outline shows, its labels fit',
  same(narrow.shown, ['rat-svg--narrow']) && narrow.fits,
  narrow.shown,
)
await root(0).locator('.rat-svg--narrow [data-rat-node="sigma-addr"] label').click()
const n1 = await snap(0)
c.check(
  'Outline click selects σ_addr in both groups and shows its panel',
  n1.sel === 'sigma-addr' &&
    same(n1.checkedAll, ['sigma-addr', 'sigma-addr']) &&
    same(n1.panels, ['sigma-addr']),
  n1,
)
await root(0).locator('.rat-svg--narrow [data-rat-node="sigma-addr"] input').focus()
await page.keyboard.press('ArrowRight')
const n2 = await snap(0)
c.check(
  'ArrowRight in the outline moves to π_name',
  n2.sel === 'pi-name' && same(n2.checkedAll, ['pi-name', 'pi-name']),
  n2.sel,
)
await page.waitForTimeout(250)
await c.shoot(page.locator('.zoomable').nth(0), '320-outline-pi-name.png')
await page.setViewportSize({ width: 900, height: 1400 })
const w1 = await snap(0)
c.check(
  'Back at 900px the wide tree keeps the outline choice',
  w1.sel === 'pi-name' && same(w1.shown, ['rat-svg--wide']),
)
await root(0).locator('.rat-svg--wide [data-rat-node="union"] label').click()
await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
await root(0).locator('.rat-svg--wide [data-rat-node="bars"] label').click()
c.check('Cleanup removes the change listener', (await snap(0)).sel === 'union')
await browser.close()
process.exit((await c.finish()) ? 0 : 1)
