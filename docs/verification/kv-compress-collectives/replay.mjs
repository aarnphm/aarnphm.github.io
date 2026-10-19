// Renders KVCompressBlocks (two instances, as the note embeds it) and AllReduceSteps through the
// shared figure harness, drives every control in Chromium and checks the drawn state against
// values read from the source figures. Writes page.html, screenshots and results.json to
// ./artifacts for each figure.
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const artifacts = path.join(here, 'artifacts')
let ok = true

// Props come from the embeds in the notes, so the replay renders what the pages render.
const embeds = async (note, component) => {
  const text = await readFile(path.join(here, '../../../content', note), 'utf8')
  return [...text.matchAll(new RegExp(`<${component}\\s([^>]*?)/>`, 'gs'))].map(m => {
    const props = {}
    for (const [, k, str, num] of m[1].matchAll(/(\w+)=(?:"([^"]*)"|\{(\d+)\})/g))
      props[k] = str ?? Number(num)
    return props
  })
}

// ---------- KVCompressBlocks ----------
{
  const out = path.join(artifacts, 'kv-compress-blocks')
  const pagePath = await renderFigurePage({
    component: 'KVCompressBlocks',
    label: 'KV-Compress block layout and eviction',
    instances: await embeds('thoughts/KV compression.md', 'KVCompressBlocks'),
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  const root = i => page.locator('[data-kv-compress-blocks]').nth(i)
  c.check(
    'the note embeds two instances, layout first',
    (await page.locator('[data-kv-compress-blocks]').count()) === 2,
  )

  // Expected cells per step, read from the paper's eviction figure. Each cell is
  // "<metric or _>:<logical index>:<mark>", with marks k kv, e empty, x evicted, f freed,
  // m moved, v vacated; a trailing * is the outlined block metric.
  const E = {
    1: '0.1:4:k _:5:e | 1.2:0:k 2.3:1:k | 0.4:0:k 2.5:1:k | 0.6:2:k 1.7:3:k',
    2: '_:5:e 0.1:4:k 0.4:0:k 0.6:2:k 1.7:3:k 2.5:1:k 1.2:0:k 2.3:1:k',
    3: '_:5:e 0.1:4:k | 0.4:0:k 0.6:2:k | 1.7:3:k 2.5:1:k | 1.2:0:k 2.3:1:k',
    4: '_:5:e 0.1:4:k* | 0.4:0:k 0.6:2:k* | 1.2:0:k 2.3:1:k* | 1.7:3:k 2.5:1:k*',
    5: '_:5:e 0.1:4:x | 0.4:0:x 0.6:2:x | 1.2:0:k 2.3:1:k | 1.7:3:k 2.5:1:k',
    6: '_:5:e 0.1:4:x 0.4:0:x 0.6:2:x 1.2:0:k 2.3:1:k 1.7:3:k 2.5:1:k',
    7: '0.4:0:x 2.5:1:k 0.6:2:x 1.7:3:k 0.1:4:x _:5:e 1.2:0:k 2.3:1:k',
    8: '1.7:0:m 2.5:1:k 0.6:2:x _:3:v 0.1:4:x _:5:e 1.2:0:k 2.3:1:k',
    9: '1.7:0:k 2.5:1:k _:2:f _:3:f _:4:f _:5:f 1.2:0:k 2.3:1:k',
    10: '_:4:f _:5:f | 1.2:0:k 2.3:1:k | 1.7:0:k 2.5:1:k | _:2:f _:3:f',
  }
  const flat = s => s.replaceAll('| ', '')
  const code = { kv: 'k', empty: 'e', evict: 'x', free: 'f', moved: 'm', vacated: 'v' }

  const drawn = i =>
    page.evaluate(
      ([i, code]) => {
        const r = document.querySelectorAll('[data-kv-compress-blocks]')[i]
        const step = r.getAttribute('data-kvc-step')
        const frames = [...r.querySelectorAll('.kvc-frame')]
        const visible = frames.filter(f => getComputedStyle(f).visibility === 'visible')
        const cells = visible[0]
          ? [...visible[0].querySelectorAll('.kvc-slot')].map(g => {
              const m = g.querySelector('.kvc-metric')?.textContent ?? '_'
              const li = g.querySelector('.kvc-index')?.textContent
              const key = g.getAttribute('data-kvc-key') === 'true' ? '*' : ''
              return `${m}:${li}:${code[g.getAttribute('data-kvc-mark')]}${key}`
            })
          : []
        const text = [...r.querySelectorAll('[data-kvc-step-text]')].find(
          p => getComputedStyle(p).display !== 'none',
        )
        return {
          step,
          visible: visible.map(f => f.getAttribute('data-kvc-frame')),
          cells: cells.join(' '),
          text: text?.getAttribute('data-kvc-step-text'),
          range: visible[0]?.querySelector('.kvc-range') ? true : false,
          move: visible[0]?.querySelector('.kvc-move') ? true : false,
        }
      },
      [i, code],
    )

  // Instance two starts on the eviction view; wait out the 180ms crossfade before each read.
  const settle = () => page.waitForTimeout(260)
  await settle()
  for (let s = 1; s <= 10; s++) {
    if (s > 1) await root(1).locator('[data-kvc-nav="1"]').click()
    await settle()
    const d = await drawn(1)
    c.check(
      `step ${s}: one frame visible and it is frame ${s}`,
      d.visible.join() === String(s),
      d.visible,
    )
    c.check(`step ${s}: step text ${s} shown`, d.text === String(s), d.text)
    c.check(`step ${s}: cells match the paper figure`, d.cells === flat(E[s]), {
      drawn: d.cells,
      expected: flat(E[s]),
    })
    c.check(
      `step ${s}: eviction range drawn only in steps 7 to 9`,
      d.range === (s >= 7 && s <= 9),
      d.range,
    )
    c.check(`step ${s}: move arc drawn only in step 8`, d.move === (s === 8), d.move)
    if ([1, 4, 5, 7, 8, 9, 10].includes(s)) await c.shoot(root(1), `light-evict-step-${s}.png`)
  }
  c.check(
    'next is disabled on the last step',
    await root(1).locator('[data-kvc-nav="1"]').isDisabled(),
  )

  const readout = await root(1)
    .locator('[data-kvc-out]')
    .evaluateAll(els => Object.fromEntries(els.map(e => [e.dataset.kvcOut, e.textContent])))
  c.check(
    'readout agrees with the figure outcome',
    readout['per-head'] === 'h0: 2 · h1: 0' &&
      readout.evicted === '0.1, 0.4, 0.6' &&
      readout.moves === '1' &&
      readout.freed === 'p1, p7' &&
      readout.kept === 'h0: 2 · h1: 2',
    readout,
  )

  // Keyboard: step radios move with arrow keys.
  await root(1).locator('input[data-kvc-step-input][value="3"]').focus()
  await page.keyboard.press('ArrowRight')
  await settle()
  const afterArrow = await drawn(1)
  c.check(
    'ArrowRight on step radios selects step 4',
    afterArrow.step === '4' && afterArrow.visible.join() === '4',
    afterArrow,
  )
  const stepOutline = await page.evaluate(
    () => getComputedStyle(document.activeElement.closest('label')).outlineStyle,
  )
  c.check('focused step segment shows an outline', stepOutline === 'solid', stepOutline)

  // Tabs: click, then arrow keys with wraparound, Home and End.
  const tabState = i =>
    root(i).evaluate(r => ({
      view: r.getAttribute('data-kvc-view'),
      selected: [...r.querySelectorAll('[role=tab]')]
        .filter(t => t.getAttribute('aria-selected') === 'true')
        .map(t => t.dataset.kvcTab),
      visible: [...r.querySelectorAll('[role=tabpanel]')]
        .filter(p => !p.hidden)
        .map(p => p.dataset.kvcPanel),
      focus: document.activeElement?.dataset?.kvcTab ?? null,
      tabindex: [...r.querySelectorAll('[role=tab]')].map(t => t.tabIndex),
    }))
  const s0 = await tabState(0)
  c.check(
    'instance one starts on the vLLM tab',
    s0.view === 'vllm' && s0.visible.join() === 'vllm',
    s0,
  )
  await root(0).locator('[data-kvc-tab="kvcompress"]').click()
  const s1 = await tabState(0)
  c.check(
    'click selects the KV-Compress tab',
    s1.selected.join() === 'kvcompress' && s1.visible.join() === 'kvcompress',
    s1,
  )
  await page.keyboard.press('ArrowRight')
  const s2 = await tabState(0)
  c.check(
    'ArrowRight wraps to the eviction tab',
    s2.selected.join() === 'evict' && s2.focus === 'evict' && s2.visible.join() === 'evict',
    s2,
  )
  await page.keyboard.press('End')
  const s3 = await tabState(0)
  c.check('End selects the last tab', s3.selected.join() === 'kvcompress', s3)
  await page.keyboard.press('Home')
  const s4 = await tabState(0)
  c.check(
    'Home selects the first tab',
    s4.selected.join() === 'evict' && JSON.stringify(s4.tabindex) === '[0,-1,-1]',
    s4,
  )
  const tabOutline = await page.evaluate(
    () => getComputedStyle(document.activeElement).outlineStyle,
  )
  c.check('focused tab shows an outline', tabOutline === 'solid', tabOutline)
  const second = await tabState(1)
  c.check(
    'instance two keeps its own view and step',
    second.view === 'evict' && (await drawn(1)).step === '4',
    second,
  )

  // Layout tables against the model's accounting.
  await root(0).locator('[data-kvc-tab="vllm"]').click()
  await c.shoot(root(0), 'light-vllm.png')
  const vllm = await root(0).evaluate(r => {
    const p = r.querySelector('[data-kvc-panel="vllm"]')
    return {
      rows: p.querySelectorAll('.kvc-table tbody tr').length,
      lit: p.querySelectorAll('.kvc-lslot[data-kvc-mark="lit"]').length,
      slots: p.querySelectorAll('.kvc-lslot').length,
      compare: [...p.querySelectorAll('.kvc-compare tbody tr')].map(tr =>
        [...tr.querySelectorAll('td')].map(td => td.textContent),
      ),
    }
  })
  c.check('vLLM: one block-table row', vllm.rows === 1, vllm.rows)
  c.check('vLLM: the lit block id spans 2 layers x 2 heads x 2 slots', vllm.lit === 8, vllm.lit)
  c.check(
    'layout comparison: 24 vs 16 slots for 14 KVs',
    JSON.stringify(vllm.compare.slice(3)) ===
      JSON.stringify([
        ['24', '16'],
        ['14', '14'],
        ['10', '2'],
      ]),
    vllm.compare,
  )
  await root(0).locator('[data-kvc-tab="kvcompress"]').click()
  await c.shoot(root(0), 'light-kvcompress.png')
  const kvc = await root(0).evaluate(r => {
    const p = r.querySelector('[data-kvc-panel="kvcompress"]')
    return {
      rows: [...p.querySelectorAll('.kvc-table tbody tr')].map(
        tr => tr.querySelector('th').textContent,
      ),
      lit: p.querySelectorAll('.kvc-lslot[data-kvc-mark="lit"]').length,
      own: p.querySelectorAll('.kvc-lslot:not([data-kvc-mark="other"])').length,
    }
  })
  c.check(
    'KV-Compress: block tables expanded to l x H = 4 rows',
    kvc.rows.join() === 'L0 h0,L0 h1,L1 h0,L1 h1',
    kvc.rows,
  )
  c.check('KV-Compress: the lit block is one head of one layer', kvc.lit === 2, kvc.lit)
  c.check('KV-Compress: 8 head-blocks of 2 slots allocated', kvc.own === 16, kvc.own)

  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'dark'))
  await root(1).locator('input[data-kvc-step-input][value="8"]').check({ force: true })
  await settle()
  await c.shoot(root(1), 'dark-evict-step-8.png')
  await c.shoot(root(0), 'dark-kvcompress.png')
  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'light'))
  await page.setViewportSize({ width: 320, height: 1400 })
  await c.shoot(root(1), '320-evict-step-8.png')
  await root(0).locator('[data-kvc-tab="vllm"]').click()
  await c.shoot(root(0), '320-vllm.png')
  await page.setViewportSize({ width: 900, height: 1400 })

  // Theme switches happen right before each shot; reduced motion keeps fills from mid-transition.
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await standardChecks(page, errors, c, '.zoomable')

  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await root(0).locator('[data-kvc-tab="evict"]').click()
  c.check(
    'cleanup removes the tab listener',
    (await tabState(0)).view === 'vllm',
    (await tabState(0)).view,
  )

  await browser.close()
  ok = (await c.finish()) && ok
}

// ---------- AllReduceSteps ----------
{
  const out = path.join(artifacts, 'all-reduce-steps')
  const pagePath = await renderFigurePage({
    component: 'AllReduceSteps',
    label: 'Ring all-reduce',
    instances: await embeds('thoughts/tsfm/tangents-3.md', 'AllReduceSteps'),
    out,
  })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  const root = page.locator('[data-all-reduce-steps]')
  const L = 'ABCDEF'

  const drawn = () =>
    root.evaluate(r => {
      const n = Number(r.getAttribute('data-ars-n'))
      const svg = r.querySelector(`[data-ars-grid="${n}"]`)
      const cells = [...svg.querySelectorAll('[data-ars-cell]')]
      return {
        n,
        step: Number(r.getAttribute('data-ars-step')),
        visibleGrids: [...r.querySelectorAll('[data-ars-grid]')].filter(
          s => !s.hasAttribute('hidden'),
        ).length,
        labels: cells.map(g => g.querySelector('text').textContent),
        states: cells.map(g => g.getAttribute('data-ars-state')),
        senders: cells.filter(g => g.getAttribute('data-ars-sender') === 'true').length,
        out: Object.fromEntries(
          [...r.querySelectorAll('[data-ars-out]')].map(e => [e.dataset.arsOut, e.textContent]),
        ),
      }
    })

  // Independent ring simulation: GPU i sends to i + 1; reduce-scatter adds, all-gather copies.
  const simulate = n => {
    const mod = a => ((a % n) + n) % n
    let buf = Array.from({ length: n }, (_, g) => Array.from({ length: n }, () => [g]))
    const frames = [buf]
    for (const phase of ['rs', 'ag'])
      for (let t = 1; t < n; t++) {
        const next = buf.map(col => col.map(x => [...x]))
        for (let i = 0; i < n; i++) {
          const ch = phase === 'rs' ? mod(i - t) : mod(i - t + 1)
          next[mod(i + 1)][ch] =
            phase === 'rs'
              ? [...new Set([...next[mod(i + 1)][ch], ...buf[i][ch]])]
              : [...buf[i][ch]]
        }
        buf = next
        frames.push(buf)
      }
    return frames.map(f =>
      Array.from({ length: n * n }, (_, k) => {
        const s = f[k % n][Math.floor(k / n)]
        return s.length === n
          ? 'Σ'
          : [...s]
              .sort((a, b) => a - b)
              .map(g => L[g])
              .join('+')
      }),
    )
  }

  for (const n of [2, 3, 4, 6]) {
    await root.locator(`input[data-ars-n-input][value="${n}"]`).check({ force: true })
    const sim = simulate(n)
    for (let s = 0; s <= 2 * (n - 1); s++) {
      if (s > 0) await root.locator('[data-ars-nav="1"]').click()
      const d = await drawn()
      c.check(
        `N=${n} step ${s}: cell labels match the ring simulation`,
        JSON.stringify(d.labels) === JSON.stringify(sim[s]),
        { drawn: d.labels, expected: sim[s] },
      )
      c.check(
        `N=${n} step ${s}: one grid visible`,
        d.visibleGrids === 1 && d.n === n,
        d.visibleGrids,
      )
      c.check(
        `N=${n} step ${s}: ${s === 0 ? 0 : n} senders and receivers lit`,
        d.senders === (s === 0 ? 0 : n) &&
          d.states.filter(x => x === 'recv').length === (s === 0 ? 0 : n),
        d.senders,
      )
      const [num, den] = [s, n]
      const g = (a, b) => (b ? g(b, a % b) : a)
      const expectSent =
        s === 0
          ? '0 S'
          : den / g(num, den) === 1
            ? `${num / den} S`
            : `${num / g(num, den)}/${den / g(num, den)} S`
      c.check(
        `N=${n} step ${s}: sent so far is ${expectSent}`,
        d.out.sent === expectSent,
        d.out.sent,
      )
      if (n === 4 && s === n - 1) {
        const diag = Array.from({ length: n }, (_, i) => d.labels[i * n + i])
        c.check(
          'N=4 after reduce-scatter: GPU i holds the full sum of chunk i',
          diag.every(x => x === 'Σ'),
          diag,
        )
        await c.shoot(root, 'light-n4-step-3.png')
      }
      if (n === 4 && (s === 1 || s === 5)) await c.shoot(root, `light-n4-step-${s}.png`)
    }
    const end = await drawn()
    c.check(
      `N=${n}: every cell is the full sum after 2(N-1) steps`,
      end.labels.every(x => x === 'Σ') && end.step === 2 * (n - 1),
      end.step,
    )
    const total = (2 * (n - 1)) / n
    c.check(
      `N=${n}: total readout is 2(N-1)/N S = ${total} S`,
      end.out.total.startsWith(`2(${n}−1)/${n} S = `) &&
        end.out.total
          .split('= ')[1]
          .replace(' S', '')
          .split('/')
          .map(Number)
          .reduce((a, b) => a / b) === total,
      end.out.total,
    )
    c.check(
      `N=${n}: next disabled at the end`,
      await root.locator('[data-ars-nav="1"]').isDisabled(),
    )
  }

  await root.locator('[data-ars-nav="reset"]').click()
  c.check('reset returns to step 0', (await drawn()).step === 0)
  await root.locator('input[data-ars-n-input][value="4"]').check({ force: true })
  await root.locator('input[data-ars-n-input][value="4"]').focus()
  await page.keyboard.press('ArrowLeft')
  c.check('ArrowLeft on the N radios selects N=3', (await drawn()).n === 3)
  await page.keyboard.press('ArrowRight')
  await root.locator('[data-ars-nav="1"]').focus()
  await page.keyboard.press('Enter')
  await page.keyboard.press('Space')
  c.check(
    'Enter and Space on next advance two steps',
    (await drawn()).step === 2,
    (await drawn()).step,
  )
  const navOutline = await page.evaluate(
    () => getComputedStyle(document.activeElement).outlineStyle,
  )
  c.check('focused button shows an outline', navOutline === 'solid', navOutline)
  for (let i = 0; i < 4; i++) await page.keyboard.press('Enter')
  const atEnd = await drawn()
  c.check(
    'focus moves to prev when next disables at the end',
    atEnd.step === 6 && (await page.evaluate(() => document.activeElement.dataset.arsNav)) === '-1',
    atEnd.step,
  )
  await c.shoot(root, 'light-n4-step-6.png')
  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'dark'))
  await root.locator('[data-ars-nav="reset"]').click()
  await root.locator('[data-ars-nav="1"]').click()
  await root.locator('[data-ars-nav="1"]').click()
  await c.shoot(root, 'dark-n4-step-2.png')
  await page.evaluate(() => document.documentElement.setAttribute('saved-theme', 'light'))
  await root.locator('input[data-ars-n-input][value="6"]').check({ force: true })
  await root.locator('[data-ars-nav="1"]').click()
  await root.locator('[data-ars-nav="1"]').click()
  await c.shoot(root, 'light-n6-step-2.png')
  await page.setViewportSize({ width: 320, height: 1400 })
  await c.shoot(root, '320-n6-step-2.png')
  await page.setViewportSize({ width: 900, height: 1400 })
  await root.locator('input[data-ars-n-input][value="4"]').check({ force: true })

  // Theme switches happen right before each shot; reduced motion keeps fills from mid-transition.
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await standardChecks(page, errors, c, '.zoomable')

  await page.evaluate(() => window.__cleanups.forEach(fn => fn()))
  await root.locator('[data-ars-nav="1"]').click()
  c.check('cleanup removes the click listener', (await drawn()).step === 0, (await drawn()).step)

  await browser.close()
  ok = (await c.finish()) && ok
}

process.exit(ok ? 0 : 1)
