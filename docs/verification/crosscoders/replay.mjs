// Replay: node docs/verification/crosscoders/replay.mjs
// Renders CrossLayerFeatures and CrosscoderMasks through the shared harness, drives every control
// by pointer and keyboard, and checks each drawn state against rules restated from Anthropic's
// crosscoders preview and the note. Also parses the four new note fences with remark-mdx.
import { mkdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import remarkMath from 'remark-math'
import remarkMdx from 'remark-mdx'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import {
  checker,
  openFigurePage,
  renderFigurePage,
  repo,
  standardChecks,
} from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const art = path.join(here, 'artifacts')
let failed = false

async function run(name, component, instances, body) {
  const out = path.join(art, name)
  const pagePath = await renderFigurePage({ component, instances, out })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  try {
    await body(page, c, out)
    await standardChecks(page, errors, c, '.zoomable')
  } finally {
    await browser.close()
  }
  console.log(`[${name}]`)
  if (!(await c.finish())) failed = true
}

// Every line endpoint must sit on a drawn node: a rect edge or a circle's rim.
const endpointsOnNodes = root =>
  root.evaluate(el => {
    const visible = n => {
      for (let p = n; p && p !== el; p = p.parentElement) {
        const cs = getComputedStyle(p)
        if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false
      }
      return true
    }
    const rects = [...el.querySelectorAll('svg rect')]
      .filter(visible)
      .map(r => ({
        x: +r.getAttribute('x'),
        y: +r.getAttribute('y'),
        w: +r.getAttribute('width'),
        h: +r.getAttribute('height'),
      }))
    const circles = [...el.querySelectorAll('svg circle')].map(c => ({
      x: +c.getAttribute('cx'),
      y: +c.getAttribute('cy'),
      r: +c.getAttribute('r'),
    }))
    const lines = [...el.querySelectorAll('svg line')].filter(visible)
    const streams = lines
      .filter(l => /stream/.test(l.getAttribute('class')))
      .map(l => ['x1', 'y1', 'x2', 'y2'].map(a => +l.getAttribute(a)))
    const e = 0.6
    const onRect = (x, y) =>
      rects.some(
        r =>
          x >= r.x - e &&
          x <= r.x + r.w + e &&
          y >= r.y - e &&
          y <= r.y + r.h + e &&
          (Math.abs(x - r.x) < e ||
            Math.abs(x - r.x - r.w) < e ||
            Math.abs(y - r.y) < e ||
            Math.abs(y - r.y - r.h) < e),
      )
    const onCircle = (x, y) => circles.some(c => Math.abs(Math.hypot(x - c.x, y - c.y) - c.r) < e)
    // A block wire may also end on the stream line it reads from or writes to.
    const onStream = (x, y) =>
      streams.some(
        ([x1, y1, x2, y2]) =>
          Math.abs(y - y1) < e &&
          Math.abs(y2 - y1) < e &&
          x >= Math.min(x1, x2) &&
          x <= Math.max(x1, x2),
      )
    const bad = []
    for (const l of lines) {
      if (/stream|grid/.test(l.getAttribute('class'))) continue
      const pts = [
        [+l.getAttribute('x1'), +l.getAttribute('y1')],
        [+l.getAttribute('x2'), +l.getAttribute('y2')],
      ]
      for (const [x, y] of pts)
        if (!onRect(x, y) && !onCircle(x, y) && !onStream(x, y)) bad.push([x, y])
    }
    return { lines: lines.length, bad }
  })

// ---------- CrossLayerFeatures ----------
await run(
  'cross-layer-features',
  'CrossLayerFeatures',
  [{ view: 'lifecycle' }, { view: 'circuit' }],
  async (page, c) => {
    const roots = page.locator('[data-cross-layer-features]')
    const life = roots.nth(0)
    const circ = roots.nth(1)

    const readLife = () =>
      life.evaluate(root => {
        const shown = n => {
          for (let p = n; p && p !== root; p = p.parentElement) {
            const cs = getComputedStyle(p)
            if (cs.display === 'none' || cs.visibility === 'hidden') return false
          }
          return true
        }
        const blocks = [...root.querySelectorAll('.clf-block')].map(g => ({
          stage: g.querySelector('.clf-stage').textContent,
          write: Number(g.querySelector('.clf-value').textContent.replace('−', '-')),
          lit: g.querySelector('.clf-blockbox--on') !== null,
        }))
        const svg = root.querySelector('svg')
        const stationX = [...svg.querySelectorAll('.clf-station')].map(s => +s.getAttribute('cx'))
        const bars = Object.fromEntries(
          [...svg.querySelectorAll('.clf-bar')].map(b => [
            Math.round(+b.getAttribute('x') + +b.getAttribute('width') / 2),
            +b.getAttribute('height'),
          ]),
        )
        const strength = stationX.map(x => {
          const t = [...svg.querySelectorAll('text.clf-value')].find(
            t => Math.abs(+t.getAttribute('x') - x) < 0.5 && !t.closest('.clf-coder,.clf-block'),
          )
          return Number(t.textContent)
        })
        const saeLatents = [...root.querySelectorAll('[data-clf-sae-latent]')]
          .filter(shown)
          .map(r => Number(r.getAttribute('data-clf-sae-latent')))
        const cc = root.querySelector('[data-clf-cc-latent]')
        const norms = [...root.querySelectorAll('[data-clf-norm]')]
          .filter(shown)
          .map(t => Number(t.textContent.split('=')[1]))
        const dd = [...root.querySelectorAll('.clf-readout dd')].map(d =>
          [...d.children]
            .filter(shown)
            .map(s => s.textContent.trim())
            .join(''),
        )
        return {
          blocks,
          stationX,
          bars,
          strength,
          saeLatents,
          ccShown: shown(cc),
          cc: { x: +cc.getAttribute('x'), w: +cc.getAttribute('width') },
          norms,
          dd,
          coder: root.querySelector('[data-clf-input]:checked').value,
        }
      })

    const s = await readLife()
    // r_{l+1} = r_l + u_l on the feature direction, with each stage's write restated from the
    // preview's lifecycle: forming and amplifying add one unit, deletion removes what is there.
    const expect = [0]
    const writes = []
    for (const b of s.blocks) {
      const before = expect.at(-1)
      const w =
        b.stage === 'forms' || b.stage === 'amplified' ? 1 : b.stage === 'deleted' ? -before : 0
      writes.push(w)
      expect.push(before + w)
    }
    c.check(
      'lifecycle stages run absent, forms twice, persists, amplified, persists, deleted',
      s.blocks.map(b => b.stage).join() ===
        'absent,absent,forms,forms,persists,amplified,persists,deleted',
      s.blocks.map(b => b.stage),
    )
    c.check(
      'block writes follow the stages and only writing blocks are sage',
      s.blocks.every((b, k) => b.write === writes[k] && b.lit === (writes[k] !== 0)),
      { drawn: s.blocks, writes },
    )
    c.check('strength labels equal s_{l+1} = s_l + w_l', s.strength.join() === expect.join(), {
      drawn: s.strength,
      expect,
    })
    const unit = Object.values(s.bars)[0] / expect[s.stationX.indexOf(+Object.keys(s.bars)[0])]
    c.check(
      'strength bars are proportional to s_l and absent where s_l = 0',
      s.stationX.every((x, l) =>
        expect[l] === 0 ? !(x in s.bars) : Math.abs(s.bars[x] - unit * expect[l]) < 0.01,
      ),
      { bars: s.bars, expect },
    )
    const active = expect.map((v, l) => [v, l]).filter(([v, l]) => l > 0 && v > 0)
    const penalty = active.reduce((a, [v]) => a + v, 0)
    c.check(
      'per-layer SAEs: one salmon latent per position where the feature is present',
      s.saeLatents.join() === active.map(([, l]) => l).join() && !s.ccShown,
      { drawn: s.saeLatents, active },
    )
    c.check(
      'per-layer readout: 8 SAEs, 5 latents, activations and penalty from the strengths',
      s.dd[0] === '8 SAEs' &&
        s.dd[1] === String(active.length) &&
        s.dd[2] === active.map(([v]) => v).join(' ') &&
        s.dd[4].includes(`${penalty}λ`),
      s.dd,
    )

    await life.locator('.clf-segment', { hasText: 'crosscoder' }).click()
    await page.waitForTimeout(250)
    const x = await readLife()
    const first = s.stationX[active[0][1]]
    const last = s.stationX[active.at(-1)[1]]
    c.check(
      'crosscoder: one latent spanning exactly the positions where the feature is present',
      x.saeLatents.length === 0 &&
        x.ccShown &&
        Math.abs(x.cc.x - (first - 9)) < 0.01 &&
        Math.abs(x.cc.x + x.cc.w - (last + 9)) < 0.01,
      { cc: x.cc, first, last },
    )
    c.check(
      'crosscoder decoder norms equal s_l at r1..r8 with z = 1',
      x.norms.join() === expect.slice(1).join(),
      x.norms,
    )
    c.check(
      'crosscoder readout: 1 latent and the same L1-of-norms penalty as the SAEs',
      x.dd[0] === '1 crosscoder' &&
        x.dd[1] === '1' &&
        x.dd[3] === expect.slice(1).join(' ') &&
        x.dd[4].includes(`${penalty}λ`),
      x.dd,
    )
    const lifeEnds = await endpointsOnNodes(life)
    c.check('lifecycle wires end on blocks or the stream', lifeEnds.bad.length === 0, lifeEnds)
    await page
      .locator('.zoomable')
      .nth(0)
      .screenshot({ path: path.join(art, 'cross-layer-features', 'lifecycle-crosscoder.png') })

    // Keyboard: the radio group moves with the arrow keys.
    await life.locator('[data-clf-input][value="crosscoder"]').focus()
    await page.keyboard.press('ArrowLeft')
    await page.waitForTimeout(250)
    const k = await readLife()
    const focusOutline = await life
      .locator('.clf-segment:has(input:focus-visible)')
      .evaluate(el => getComputedStyle(el).outlineStyle)
      .catch(() => 'none')
    c.check(
      'ArrowLeft returns to per-layer SAEs with visible focus',
      k.coder === 'sae' && k.saeLatents.length === active.length && focusOutline === 'solid',
      { coder: k.coder, focusOutline },
    )

    // ---- circuit ----
    const readCirc = () =>
      circ.evaluate(root => {
        const shown = n => {
          for (let p = n; p && p !== root; p = p.parentElement) {
            const cs = getComputedStyle(p)
            if (cs.display === 'none' || cs.visibility === 'hidden') return false
          }
          return true
        }
        const nodes = [...root.querySelectorAll('[data-clf-node]')]
          .filter(shown)
          .map(r => ({
            label: r.getAttribute('data-clf-node'),
            x: +r.getAttribute('x'),
            y: +r.getAttribute('y'),
            w: +r.getAttribute('width'),
            h: +r.getAttribute('height'),
            lit: r.classList.contains('clf-node--lit'),
          }))
        const edges = [...root.querySelectorAll('[data-clf-edge]')]
          .filter(shown)
          .map(l => ({
            kind: l.getAttribute('data-clf-edge'),
            p: ['x1', 'y1', 'x2', 'y2'].map(a => +l.getAttribute(a)),
          }))
        const dd = [...root.querySelectorAll('.clf-readout dd')].map(d =>
          [...d.children]
            .filter(shown)
            .map(s => s.textContent.trim())
            .join(''),
        )
        return { nodes, edges, dd }
      })
    const owner = (nodes, x, y) =>
      nodes.find(
        n => x >= n.x - 0.6 && x <= n.x + n.w + 0.6 && y >= n.y - 0.6 && y <= n.y + n.h + 0.6,
      )
    // The preview's example: features 1 and 2 at L..L+4, feature 3 formed at L+2 and L+3 from
    // features 1 and 2 one layer earlier, persisting to L+4.
    const present = { 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4], 3: [2, 3, 4] }
    const want = new Set()
    for (const [f, ls] of Object.entries(present))
      for (const l of ls.slice(1))
        want.add(`${f}${'abcde'[ls.indexOf(l) - 1]}>${f}${'abcde'[ls.indexOf(l)]}`)
    for (const [l3, tag] of [
      [2, 'a'],
      [3, 'b'],
    ])
      for (const p of [1, 2]) want.add(`${p}${'abcde'[l3 - 1]}>3${tag}`)
    const cs = await readCirc()
    const drawn = new Set(
      cs.edges.map(e => {
        const a = owner(cs.nodes, e.p[0], e.p[1])
        const b = owner(cs.nodes, e.p[2], e.p[3])
        return `${a?.label}>${b?.label}`
      }),
    )
    c.check(
      'per-layer SAEs: 13 latents, as in the preview',
      cs.nodes.length === 13 &&
        cs.nodes
          .map(n => n.label)
          .sort()
          .join() === '1a,1b,1c,1d,1e,2a,2b,2c,2d,2e,3a,3b,3c',
      cs.nodes.map(n => n.label),
    )
    c.check(
      'per-layer SAEs: the 14 drawn edges are the 10 persistence and 4 formation edges',
      drawn.size === 14 && [...want].every(w => drawn.has(w)) && want.size === 14,
      { drawn: [...drawn], want: [...want] },
    )
    c.check(
      'per-layer readout counts equal the drawing',
      cs.dd.join() ===
        [
          cs.nodes.length,
          cs.edges.length,
          cs.edges.filter(e => e.kind === 'persist').length,
          cs.edges.filter(e => e.kind === 'form').length,
        ].join(),
      cs.dd,
    )
    c.check(
      'only feature 3 is salmon',
      cs.nodes.every(n => n.lit === n.label.startsWith('3')),
      cs.nodes.filter(n => n.lit).map(n => n.label),
    )
    const circEnds = await endpointsOnNodes(circ)
    c.check('per-layer circuit edges end on latents', circEnds.bad.length === 0, circEnds)

    await circ.locator('[data-clf-input][value="sae"]').focus()
    await page.keyboard.press('ArrowRight')
    await page.waitForTimeout(250)
    const cx = await readCirc()
    const drawnX = new Set(
      cx.edges.map(
        e => `${owner(cx.nodes, e.p[0], e.p[1])?.label}>${owner(cx.nodes, e.p[2], e.p[3])?.label}`,
      ),
    )
    c.check(
      'crosscoder (keyboard): 3 latents and the edges 1>3 and 2>3',
      cx.nodes.length === 3 && drawnX.size === 2 && drawnX.has('1>3') && drawnX.has('2>3'),
      { nodes: cx.nodes.map(n => n.label), edges: [...drawnX] },
    )
    c.check('crosscoder readout counts equal the drawing', cx.dd.join() === '3,2,0,2', cx.dd)
    const ccEnds = await endpointsOnNodes(circ)
    c.check('crosscoder circuit edges end on latents', ccEnds.bad.length === 0, ccEnds)
    await page
      .locator('.zoomable')
      .nth(1)
      .screenshot({ path: path.join(art, 'cross-layer-features', 'circuit-crosscoder.png') })
    await circ.locator('.clf-segment', { hasText: 'per-layer SAEs' }).click()
    await page.mouse.move(0, 0)
    await page.waitForTimeout(300)
  },
)

// ---------- CrosscoderMasks ----------
const N = 5
const K = 2
// Rules restated from the preview: which (read layer, reconstructed layer) pairs a coder allows.
const allowed = (variant, i, j) => {
  if (variant === 'sae') return i === j
  if (variant === 'transcoder') return j === i + 1
  const [causality, locality] = variant.split('-')
  if (locality === 'skip' && (i % 2 || j % 2)) return false
  if (locality === 'local' && Math.abs(i - j) > K - 1) return false
  if (causality === 'weak') return j >= i
  if (causality === 'strict') return j > i
  return true
}
const groupCount = variant => {
  if (variant === 'sae') return N
  if (variant === 'transcoder') return N - 1
  const [causality, locality] = variant.split('-')
  const used = locality === 'skip' ? Math.ceil(N / 2) : N
  if (causality === 'acausal') return locality === 'local' ? N - K + 1 : 1
  return causality === 'weak' ? used : used - 1
}

await run(
  'crosscoder-masks',
  'CrosscoderMasks',
  [{ dims: 'coder' }, { dims: 'variant', causality: 'weak', locality: 'global' }],
  async (page, c, out) => {
    const roots = page.locator('[data-crosscoder-masks]')
    const read = i =>
      roots.nth(i).evaluate(root => {
        const shown = n => {
          for (let p = n; p && p !== root; p = p.parentElement) {
            const cs = getComputedStyle(p)
            if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0')
              return false
          }
          return true
        }
        const cells = [...root.querySelectorAll('[data-ccm-cell]')].map(c =>
          c.getAttribute('data-state'),
        )
        const group = [...root.querySelectorAll('[data-ccm-variant-group]')].filter(shown)
        const boxes = group.flatMap(g => [...g.querySelectorAll('[data-ccm-box]')])
        const lit = boxes.filter(b => b.getAttribute('data-lit') === 'true')
        const stations = [...root.querySelectorAll('[data-ccm-station]')].map(s => ({
          x: +s.getAttribute('cx'),
          y: +s.getAttribute('cy'),
          r: +s.getAttribute('r'),
          used: s.getAttribute('data-used'),
        }))
        // Map each lit edge end to a station, and record whether it carries an arrowhead there.
        const touched = {}
        // Lit state comes from the attribute; computed opacity is mid-transition right after a click.
        const litEdges = group
          .flatMap(g => [...g.querySelectorAll('.ccm-edges')])
          .filter(e => e.getAttribute('data-lit') === 'true')
        for (const l of litEdges)
          for (const line of l.querySelectorAll('line')) {
            const ends = [
              [
                +line.getAttribute('x1'),
                +line.getAttribute('y1'),
                line.hasAttribute('marker-start'),
              ],
              [+line.getAttribute('x2'), +line.getAttribute('y2'), line.hasAttribute('marker-end')],
            ]
            const k = ends.findIndex(([x, y]) =>
              stations.some(s => Math.abs(Math.hypot(x - s.x, y - s.y) - s.r) < 0.6),
            )
            if (k < 0) continue
            const [x, y, head] = ends[k]
            const [, , boxHead] = ends[1 - k]
            const st = stations.findIndex(s => Math.abs(Math.hypot(x - s.x, y - s.y) - s.r) < 0.6)
            touched[st] = touched[st] ?? { write: false, read: false }
            touched[st].write ||= head
            touched[st].read ||= boxHead
          }
        const out = Object.fromEntries(
          [...root.querySelectorAll('[data-ccm-out]')].map(d => [
            d.getAttribute('data-ccm-out'),
            d.textContent,
          ]),
        )
        const rules = [...root.querySelectorAll('[data-ccm-rule]')]
          .filter(shown)
          .map(r => r.getAttribute('data-ccm-rule'))
        return {
          variant: root.getAttribute('data-ccm-variant'),
          cells,
          boxes: boxes.length,
          lit: lit.length,
          touched,
          used: stations.map(s => s.used).join(''),
          out,
          rules,
        }
      })

    const verify = async (i, variant, f, tag) => {
      const s = await read(i)
      const want = []
      for (let r = 0; r < N; r++)
        for (let q = 0; q < N; q++) want.push(allowed(variant, r, q) ? (r === f ? '2' : '1') : '0')
      const pairs = want.filter(x => x !== '0').length
      const writes = [...Array(N).keys()].filter(j => allowed(variant, f, j))
      const readsF = writes.length > 0
      // The drawing shows every edge of the groups that read f. A per-layer encoder reads f
      // alone; an acausal group reads and reconstructs its whole layer set or window.
      const [causality, locality] = variant.split('-')
      let drawnReads = readsF ? [f] : []
      let drawnWrites = writes
      if (causality === 'acausal') {
        const layers = [...Array(N).keys()]
        drawnReads =
          locality === 'local'
            ? layers.filter(l =>
                layers
                  .slice(0, N - K + 1)
                  .some(w => w <= f && f <= w + K - 1 && w <= l && l <= w + K - 1),
              )
            : readsF
              ? layers.filter(l => allowed(variant, l, l))
              : []
        drawnWrites = drawnReads
      }
      const stationsDrawn = new Set([...drawnReads, ...drawnWrites])
      const touchedOk =
        Object.keys(s.touched).length === stationsDrawn.size &&
        [...stationsDrawn].every(
          l =>
            s.touched[l] &&
            s.touched[l].read === drawnReads.includes(l) &&
            s.touched[l].write === drawnWrites.includes(l),
        )
      const used = [...Array(N).keys()]
        .map(l =>
          [...Array(N).keys()].some(j => allowed(variant, l, j) || allowed(variant, j, l))
            ? '1'
            : '0',
        )
        .join('')
      const ok =
        s.variant === variant &&
        s.cells.join('') === want.join('') &&
        s.boxes === groupCount(variant) &&
        s.out.groups === String(groupCount(variant)) &&
        s.out.pairs === `${pairs} of ${N * N}` &&
        s.out.reads === String(s.lit) &&
        (readsF ? s.lit > 0 : s.lit === 0) &&
        s.out.writes === (writes.length ? writes.map(j => `L${j}`).join(' ') : '–') &&
        touchedOk &&
        s.used === used &&
        s.rules.join() === variant
      return { ok, detail: { tag, variant, f, s, want: want.join(''), writes } }
    }

    const pick = async (i, name, value) => {
      await roots
        .nth(i)
        .locator(`.ccm-segment:has([data-ccm-input="${name}"][value="${value}"])`)
        .click()
    }

    // Instance 0: coder selector.
    const bad0 = []
    for (const v of ['sae', 'transcoder', 'acausal-global']) {
      await pick(0, 'coder', v)
      for (let f = 0; f < N; f++) {
        await pick(0, 'read', String(f))
        const r = await verify(0, v, f, 'coder')
        if (!r.ok) bad0.push(r.detail)
      }
      const ends = await endpointsOnNodes(roots.nth(0))
      if (ends.bad.length) bad0.push({ v, ends })
      await page.waitForTimeout(250)
      await page
        .locator('.zoomable')
        .nth(0)
        .screenshot({ path: path.join(out, `coder-${v}.png`) })
    }
    c.check(
      'coder figure: grid, boxes, edges, stations and readout match the rule for 3 coders x 5 read layers',
      bad0.length === 0,
      bad0.slice(0, 3),
    )

    // Instance 1: causality x locality.
    const bad1 = []
    const counts = {}
    for (const causality of ['acausal', 'weak', 'strict'])
      for (const locality of ['global', 'local', 'skip']) {
        const v = `${causality}-${locality}`
        await pick(1, 'causality', causality)
        await pick(1, 'locality', locality)
        for (let f = 0; f < N; f++) {
          await pick(1, 'read', String(f))
          const r = await verify(1, v, f, 'variant')
          if (!r.ok) bad1.push(r.detail)
        }
        counts[v] = (await read(1)).out.pairs
        const ends = await endpointsOnNodes(roots.nth(1))
        if (ends.bad.length) bad1.push({ v, ends })
        await page.waitForTimeout(250)
        await page
          .locator('.zoomable')
          .nth(1)
          .screenshot({ path: path.join(out, `variant-${v}.png`) })
      }
    c.check(
      'variant figure: grid, boxes, edges, stations and readout match the rule for 9 variants x 5 read layers',
      bad1.length === 0,
      bad1.slice(0, 3),
    )
    c.check(
      'pair counts: acausal 25/13/9, weakly causal 15/9/6, strictly causal 10/4/3',
      Object.values(counts)
        .map(p => p.split(' ')[0])
        .join() === '25,13,9,15,9,6,10,4,3',
      counts,
    )

    // Keyboard on the read-layer radio group, then restore the default.
    await pick(1, 'causality', 'weak')
    await pick(1, 'locality', 'global')
    await pick(1, 'read', '1')
    await roots.nth(1).locator('[data-ccm-input="read"][value="1"]').focus()
    await page.keyboard.press('ArrowRight')
    const k1 = await verify(1, 'weak-global', 2, 'keyboard')
    const ring = await roots
      .nth(1)
      .locator('.ccm-segment:has(input:focus-visible)')
      .evaluate(el => getComputedStyle(el).outlineStyle)
      .catch(() => 'none')
    await page.keyboard.press('ArrowLeft')
    const k2 = await verify(1, 'weak-global', 1, 'keyboard')
    c.check(
      'ArrowRight and ArrowLeft move the read layer with visible focus',
      k1.ok && k2.ok && ring === 'solid',
      { k1: k1.ok, k2: k2.ok, ring },
    )
    await roots.nth(1).locator('[data-ccm-input="causality"][value="weak"]').focus()
    await page.keyboard.press('ArrowRight')
    const k3 = await verify(1, 'strict-global', 1, 'keyboard')
    await page.keyboard.press('ArrowLeft')
    c.check('ArrowRight on causality selects strictly causal', k3.ok, k3.detail)
    await pick(0, 'coder', 'acausal-global')
    await pick(0, 'read', '1')
    await page.mouse.move(0, 0)
    await page.waitForTimeout(300)
  },
)

// ---------- note fences ----------
{
  const out = path.join(art, 'notes')
  const c = checker(out)
  await mkdir(out, { recursive: true })
  const parse = unified().use(remarkParse).use(remarkMdx).use(remarkMath)
  const note = 'content/thoughts/sparse crosscoders.md'
  const names = ['CrossLayerFeatures', 'CrossLayerFeatures', 'CrosscoderMasks', 'CrosscoderMasks']
  const src = await readFile(path.join(repo, note), 'utf8')
  const fences = [...src.matchAll(/```jsx imports=\{([^}]*)\}\n([\s\S]*?)```/g)].filter(m =>
    names.some(n => m[1].includes(n)),
  )
  const found = []
  for (const m of fences) {
    const tree = parse.parse(m[2])
    const walk = n =>
      n.type?.startsWith('mdxJsx') && names.includes(n.name)
        ? [n]
        : (n.children ?? []).flatMap(walk)
    for (const n of walk(tree)) {
      const cap = n.attributes.find(a => a.name === 'caption')
      if (typeof cap?.value === 'string' && /after Anthropic's crosscoders/.test(cap.value))
        found.push(n.name)
    }
  }
  c.check(
    'four fences parse as MDX with an attributed caption',
    found.join() === names.join(),
    found,
  )
  const left = [
    'additive-residual-stream-llm.webp',
    'feature-neurons.webp',
    'one-step-circuit.webp',
    'parallel-joint-branch.webp',
  ]
  const gone = [
    'proposal-formed-by-cross-layers-superposition.webp',
    'dedup-features-persistent.webp',
    'crosscoder-setup.webp',
    'crosscoders-variants.webp',
  ]
  c.check(
    'four images replaced, four kept, DataChart embed unchanged',
    left.every(f => src.includes(f)) &&
      gone.every(f => !src.includes(f)) &&
      src.includes(
        '<Zoomable label="crosscoder and per-layer SAE loss against number of features">',
      ),
  )
  console.log('[notes]')
  if (!(await c.finish())) failed = true
}

process.exit(failed ? 1 : 0)
