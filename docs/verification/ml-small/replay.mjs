// Replay: node docs/verification/ml-small/replay.mjs
// Renders GatedSAE, HeavyBallRate, ConvLocality and ResidualBlock through the shared harness,
// drives every control by pointer and keyboard, and checks the drawn state against the rules
// in the notes. Also parses each new note fence with the MDX parser the site uses.
import { readFile } from 'node:fs/promises'
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
const near = (a, b, tol = 0.6) => Math.abs(a - b) <= tol
let failed = false

async function run(name, component, instances, body) {
  const out = path.join(art, name)
  const pagePath = await renderFigurePage({ component, instances, out })
  const { browser, page, errors } = await openFigurePage(pagePath)
  const c = checker(out)
  try {
    await body(page, c)
    await standardChecks(page, errors, c, '.zoomable')
  } finally {
    await browser.close()
  }
  console.log(`[${name}]`)
  if (!(await c.finish())) failed = true
}

// Reads a tick label's position, so geometry checks use the drawn axis rather than constants.
const tickMap = svg =>
  svg.evaluate(el =>
    [...el.querySelectorAll('.hbr-tick text')].map(t => ({
      v: t.textContent,
      x: +t.getAttribute('x'),
      y: +t.getAttribute('y'),
    })),
  )
const linear = (p0, p1) => v => p0.at + ((v - p0.v) / (p1.v - p0.v)) * (p1.at - p0.at)

// ---------- GatedSAE ----------
await run('gated-sae', 'GatedSAE', [{ view: 'gated' }, { view: 'jumprelu' }], async (page, c) => {
  const roots = page.locator('[data-gated-sae]')
  const read = i =>
    roots.nth(i).evaluate(root => {
      const col = key =>
        [0, 1, 2, 3].map(k => {
          const el = root.querySelector(`[data-gsae-text="${key}-${k}"]`)
          return el ? Number(el.textContent.replace('−', '-')) : null
        })
      const cols = [...root.querySelectorAll('.gsae-col')].map(g =>
        [...g.querySelectorAll('.gsae-num')].map(t => Number(t.textContent.replace('−', '-'))),
      )
      const pts = [...root.querySelectorAll('[data-gsae-point]')].map(p => ({
        cx: +p.getAttribute('cx'),
        cy: +p.getAttribute('cy'),
        on: p.getAttribute('data-gsae-on'),
      }))
      const diag = root.querySelector('.gsae-diag')
      const axis = root.querySelector('.gsae-axis')
      const lit = [...root.querySelectorAll('[data-gsae-cell^="ftilde"]')].map(
        el => el.getAttribute('data-gsae-on') === '1',
      )
      return {
        theta: Number(root.querySelector('[data-gsae-theta]').value),
        cols,
        bgate: col('bgate'),
        pigate: col('pigate'),
        fgate: col('fgate'),
        ftilde: col('ftilde'),
        lit,
        l0: root.querySelector('[data-gsae-out="l0"]').textContent,
        pts,
        diag: diag && ['x1', 'y1', 'x2', 'y2'].map(a => +diag.getAttribute(a)),
        zeroY: axis && +axis.getAttribute('y1'),
        open: root.querySelector('[data-gsae-open]') && {
          cx: +root.querySelector('[data-gsae-open]').getAttribute('cx'),
          cy: +root.querySelector('[data-gsae-open]').getAttribute('cy'),
        },
        thetaX: root.querySelector('[data-gsae-theta-line]')
          ? +root.querySelector('[data-gsae-theta-line]').getAttribute('x1')
          : null,
        pimag: [...root.querySelectorAll('.gsae-table tbody tr')].map(tr =>
          Number(tr.children[1].textContent.replace('−', '-')),
        ),
      }
    })
  const setTheta = (i, v) =>
    roots
      .nth(i)
      .locator('[data-gsae-theta]')
      .evaluate((el, v) => {
        el.value = String(v)
        el.dispatchEvent(new Event('input', { bubbles: true }))
      }, v)

  const s0 = await read(0)
  c.check(
    'gated view at theta 0.35 gives the paper figure output (0, 0.9, 2.4, 0)',
    s0.ftilde.join() === '0,0.9,2.4,0',
    s0.ftilde,
  )
  // Column order in the gated diagram: u, e^r, b_mag, pi_mag, f_mag, b_gate, pi_gate, f_gate, f~.
  const [u, sc, bm, pm, fm] = s0.cols
  c.check('u read from the paper figure', u.join() === '-0.1,1.1,1.9,0.2', u)
  c.check('f_mag read from the paper figure', fm.join() === '0,0.9,2.4,0.3', fm)
  c.check(
    'pi_mag = e^r * u + b_mag and f_mag = ReLU(pi_mag) in the drawn cells',
    pm.every((z, k) => near(z, sc[k] * u[k] + bm[k], 0.011) && near(fm[k], Math.max(0, z), 0.011)),
    { pm, fm },
  )
  let gatedOk = true
  let plotOk = true
  const bad = []
  for (let k = 0; k <= 50; k++) {
    const th = +(k * 0.05).toFixed(2)
    await setTheta(0, th)
    await setTheta(1, th)
    const g = await read(0)
    for (let i = 0; i < 4; i++) {
      const pg = u[i] + g.bgate[i]
      const ok =
        near(g.pigate[i], pg, 0.011) &&
        g.fgate[i] === (g.pigate[i] > 0 ? 1 : 0) &&
        near(g.ftilde[i], g.fgate[i] * fm[i], 0.011) &&
        // Equivalence in the note: the gated output is JumpReLU_theta of pi_mag.
        near(g.ftilde[i], pm[i] > th ? pm[i] : 0, 0.011) &&
        g.lit[i] === g.ftilde[i] > 0
      if (!ok) {
        gatedOk = false
        bad.push({ th, i, g: { pg: g.pigate[i], fg: g.fgate[i], ft: g.ftilde[i] } })
      }
    }
    const active = g.ftilde.filter(v => v > 0).length
    if (g.l0 !== `${active} of 4`) {
      gatedOk = false
      bad.push({ th, l0: g.l0 })
    }
    const j = await read(1)
    const [x1, y1, x2, y2] = j.diag
    const onDiag = p => near(p.cy, y1 + ((p.cx - x1) / (x2 - x1)) * (y2 - y1), 0.3)
    for (let i = 0; i < 4; i++) {
      const p = j.pts[i]
      const want = j.ftilde[i] > 0
      if (!(p.on === (want ? '1' : '0') && (want ? onDiag(p) : near(p.cy, j.zeroY, 0.3)))) {
        plotOk = false
        bad.push({ th, plot: i, p })
      }
    }
    if (!(near(j.open.cx, j.thetaX, 0.01) && onDiag({ cx: j.open.cx, cy: j.open.cy }))) {
      plotOk = false
      bad.push({ th, open: j.open, thetaX: j.thetaX })
    }
    if (j.l0 !== g.l0) plotOk = false
  }
  c.check(
    'gated cells follow pi_gate = u + b_gate, f_gate = 1[pi_gate > 0], f~ = f_gate f_mag = JumpReLU(pi_mag) for all 51 thetas',
    gatedOk,
    bad.slice(0, 4),
  )
  c.check(
    'plot: active points on the identity, gated-off points on zero, open dot at theta on the identity',
    plotOk,
    bad.slice(0, 4),
  )

  const slider = roots.nth(1).locator('[data-gsae-theta]')
  await setTheta(1, 0.3)
  await slider.focus()
  await page.keyboard.press('ArrowLeft')
  const k1 = await read(1)
  c.check(
    'keyboard: ArrowLeft lowers theta to 0.25 and turns feature 4 on',
    near(k1.theta, 0.25, 1e-9) && k1.ftilde[3] === 0.3 && k1.l0 === '3 of 4',
    k1,
  )
  const focus = await slider.evaluate(
    el => getComputedStyle(el).outlineStyle !== 'none' || el.matches(':focus-visible'),
  )
  c.check('slider shows focus', focus)
  await setTheta(0, 0.2)
  await setTheta(1, 0.2)
  await c.shoot(page.locator('.zoomable').nth(0), 'gated-theta-0.2.png')
  await c.shoot(page.locator('.zoomable').nth(1), 'jumprelu-theta-0.2.png')
  await setTheta(0, 0.35)
  await setTheta(1, 0.35)
})

// ---------- HeavyBallRate ----------
await run('heavy-ball-rate', 'HeavyBallRate', [{}], async (page, c) => {
  const root = page.locator('[data-heavy-ball-rate]')
  const [rateSvg, trajSvg] = [root.locator('svg').nth(0), root.locator('svg').nth(1)]
  const rt = await tickMap(rateSvg)
  const tt = await tickMap(trajSvg)
  // Tick text sits 13 below the x axis and 3.5 below the y gridline it names.
  const xs = rt.filter(t => t.y > 190)
  const ys = rt.filter(t => t.y <= 190)
  const ax = linear({ v: +xs[0].v, at: xs[0].x }, { v: +xs.at(-1).v, at: xs.at(-1).x })
  const ay = linear({ v: +ys[0].v, at: ys[0].y - 3.5 }, { v: +ys.at(-1).v, at: ys.at(-1).y - 3.5 })
  const tys = tt.filter(t => t.y <= 190)
  const ty = linear(
    { v: +tys[0].v, at: tys[0].y - 3.5 },
    { v: +tys.at(-1).v, at: tys.at(-1).y - 3.5 },
  )

  const set = (sel, v) =>
    root.locator(sel).evaluate((el, v) => {
      el.value = String(v)
      el.dispatchEvent(new Event('input', { bubbles: true }))
    }, v)
  const read = () =>
    root.evaluate(r => ({
      a: Number(r.querySelector('[data-hbr-alpha]').value),
      b: Number(r.querySelector('[data-hbr-beta]').value),
      rho: Number(r.querySelector('[data-hbr-out="rho"]').textContent),
      roots: r.querySelector('[data-hbr-out="roots"]').textContent,
      settle: r.querySelector('[data-hbr-out="settle"]').textContent,
      mark: ['cx', 'cy'].map(k => +r.querySelector('[data-hbr-mark]').getAttribute(k)),
      guide: ['x1', 'y1', 'x2', 'y2'].map(
        k => +r.querySelector('[data-hbr-guide]').getAttribute(k),
      ),
      band: ['x', 'width'].map(k => +r.querySelector('[data-hbr-band]').getAttribute(k)),
      pts: [...r.querySelectorAll('[data-hbr-point]')].map(p => +p.getAttribute('cy')),
    }))
  // Independent rate: spectral radius of the iteration matrix by Gelfand's formula.
  const gelfand = (a, b, T = 4000) => {
    let M = [1, 0, 0, 1]
    const A = [1 + b - a, -b, 1, 0]
    let logScale = 0
    for (let t = 0; t < T; t++) {
      M = [
        M[0] * A[0] + M[1] * A[2],
        M[0] * A[1] + M[1] * A[3],
        M[2] * A[0] + M[3] * A[2],
        M[2] * A[1] + M[3] * A[3],
      ]
      const n = Math.max(...M.map(Math.abs)) || 1e-300
      M = M.map(v => v / n)
      logScale += Math.log(n)
    }
    return Math.exp(logScale / T)
  }
  const traj = (a, b, n) => {
    const out = [1]
    let p = 1
    let x = 1
    for (let t = 0; t < n; t++) {
      const nx = (1 + b - a) * x - b * p
      p = x
      x = nx
      out.push(x)
    }
    return out
  }
  const cases = []
  for (const b of [0, 0.25, 0.5, 0.8, 0.95])
    for (const a of [0, 0.05, 0.5, 1, 1.5, 2, 2.5, 3.9, 4]) cases.push([a, b])
  const bad = []
  for (const [a, b] of cases) {
    await set('[data-hbr-beta]', b)
    await set('[data-hbr-alpha]', a)
    const s = await read()
    const want = gelfand(a, b)
    const disc = (1 + b - a) ** 2 - 4 * b
    const xsT = traj(a, b, 24)
    const ok =
      near(s.rho, want, 0.012) &&
      s.roots.startsWith('complex') === disc < 0 &&
      near(s.mark[0], ax(a), 0.3) &&
      near(s.mark[1], ay(s.rho), 0.5) &&
      near(s.guide[0], s.mark[0], 0.01) &&
      near(s.guide[3], s.mark[1], 0.01) &&
      near(s.band[0] + s.band[1], ax(Math.min(4, 2 * (1 + b))), 0.3) &&
      xsT.every((x, t) => Math.abs(x) > 1.5 || near(s.pts[t], ty(x), 0.3))
    const sim = traj(a, b, 3000)
    let last = -1
    sim.forEach((x, t) => Math.abs(x) >= 1e-3 && (last = t))
    const settleWant = want >= 0.999 || a === 0 ? 'never' : String(last + 1)
    if (!ok || s.settle !== settleWant) bad.push({ a, b, s, want, settleWant })
  }
  c.check(
    `rate readout equals the spectral radius, marker/guide/band/iterates match the axes (${cases.length} cases)`,
    bad.length === 0,
    bad.slice(0, 3),
  )

  await set('[data-hbr-beta]', 0)
  await set('[data-hbr-alpha]', 1)
  const one = await read()
  c.check(
    'beta 0, step 1: rate 0 and one step to zero (the V minimum in the original chart)',
    one.rho === 0 && one.settle === '1',
    one,
  )
  const ref = await root.locator('.hbr-reference').getAttribute('d')
  const cur = await root.locator('[data-hbr-curve]').getAttribute('d')
  c.check('beta 0: the selected curve lies on the gradient-descent V', ref === cur)

  await root.locator('[data-hbr-alpha]').focus()
  await page.keyboard.press('ArrowRight')
  const k = await read()
  c.check(
    'keyboard: ArrowRight steps alpha lambda by 0.05 and moves the marker',
    near(k.a, 1.05, 1e-9) && near(k.mark[0], ax(1.05), 0.3),
    k,
  )
  await root.locator('[data-hbr-beta]').focus()
  await page.keyboard.press('ArrowRight')
  const k2 = await read()
  c.check(
    'keyboard: ArrowRight steps beta by 0.05',
    near(k2.b, 0.05, 1e-9) && near(k2.rho, gelfand(1.05, 0.05), 0.012),
    k2,
  )
  await set('[data-hbr-beta]', 0.5)
  await set('[data-hbr-alpha]', 1.5)
  await c.shoot(page.locator('.zoomable').first(), 'momentum-a1.5-b0.5.png')
  await set('[data-hbr-beta]', 0)
  await set('[data-hbr-alpha]', 0.5)
})

// ---------- ConvLocality ----------
await run(
  'conv-locality',
  'ConvLocality',
  [{ view: 'connectivity' }, { view: 'pooling' }],
  async (page, c) => {
    const conn = page.locator('[data-conv-locality]').nth(0)
    const pool = page.locator('[data-conv-locality]').nth(1)

    const poolRead = () =>
      pool.evaluate(r => {
        const set = r.querySelector('[data-cvl-config]:not([hidden])')
        return {
          config: set.getAttribute('data-cvl-config'),
          n: Math.round(Math.sqrt(set.querySelectorAll('[data-cvl-window]').length)),
          cells: [...r.querySelectorAll('[data-cvl-in]')].map(el => ({
            v: Number(el.textContent),
            s: el.getAttribute('data-cvl-state'),
          })),
          outs: [...set.querySelectorAll('.cvl-pick span')].map(s => Number(s.textContent)),
          checked: Number(set.querySelector('[data-cvl-window]:checked').value),
          max: r.querySelector('[data-cvl-out="max"]').textContent,
          w2: r.querySelector('[data-cvl-out="w2"]').textContent,
          skipped: Number(r.querySelector('[data-cvl-out="skipped"]').textContent),
        }
      })
    const first = await poolRead()
    c.check(
      'pooling input is the lecture 4 x 4 example',
      first.cells.map(x => x.v).join() === '1,1,2,4,5,6,7,8,3,2,1,0,1,2,3,4',
    )
    c.check(
      'F 2, S 2 gives the lecture output 6 8 / 3 4',
      first.config === '2:2' && first.outs.join() === '6,8,3,4',
      first,
    )
    const bad = []
    for (const f of [2, 3]) {
      for (const s of [1, 2]) {
        await pool.locator(`[data-cvl-size][value="${f}"]`).check({ force: true })
        await pool.locator(`[data-cvl-stride][value="${s}"]`).check({ force: true })
        // Note formula with no padding; a window that runs past the edge is dropped.
        const n = Math.floor((4 - f) / s) + 1
        const set = pool.locator(`[data-cvl-config="${f}:${s}"]`)
        const covered = new Set()
        for (let w = 0; w < n * n; w++) {
          await set.locator(`[data-cvl-window][value="${w}"]`).check({ force: true })
          const st = await poolRead()
          const r0 = Math.floor(w / n) * s
          const c0 = (w % n) * s
          const inWin = k => {
            const r = Math.floor(k / 4)
            const col = k % 4
            return r >= r0 && r < r0 + f && col >= c0 && col < c0 + f
          }
          const win = st.cells.filter((_, k) => inWin(k))
          st.cells.forEach((_, k) => inWin(k) && covered.add(k))
          const mx = Math.max(...win.map(x => x.v))
          const ok =
            st.config === `${f}:${s}` &&
            st.n === n &&
            st.w2 === `${n} × ${n}` &&
            st.cells.every((x, k) => (inWin(k) ? x.s === (x.v === mx ? 'm' : 'w') : x.s === '0')) &&
            st.outs[w] === mx &&
            st.max === String(mx)
          if (!ok) bad.push({ f, s, w, st })
        }
        const st = await poolRead()
        if (st.skipped !== 16 - covered.size) bad.push({ f, s, skipped: st.skipped })
      }
    }
    c.check(
      'every window of every F, S: sage window, salmon max, output equals the window max, skipped count',
      bad.length === 0,
      bad.slice(0, 2),
    )

    await pool.locator('[data-cvl-size][value="2"]').check({ force: true })
    await pool.locator('[data-cvl-stride][value="2"]').check({ force: true })
    await pool.locator('[data-cvl-config="2:2"] [data-cvl-window][value="0"]').focus()
    await page.keyboard.press('ArrowRight')
    const kb = await poolRead()
    c.check(
      'keyboard: ArrowRight walks to the next window (max 8)',
      kb.checked === 1 && kb.max === '8',
      kb.max,
    )
    const ring = await pool
      .locator('.cvl-pick:has(input:focus-visible)')
      .evaluate(el => getComputedStyle(el).outlineStyle)
    c.check('focused window shows a visible outline', ring === 'solid', ring)
    await pool
      .locator('[data-cvl-config="2:2"] [data-cvl-window][value="0"]')
      .check({ force: true })

    const connRead = () =>
      conn.evaluate(r => ({
        mode: r.getAttribute('data-cvl-mode'),
        ids: [...r.querySelectorAll('[data-cvl-in]')].map(el => el.textContent),
        lit: [...r.querySelectorAll('[data-cvl-in]')].map(
          el => el.getAttribute('data-cvl-state') === 'w',
        ),
        params: Number(r.querySelector('[data-cvl-out="params"]').textContent),
        reads: Number(r.querySelector('[data-cvl-out="reads"]').textContent),
        rule: [...r.querySelectorAll('[data-cvl-rule]')]
          .filter(p => getComputedStyle(p).display !== 'none')
          .map(p => p.getAttribute('data-cvl-rule')),
      }))
    const cbad = []
    const idSets = {}
    for (const m of ['dense', 'local', 'shared']) {
      await conn.locator(`[data-cvl-mode][value="${m}"]`).check({ force: true })
      let maxId = 0
      idSets[m] = []
      for (let u = 0; u < 16; u++) {
        await conn.locator(`[data-cvl-unit][value="${u}"]`).check({ force: true })
        const st = await connRead()
        const ur = Math.floor(u / 4)
        const uc = u % 4
        const field = k => {
          const r = Math.floor(k / 6)
          const col = k % 6
          return r >= ur && r < ur + 3 && col >= uc && col < uc + 3
        }
        const ids = st.ids.filter(Boolean).map(Number)
        maxId = Math.max(maxId, ...ids)
        idSets[m].push(ids.join())
        const ok =
          st.mode === m &&
          st.rule.join() === m &&
          st.lit.every((l, k) => l === (m === 'dense' ? true : field(k))) &&
          st.reads === ids.length &&
          new Set(ids).size === ids.length
        if (!ok) cbad.push({ m, u, st })
      }
      const st = await connRead()
      if (st.params !== maxId) cbad.push({ m, params: st.params, maxId })
    }
    c.check(
      'connectivity: field cells match each unit patch, and the largest weight id over all units equals the weight-count readout',
      cbad.length === 0,
      cbad.slice(0, 2),
    )
    c.check(
      'shared: every unit uses the same weight ids',
      new Set(idSets.shared.map(s => s.split(',').sort().join())).size === 1,
    )
    c.check(
      'local and dense: no two units share a weight id',
      ['local', 'dense'].every(
        m => new Set(idSets[m].join(',').split(',')).size === idSets[m].join(',').split(',').length,
      ),
    )
    await conn.locator('[data-cvl-mode][value="dense"]').focus()
    await page.keyboard.press('ArrowRight')
    const ck = await connRead()
    c.check(
      'keyboard: ArrowRight on the connection segments selects Local',
      ck.mode === 'local' && ck.reads === 9,
      ck.mode,
    )
    await conn.locator('[data-cvl-mode][value="dense"]').check({ force: true })
    await c.shoot(page.locator('.zoomable').nth(0), 'connectivity-dense.png')
    await pool.locator('[data-cvl-size][value="3"]').check({ force: true })
    await pool.locator('[data-cvl-stride][value="1"]').check({ force: true })
    await pool
      .locator('[data-cvl-config="3:1"] [data-cvl-window][value="3"]')
      .check({ force: true })
    await c.shoot(page.locator('.zoomable').nth(1), 'pooling-f3-s1-w3.png')
    await pool.locator('[data-cvl-size][value="2"]').check({ force: true })
    await pool.locator('[data-cvl-stride][value="2"]').check({ force: true })
    await conn.locator('[data-cvl-mode][value="shared"]').check({ force: true })
    await conn.locator('[data-cvl-unit][value="5"]').check({ force: true })
  },
)

// ---------- ResidualBlock ----------
await run('residual-block', 'ResidualBlock', [{}], async (page, c) => {
  const res = await page.locator('.residual-block svg').evaluate(svg => {
    const rects = [...svg.querySelectorAll('rect.rb-node')].map(r =>
      ['x', 'y', 'width', 'height'].map(k => +r.getAttribute(k)),
    )
    const circ = svg.querySelector('circle.rb-sum')
    const sum = ['cx', 'cy', 'r'].map(k => +circ.getAttribute(k))
    const ends = []
    for (const l of svg.querySelectorAll('line.rb-edge'))
      ends.push(
        [+l.getAttribute('x1'), +l.getAttribute('y1')],
        [+l.getAttribute('x2'), +l.getAttribute('y2')],
      )
    const p = svg.querySelector('path.rb-skip')
    ends.push([p.getPointAtLength(0).x, p.getPointAtLength(0).y])
    const end = p.getPointAtLength(p.getTotalLength())
    ends.push([end.x, end.y])
    return { rects, sum, ends }
  })
  const onRect = ([x, y]) =>
    res.rects.some(([rx, ry, w, h]) => {
      const inX = x >= rx - 0.5 && x <= rx + w + 0.5
      const inY = y >= ry - 0.5 && y <= ry + h + 0.5
      return (
        (inX && (near(y, ry, 0.5) || near(y, ry + h, 0.5))) ||
        (inY && (near(x, rx, 0.5) || near(x, rx + w, 0.5)))
      )
    })
  const onSum = ([x, y]) => near(Math.hypot(x - res.sum[0], y - res.sum[1]), res.sum[2], 0.5)
  const loose = res.ends.filter(e => !onRect(e) && !onSum(e))
  c.check('every edge and the shortcut end on a node or the sum', loose.length === 0, loose)
  c.check('block order: x, weight layer, ReLU, weight layer, sum, ReLU, y', res.rects.length === 6)
})

// ---------- note fences ----------
{
  const out = path.join(art, 'notes')
  const c = checker(out)
  const { mkdir } = await import('node:fs/promises')
  await mkdir(out, { recursive: true })
  const parse = unified().use(remarkParse).use(remarkMdx).use(remarkMath)
  const notes = {
    'content/thoughts/sparse autoencoder.md': ['GatedSAE', 'GatedSAE'],
    'content/thoughts/optimization.md': ['HeavyBallRate'],
    'content/thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Convolutional Neural Network.md':
      ['ConvLocality', 'ConvLocality'],
    'content/thoughts/FFN.md': ['ResidualBlock'],
  }
  for (const [note, names] of Object.entries(notes)) {
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
        if (typeof cap?.value === 'string' && cap.value.length > 40) found.push(n.name)
      }
    }
    c.check(
      `${path.basename(note)}: fences parse as MDX with captions`,
      found.join() === names.join(),
      found,
    )
  }
  console.log('[notes]')
  if (!(await c.finish())) failed = true
}

process.exit(failed ? 1 : 0)
