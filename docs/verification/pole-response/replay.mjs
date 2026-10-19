// Renders PoleResponse with the props each note embeds, drives every control in Chromium and checks
// that the drawn marks equal the readout. Writes pages, screenshots and results.json to ./artifacts.
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { checker, openFigurePage, renderFigurePage, standardChecks } from '../figure-harness.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const art = path.join(here, 'artifacts')

const NOTES = {
  // content/thoughts/university/twenty-four-twenty-five/sfwr-4aa4/system response.md
  'system-response': [
    { mode: 'first', a: 1, caption: 'first order' },
    { re: 0, im: Math.PI, lock: 'sigma', caption: 'undamped' },
    { preset: 'underdamped', cases: true, regions: true, caption: 'regions' },
    { preset: 'underdamped', caption: 'peak' },
    { re: -1, im: 3, locks: true, caption: 'relations' },
  ],
  // content/thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Time response.md
  'time-response': [
    { mode: 'first', a: 1, caption: 'time constant' },
    { preset: 'underdamped', cases: true, caption: 'course cases' },
  ],
  // content/thoughts/university/twenty-three-twenty-four/sfwr-3dx4/Root locus control.md
  'root-locus-control': [
    { mode: 'locus', comp: 'none', caption: 'gain only' },
    { mode: 'locus', comp: 'integrator', compensators: true, caption: 'integrator' },
    { mode: 'locus', comp: 'pi', compensators: true, caption: 'pi' },
  ],
  // content/thoughts/Z-transform.md
  'z-transform': [{ re: -2, im: 2, regions: true, caption: 'stability' }],
}

const minus = t => Number(t.replace(/−/g, '-').replace(/[^\d.-]/g, ''))
const near = (a, b, tol = 0.011) => Math.abs(a - b) <= tol

const snap = (pg, i) =>
  pg.evaluate(idx => {
    const root = document.querySelectorAll('[data-pole-response]')[idx]
    const out = Object.fromEntries(
      [...root.querySelectorAll('[data-prs-out]')].map(d => [d.dataset.prsOut, d.textContent]),
    )
    const el = k => root.querySelector(`[data-prs-el="${k}"]`)
    const xy = k => {
      const m = /translate\(([-\d.]+) ([-\d.]+)\)/.exec(el(k)?.getAttribute('transform') ?? '')
      return m ? [Number(m[1]), Number(m[2])] : null
    }
    const attr = (k, a) => el(k)?.getAttribute(a)
    const visible = k => attr(k, 'visibility') !== 'hidden'
    const curve = attr('curve', 'd') ?? ''
    const pts = [...curve.matchAll(/[ML]([-\d.]+) ([-\d.]+)/g)].map(m => [
      Number(m[1]),
      Number(m[2]),
    ])
    const checked = [...root.querySelectorAll('input[type=radio]:checked')].map(r => r.value)
    const closed = [...root.querySelectorAll('[data-prs-cl]')]
      .filter(c => c.getAttribute('visibility') !== 'hidden')
      .map(c => [Number(c.getAttribute('cx')), Number(c.getAttribute('cy'))])
    const locus = [...root.querySelectorAll('[data-prs-locus]')]
      .filter(g => getComputedStyle(g).display !== 'none')
      .map(g => g.dataset.prsLocus)
    return {
      out,
      h0: xy('h0'),
      h1: visible('h1') ? xy('h1') : null,
      conj: visible('conj') ? xy('conj') : null,
      ref0: visible('ref0') ? xy('ref0') : null,
      peak: visible('peak') ? [Number(attr('peak', 'cx')), Number(attr('peak', 'cy'))] : null,
      ts: visible('ts') ? [Number(attr('ts', 'cx')), Number(attr('ts', 'cy'))] : null,
      tau: visible('tau') ? [Number(attr('tau', 'cx')), Number(attr('tau', 'cy'))] : null,
      pts,
      checked,
      closed,
      locus,
      caseAttr: root.dataset.prsCase,
      k: root.dataset.prsK,
    }
  }, i)

// Plane and plot geometry, from pole-response-model.ts (UNIT 18, PLANE_X 24, PLOT_X 30, tops 22).
const TIME = { re0: -9, im1: 7, t1: 8, c1: 2, c0: 0 }
const LOCUS = { re0: -13, im1: 7, t1: 12, c1: 2, c0: 0 }
const toS = (g, [x, y]) => [(x - 24) / 18 + g.re0, g.im1 - (y - 22) / 18]
const toT = (g, [x, y]) => [((x - 30) / 300) * g.t1, g.c1 - ((y - 22) / 252) * (g.c1 - g.c0)]
const yOnCurve = (pts, x) =>
  pts.reduce((b, p) => (Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b))[1]

const parsePair = t => {
  const m = /([−\d.]+) ± j([\d.]+)/.exec(t)
  return m ? [minus(m[1]), Number(m[2])] : null
}

// Drawn time-figure state must equal its readout.
const timeConsistent = (c, s, label) => {
  const g = TIME
  const [re, im] = toS(g, s.h0)
  const pair = parsePair(s.out.poles ?? '')
  if (pair) {
    c.check(
      `${label}: handle at readout pole`,
      near(re, pair[0], 0.006) && near(im, pair[1], 0.006),
      { re, im, pair },
    )
    if (s.conj)
      c.check(`${label}: conjugate mirrors`, near(toS(g, s.conj)[1], -pair[1], 0.006), s.conj)
  }
  if (s.peak) {
    const [t, v] = toT(g, s.peak)
    c.check(`${label}: peak dot at T_p`, near(t, minus(s.out.tp), 0.03), { t, tp: s.out.tp })
    const os = minus(s.out.os)
    c.check(`${label}: peak height is 1 + %OS`, near(v, 1 + os / 100, 0.01), { v, os })
    c.check(
      `${label}: peak dot on the curve`,
      Math.abs(yOnCurve(s.pts, s.peak[0]) - s.peak[1]) < 1.5,
      s.peak,
    )
  }
  if (s.ts) {
    const [t] = toT(g, s.ts)
    c.check(`${label}: settling dot at T_s`, near(t, minus(s.out.tsBand), 0.03), {
      t,
      ts: s.out.tsBand,
    })
    c.check(
      `${label}: settling dot on the curve`,
      Math.abs(yOnCurve(s.pts, s.ts[0]) - s.ts[1]) < 1.5,
      s.ts,
    )
  }
  c.check(
    `${label}: no T_s mark when readout has none`,
    (s.out.tsBand === '–') === !s.ts,
    s.out.tsBand,
  )
}

const screenOf = (pg, i, re, im, scale = TIME) =>
  pg.evaluate(
    ([idx, x, y]) => {
      const svg = document
        .querySelectorAll('[data-pole-response]')
        [idx].querySelector('[data-prs-plane]')
      const p = new DOMPoint(x, y).matrixTransform(svg.getScreenCTM())
      return [p.x, p.y]
    },
    [i, 24 + (re - scale.re0) * 18, 22 + (scale.im1 - im) * 18],
  )

const dragS = async (pg, i, from, to, scale = TIME) => {
  await pg.locator('[data-prs-plane]').nth(i).scrollIntoViewIfNeeded()
  const a = await screenOf(pg, i, ...from, scale)
  const b = await screenOf(pg, i, ...to, scale)
  await pg.mouse.move(...a)
  await pg.mouse.down()
  await pg.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, { steps: 4 })
  await pg.mouse.move(...b, { steps: 4 })
  await pg.mouse.up()
}

const roots = i => `[data-pole-response] >> nth=${i}`

let ok = true
for (const [note, instances] of Object.entries(NOTES)) {
  const out = path.join(art, note)
  const pagePath = await renderFigurePage({
    component: 'PoleResponse',
    instances,
    out,
    label: note,
  })
  const { browser, page: pg, errors } = await openFigurePage(pagePath)
  const c = checker(out)

  if (note === 'system-response' || note === 'time-response') {
    for (let i = 0; i < instances.length; i++) {
      const s = await snap(pg, i)
      if (instances[i].mode === 'first') {
        c.check(`#${i} first order T_r = ln 9`, s.out.tr === '2.20 s', s.out.tr)
        c.check(`#${i} first order T_s band = -ln 0.02`, s.out.tsBand === '3.91 s', s.out.tsBand)
        c.check(
          `#${i} tau mark at 63.2%`,
          near(toT(TIME, s.tau)[0], 1, 0.02) && near(toT(TIME, s.tau)[1], 0.632, 0.01),
          s.tau,
        )
        // Keyboard: move the real pole left by 1, so a = 2.
        await pg.locator(roots(i)).locator('[data-prs-handle="0"]').focus()
        for (let n = 0; n < 10; n++) await pg.keyboard.press('ArrowLeft')
        const k = await snap(pg, i)
        c.check(
          `#${i} ArrowLeft x10 moves the pole to -2`,
          k.out.pole === '−2.00' && k.out.tau === '0.50 s' && k.out.tr === '1.10 s',
          k.out,
        )
        c.check(`#${i} handle follows keyboard`, near(toS(TIME, k.h0)[0], -2, 0.006), k.h0)
        await c.shoot(pg.locator('.zoomable').nth(i), `key-first-${i}.png`)
      } else timeConsistent(c, s, `#${i} SSR`)
    }
  }

  if (note === 'system-response') {
    let s = await snap(pg, 1)
    c.check(
      'undamped: case and no settling',
      s.out.case === 'undamped' && s.out.tsRule === '–' && s.out.os === '100.0%',
      s.out,
    )
    await dragS(pg, 1, [0, Math.PI], [-3, 2])
    s = await snap(pg, 1)
    c.check(
      'lock sigma keeps sigma_d = 0 under drag',
      s.out.sigmaD === '0.00 1/s' && s.out.wd === '2.00 rad/s',
      s.out,
    )
    timeConsistent(c, s, 'undamped after drag')

    // Regions figure: course presets, pointer drags across every region.
    s = await snap(pg, 2)
    c.check(
      'course underdamped example',
      s.out.tp === '1.11 s' && s.out.os === '32.9%' && s.out.zeta === '0.333',
      s.out,
    )
    await pg.locator(roots(2)).locator('input[value="overdamped"]').check({ force: true })
    s = await snap(pg, 2)
    c.check(
      'overdamped preset poles',
      s.out.poles === '−1.15, −7.85' && s.out.denom === 's² + 9.00s + 9.00',
      s.out,
    )
    c.check(
      'overdamped draws two real handles',
      s.h1 && near(toS(TIME, s.h0)[1], 0, 0.006) && near(toS(TIME, s.h1)[0], -7.854, 0.01),
      [s.h0, s.h1],
    )
    timeConsistent(c, s, 'overdamped')
    await pg.locator(roots(2)).locator('input[value="critical"]').check({ force: true })
    s = await snap(pg, 2)
    c.check(
      'critical preset',
      s.out.poles === '−3.00 (double)' && s.out.case === 'critically damped',
      s.out,
    )
    // Up arrow lifts the double pole into a pair.
    await pg.locator(roots(2)).locator('[data-prs-handle="0"]').focus()
    await pg.keyboard.press('ArrowUp')
    s = await snap(pg, 2)
    c.check(
      'ArrowUp splits a double pole into a pair',
      s.out.poles === '−3.00 ± j0.10' && s.checked.includes('underdamped'),
      s.out,
    )
    await dragS(pg, 2, [-3, 0.1], [1.5, 2.5])
    s = await snap(pg, 2)
    c.check(
      'drag into the right half plane is unstable',
      s.out.case === 'unstable' && s.out.tsBand === '–' && s.checked.includes('unstable'),
      s.out,
    )
    timeConsistent(c, s, 'unstable')
    await c.shoot(pg.locator('.zoomable').nth(2), 'drag-unstable.png')
    await dragS(pg, 2, [1.5, 2.5], [0.1, 4])
    s = await snap(pg, 2)
    c.check(
      'drag near the jw axis snaps to undamped',
      s.out.case === 'undamped' && s.checked.includes('undamped'),
      s.out,
    )
    await dragS(pg, 2, [0, 4], [-4, 0.1])
    s = await snap(pg, 2)
    c.check(
      'drag onto the real axis gives a double pole',
      s.out.case === 'critically damped' && s.h1 !== null,
      s.out,
    )
    await dragS(pg, 2, [-4, 0], [-1.5, 0])
    s = await snap(pg, 2)
    c.check(
      'dragging one real pole splits them: overdamped',
      s.out.case === 'overdamped' && /−1\.50/.test(s.out.poles) && /−4\.00/.test(s.out.poles),
      s.out,
    )
    timeConsistent(c, s, 'overdamped after drag')
    await c.shoot(pg.locator('.zoomable').nth(2), 'drag-overdamped.png')

    // Peak figure SSR equals the course example.
    s = await snap(pg, 3)
    c.check('peak figure starts at -1 ± j2.83', s.out.poles === '−1.00 ± j2.83', s.out.poles)

    // Relations figure: each lock keeps its quantity while dragging.
    const rel = pg.locator(roots(4))
    for (const [lock, key, to] of [
      ['zeta', 'zeta', [-2, 6]],
      ['sigma', 'sigmaD', [-4, 1.2]],
      ['omega', 'wd', [-5, 5]],
    ]) {
      await rel.locator(`input[value="${lock}"]`).check({ force: true })
      const before = (await snap(pg, 4)).out[key]
      const start = parsePair((await snap(pg, 4)).out.poles)
      await dragS(pg, 4, start, to)
      s = await snap(pg, 4)
      c.check(`lock ${lock} keeps ${key} (${before})`, s.out[key] === before, {
        before,
        after: s.out[key],
        poles: s.out.poles,
      })
      c.check(`lock ${lock} shows sage reference poles`, s.ref0 !== null, s.ref0)
      c.check(
        `lock ${lock}: pole moved`,
        parsePair(s.out.poles).join() !== start.join(),
        s.out.poles,
      )
      timeConsistent(c, s, `lock ${lock}`)
      await c.shoot(pg.locator('.zoomable').nth(4), `lock-${lock}.png`)
    }
    await rel.locator('input[value="none"]').check({ force: true })
    s = await snap(pg, 4)
    c.check('free lock hides reference', s.ref0 === null, s.ref0)
    // Keyboard on the relations handle: Left 5, Up 5.
    await rel.locator('[data-prs-handle="0"]').focus()
    const k0 = parsePair(s.out.poles)
    for (let n = 0; n < 5; n++) await pg.keyboard.press('ArrowLeft')
    for (let n = 0; n < 5; n++) await pg.keyboard.press('ArrowUp')
    s = await snap(pg, 4)
    const k1 = parsePair(s.out.poles)
    c.check(
      'arrow keys move the pole by 0.1 per press',
      near(k1[0], k0[0] - 0.5) && near(k1[1], k0[1] + 0.5),
      { k0, k1 },
    )
    const ring = await pg.evaluate(
      () => getComputedStyle(document.activeElement.querySelector('.prs-ring')).stroke,
    )
    c.check('focused handle shows its ring', ring !== 'none', ring)
    await c.shoot(pg.locator('.zoomable').nth(4), 'focus-keyboard.png')
    const other = await snap(pg, 3)
    c.check('instances are independent', other.out.poles === '−1.00 ± j2.83', other.out.poles)
  }

  if (note === 'time-response') {
    const r = pg.locator(roots(1))
    const want = {
      overdamped: '−1.15, −7.85',
      critical: '−3.00 (double)',
      underdamped: '−1.00 ± j2.83',
      undamped: '0.00 ± j3.00',
    }
    for (const [id, poles] of Object.entries(want)) {
      await r.locator(`input[value="${id}"]`).check({ force: true })
      const s = await snap(pg, 1)
      c.check(
        `course preset ${id}`,
        s.out.poles === poles && s.out.wn === '3.00 rad/s' && s.caseAttr === id,
        s.out,
      )
      timeConsistent(c, s, `preset ${id}`)
    }
    // Radio keyboard: ArrowLeft from undamped selects underdamped.
    await r.locator('input[value="undamped"]').focus()
    await pg.keyboard.press('ArrowLeft')
    const s = await snap(pg, 1)
    c.check(
      'radio arrow key selects the previous case',
      s.checked.includes('underdamped') && s.out.os === '32.9%',
      s.out,
    )
  }

  if (note === 'root-locus-control') {
    const expect = {
      none: ['164.5', '−180.0°', '0.108'],
      integrator: ['164.5', '−280.0°', '–'],
      pi: ['164.5', '−181.4°', '0.000'],
    }
    for (let i = 0; i < 3; i++) {
      const s = await snap(pg, i)
      const id = instances[i].comp
      c.check(
        `${id}: SSR at the uncompensated design gain`,
        s.out.k === expect[id][0] && s.out.angle === expect[id][1] && s.out.ess === expect[id][2],
        s.out,
      )
      c.check(
        `${id}: shows its own locus`,
        s.locus.length === 2 && s.locus.every(l => l === id),
        s.locus,
      )
      // Closed-loop dots equal the readout poles.
      const drawn = s.closed.map(p => toS(LOCUS, p)).filter(p => p[1] >= -1e-6)
      const listed = s.out.poles.split(', ').map(t => parsePair(t) ?? [minus(t), 0])
      c.check(
        `${id}: closed-loop dots match readout`,
        listed.every(l => drawn.some(d => near(d[0], l[0], 0.06) && near(d[1], l[1], 0.06))),
        { drawn, listed },
      )
    }
    c.check(
      'integrator alone is unstable at K = 164.5',
      (await snap(pg, 1)).out.stab === 'unstable',
    )
    const pi = pg.locator(roots(2))
    await pi.locator('[data-prs-design]').click()
    let s = await snap(pg, 2)
    c.check(
      'PI design gain K = 158.1 at zeta 0.174',
      s.out.k === '158.1' &&
        s.out.zeta === '0.174' &&
        s.out.poles === '−0.09, −0.68 ± j3.84, −11.55',
      s.out,
    )
    const pts = s.pts.map(p => toT(LOCUS, p))
    // The closed-loop pole at -0.09 leaves a slow tail: still rising toward 1 at t = 12 s.
    const [, c10] = pts.find(p => p[0] >= 10)
    const [, c12] = pts[pts.length - 1]
    c.check('PI step response creeps up toward 1', c12 > c10 && c12 > 0.95 && c12 < 1, { c10, c12 })
    await pi.locator('input[value="none"]').check({ force: true })
    s = await snap(pg, 2)
    c.check(
      'switching to gain only swaps the locus',
      s.locus.every(l => l === 'none') && s.out.ess !== '0.000',
      s,
    )
    const end = s.pts.map(p => toT(LOCUS, p)).pop()
    c.check('gain-only final value is 1 - e(inf)', near(end[1], 1 - minus(s.out.ess), 0.02), {
      end,
      ess: s.out.ess,
    })
    await pi.locator('[data-prs-design]').click()
    s = await snap(pg, 2)
    c.check('gain-only design gain 164.5', s.out.k === '164.5' && s.out.zeta === '0.174', s.out)
    // Slider keyboard.
    await pi.locator('[data-prs-k-input]').focus()
    for (let n = 0; n < 5; n++) await pg.keyboard.press('ArrowRight')
    s = await snap(pg, 2)
    c.check('slider ArrowRight steps K by 0.1', s.out.k === '165.0', s.out.k)
    // Drag along the locus: pointer near the jw crossing of the gain-only locus (K = 396, w = 5.66).
    await dragS(pg, 2, [-0.7, 3.9], [0, 5.66], LOCUS)
    s = await snap(pg, 2)
    c.check(
      'dragging to the jw crossing sets K near 396',
      Math.abs(minus(s.out.k) - 396) < 12,
      s.out.k,
    )
    await c.shoot(pg.locator('.zoomable').nth(2), 'drag-jw-crossing.png')
    const integ = pg.locator(roots(1))
    await integ.locator('[data-prs-design]').click()
    s = await snap(pg, 1)
    c.check(
      'integrator design point is far from A',
      s.out.zeta === '0.174' && minus(s.out.k) < 30,
      s.out,
    )
    await c.shoot(pg.locator('.zoomable').nth(1), 'integrator-design.png')
  }

  if (note === 'z-transform') {
    let s = await snap(pg, 0)
    timeConsistent(c, s, 'z SSR')
    await dragS(pg, 0, [-2, 2], [1, 2])
    s = await snap(pg, 0)
    c.check('z: right half plane is unstable', s.out.case === 'unstable', s.out.case)
  }

  // Cleanup removes every listener: a click after cleanup changes nothing.
  const before = await snap(pg, 0)
  await pg.evaluate(() => window.__cleanups.forEach(fn => fn()))
  const box = await pg.locator(roots(0)).locator('[data-prs-plane]').boundingBox()
  await pg.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.2)
  const after = await snap(pg, 0)
  c.check(
    'cleanup removes pointer listeners',
    JSON.stringify(after.out) === JSON.stringify(before.out),
    after.out,
  )

  await standardChecks(pg, errors, c, '.zoomable')
  await browser.close()
  console.log(note)
  ok = (await c.finish()) && ok
}
process.exit(ok ? 0 : 1)
