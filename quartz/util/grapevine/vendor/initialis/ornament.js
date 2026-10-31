// The growing ornament. Stems step forward through the clearance field, curl more as they age
// (scrolls), branch to one side, carry leaves, end in curls, leaves or flowers. Then the biggest
// remaining gaps are filled one at a time (new stems sprouting off whatever is nearest, flowers,
// leaves, berries) until nothing is left above the density threshold.

import { HEADS } from './kinds.js'
import { leaf, flower, circlePoly } from './shapes.js'
import { TAU, clamp, lerp } from './util.js'

// flowers on the end of a stem (head: true in their modules)
const HEAD = HEADS

// sizes the growth works with; the field's clearance cap has to cover all of them
export function ornSizes(o) {
  const m = o.margin
  const leafMax = o.leafSize * (55 + o.stemW * 5.5)
  const flowerMax = o.flowerSize * (28 + o.stemW * 2.6)
  const fillMin = lerp(70, m * 0.55 + 1.5, o.density)
  const sproutMin = Math.min(140, Math.max(fillMin, o.stemW * 2 + m * 2, leafMax * 0.3))
  const cap = Math.min(
    180,
    Math.max(40, flowerMax + m + 4, leafMax * o.leafW * 0.5 + m + 4, fillMin + 4, sproutMin + 4),
  )
  return { leafMax, flowerMax, fillMin, sproutMin, cap }
}

export function grow(field, o, R, noise, seeds) {
  const m = o.margin
  const items = { stems: [], leaves: [], flowers: [], dots: [], beads: [] }
  const rMin = Math.max(o.stemW * 0.8 + m * 0.6, 3)
  const rMax = lerp(900, rMin * 1.5, Math.pow(o.curl, 0.5))
  const branchEvery = lerp(420, 36, o.branch)
  const leafEvery = lerp(240, 22, o.leafEvery)
  const baseLen = 650 * o.length
  const { leafMax, flowerMax, fillMin, sproutMin } = ornSizes(o)
  const thin = 1 - 0.45 * (o.thin ?? 0.5)
  const started = performance.now()
  const budget = o.budget || 1500
  const late = () => performance.now() - started > budget
  const queue = []
  let stemId = 0
  // growth time, in units of stem length: a branch starts when its parent reaches it, leaves and
  // flowers when their stem does; gap-fillers trail in after the main growth
  let endMax = 0,
    fillN = 0
  const fillT = () => endMax * 0.45 + fillN++ * 2.5

  // the largest circle on the curling side that the stem could follow without hitting anything
  const fitRadius = (x, y, th, sgn, need) => {
    const nx = -Math.sin(th) * sgn,
      ny = Math.cos(th) * sgn
    for (let r = rMax; r > rMin; r /= 1.35) {
      const cx = x + nx * r,
        cy = y + ny * r
      let ok = true
      for (const a of [0.9, 1.7, 2.6, 3.5]) {
        // point a radians further round the circle, turning the way the stem turns
        const ang = Math.atan2(y - cy, x - cx) + a * sgn
        if (field.at(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r) < need) {
          ok = false
          break
        }
      }
      if (ok) return r
    }
    return rMin
  }

  const tryStem = s => {
    const w0 = Math.max(0.6, s.w)
    const step = clamp(Math.min(rMin * 0.35, w0 * 0.6 + 1.5), 1.2, 5)
    const lag = Math.ceil((w0 + m) / step) + 2
    let x = s.x,
      y = s.y,
      th = s.dir
    let sign = s.sign
    if (!sign) {
      const look = Math.min(90, rMin * 4 + 30)
      const l = field.at(x + Math.cos(th - 1) * look, y + Math.sin(th - 1) * look)
      const r = field.at(x + Math.cos(th + 1) * look, y + Math.sin(th + 1) * look)
      sign = (r > l ? 1 : -1) * (R() < 0.15 ? -1 : 1)
    }
    const id = stemId++
    const wave = s.depth === 0 && o.undulate > 0
    const lambda = 160 + rMin * 6
    const phase = R() * TAU
    const ramp = Math.max(20, s.len * lerp(0.7, 0.18, o.curl) * (s.depth ? 0.6 : 1))
    const pts = [[x, y]],
      ws = [w0],
      dirs = [th],
      ks = [0]
    const pend = []
    let len = 0,
      turned = 0,
      end = 'len',
      stepN = 0,
      fitR = rMax
    field.begin()
    while (len < s.len) {
      const t = len / s.len
      const w = Math.max(0.5, w0 * (1 - o.taper * t))
      const age = Math.min(1, len / ramp)
      let k
      if (wave) k = o.undulate * Math.sin((len / lambda) * TAU + phase) * (TAU / lambda) * 1.4
      else {
        // the fitting circle changes slowly; refit every few steps
        if (stepN++ % 3 === 0) fitR = fitRadius(x, y, th, sign, w / 2 + m)
        k = (sign / fitR) * o.curl * (0.12 + 0.88 * age * age)
      }
      k += o.wander * noise(len * 0.006 + id * 7.3, id * 3.1) * 0.02
      const need = w / 2 + m
      const grace = len < s.grace
      const look = need * 0.9 + step
      // near the start, the parent it springs from doesn't count as an obstacle
      const clear = (px, py) => {
        const c = field.at(px, py)
        if (!grace || c >= need) return c
        if (!s.parent) return field.atBase(px, py)
        return stemDist(px, py, s.parent) - c < field.cell * 1.6 + 1
          ? Math.min(field.atBase(px, py), need)
          : c
      }
      let chosen = null
      for (let c = 0; c < 15; c++) {
        const dev = c === 0 ? 0 : Math.ceil(c / 2) * 0.075 * (c % 2 ? sign : -sign)
        const a = th + k * step + dev
        const ca = Math.cos(a),
          sa = Math.sin(a)
        const nx = x + ca * step,
          ny = y + sa * step
        if (clear(nx, ny) < need) continue
        if (clear(x + ca * look, y + sa * look) < need * 0.5) continue
        chosen = [a, nx, ny]
        break
      }
      if (!chosen) {
        end = 'blocked'
        break
      }
      turned += chosen[0] - th
      th = chosen[0]
      x = chosen[1]
      y = chosen[2]
      pts.push([x, y])
      ws.push(w)
      dirs.push(th)
      ks.push(k)
      pend.push([x, y, w / 2])
      if (pend.length > lag) {
        const q = pend.shift()
        field.stamp(q[0], q[1], q[2])
      }
      len += step
      if (!wave && Math.abs(turned) > o.turns * TAU + 0.6) {
        end = 'curl'
        break
      }
      if (len > (o.maxLen || 4000)) break
    }
    const minLen = Math.max(o.stemW * 2.5, m * 3, 14)
    if (len < minLen) {
      field.rollback()
      return null
    }
    for (const q of pend) field.stamp(q[0], q[1], q[2])
    field.stamp(pts[0][0], pts[0][1], ws[0] / 2)
    const log = field.detach()
    const acc = [0]
    for (let i = 1; i < pts.length; i++)
      acc.push(acc[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    const t0 = s.t0 || 0
    endMax = Math.max(endMax, t0 + len)
    const stem = {
      pts,
      ws,
      dirs,
      ks,
      sign,
      end,
      depth: s.depth,
      len: acc[acc.length - 1],
      wave,
      id,
      log,
      acc,
      t0,
    }
    items.stems.push(stem)
    return stem
  }

  // distance from p to the edge of a stem
  // (stem points within skipR of (sx, sy) are ignored: that's where a leaf is allowed to touch)
  const stemDist = (px, py, st, sx = 0, sy = 0, skipR = 0) => {
    let best = Infinity
    const P = st.pts,
      s2 = skipR * skipR
    for (let i = 0; i < P.length; i++) {
      if (s2 && (P[i][0] - sx) ** 2 + (P[i][1] - sy) ** 2 < s2) continue
      const d = Math.hypot(P[i][0] - px, P[i][1] - py) - st.ws[i] / 2
      if (d < best) best = d
    }
    return best
  }

  // fit a leaf: shrink until every sample past the base clears its own half width.
  // `stem` is the stem it hangs from, lifted out of the field while its leaves are placed
  const tryLeaf = (type, x, y, dir, Lmax, side, stem, t0 = 0) => {
    for (let L = Lmax; L >= Math.max(6, Lmax * 0.3); L *= 0.84) {
      const shape = leaf(type, {
        x,
        y,
        dir,
        len: L,
        width: o.leafW,
        lobes: o.lobes,
        depth: o.lobeDepth,
        serr: o.serr,
        bend: -side * o.leafCurl * 0.9 * (0.7 + R() * 0.6),
        curl: -side * o.leafCurl * 1.3,
        detail: o.detail,
        settings: o.settings,
      })
      let ok = true
      for (const [px, py, r, t] of shape.cover) {
        if (t < 0.12) continue
        if (field.at(px, py) < r + m) {
          ok = false
          break
        }
        if (stem && t > 0.25 && stemDist(px, py, stem, x, y, L * 0.55) < r + m) {
          ok = false
          break
        }
      }
      if (!ok) continue
      for (const [px, py, r] of shape.cover) if (r > 0.5) field.stamp(px, py, r)
      shape.t0 = t0
      shape.bx = x
      shape.by = y
      shape.dur = L * 0.9
      items.leaves.push(shape)
      return shape
    }
    return null
  }

  const tryFlower = (type, x, y, dir, rmax, t0 = 0) => {
    const r = Math.min(rmax, flowerMax)
    if (r < 5) return null
    const shape = flower(type, {
      x,
      y,
      r,
      dir,
      rot: R() * TAU,
      petals: o.petals,
      detail: o.detail,
      settings: o.settings,
    })
    if (HEAD.has(type)) {
      // check along the head's length
      const L = shape.radius
      for (let u = r * 0.4; u <= L; u += Math.max(3, r * 0.3)) {
        if (field.at(x + Math.cos(dir) * u, y + Math.sin(dir) * u) < r * 0.55 + m) return null
      }
      for (let u = 0; u <= L; u += Math.max(3, r * 0.3))
        field.stamp(x + Math.cos(dir) * u, y + Math.sin(dir) * u, r * 0.6)
    } else {
      field.stamp(x, y, r)
    }
    shape.t0 = t0
    shape.cx = x
    shape.cy = y
    shape.dur = r * 1.6
    items.flowers.push(shape)
    return shape
  }

  const dot = (x, y, r, t0 = 0) => {
    items.dots.push([x, y, r, t0])
    field.stamp(x, y, r)
  }

  const decorate = st => {
    const { pts, ws, dirs, ks } = st
    const n = pts.length
    const [ex, ey] = pts[n - 1],
      ed = dirs[n - 1],
      ew = ws[n - 1]
    const term = o.terminal
    // leaves (and the end) are placed with the stem lifted out of the field, so its own
    // clearance doesn't refuse things that grow from it; distance to it is checked directly
    field.lift(st.log)
    st.log = null
    if (o.leaf !== 'none' && o.leafEvery > 0) {
      let next = leafEvery * (0.4 + R() * 0.6)
      let along = 0,
        side = R() < 0.5 ? 1 : -1
      for (let i = 1; i < n - 2 && !late(); i++) {
        along += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if (along < next) continue
        next = along + leafEvery * (0.75 + R() * 0.5)
        const sd = o.leafSide === 'outer' ? (ks[i] >= 0 ? -1 : 1) : (side = -side)
        const a = dirs[i] + sd * o.leafAngle
        const nx = -Math.sin(dirs[i]) * sd,
          ny = Math.cos(dirs[i]) * sd
        const bx = pts[i][0] + nx * ws[i] * 0.3,
          by = pts[i][1] + ny * ws[i] * 0.3
        tryLeaf(o.leaf, bx, by, a, leafMax * (0.75 + R() * 0.4), sd, st, st.t0 + along)
      }
    }
    let endDot = false
    if (st.end === 'curl' || term === 'ball') {
      endDot = term === 'ball' || term === 'curl'
    } else {
      // flowers end some stems whatever the ends are set to; 'flower' ends get them more often
      let done = false
      if (
        o.flower !== 'none' &&
        term !== 'point' &&
        R() < (term === 'flower' ? o.flowerChance + 0.25 : o.flowerChance * 0.6)
      ) {
        const room = field.at(
          ex + Math.cos(ed) * flowerMax * 0.5,
          ey + Math.sin(ed) * flowerMax * 0.5,
        )
        if (HEAD.has(o.flower))
          done = !!tryFlower(o.flower, ex, ey, ed, Math.max(6, room * 0.8), st.t0 + st.len)
        else if (room > 10) {
          const r = Math.min(room - m, flowerMax)
          const cx = ex + Math.cos(ed) * (r + 1),
            cy = ey + Math.sin(ed) * (r + 1)
          if (field.at(cx, cy) > r * 0.8 + m)
            done = !!tryFlower(o.flower, cx, cy, ed, r, st.t0 + st.len)
        }
      }
      if (!done && (term === 'leaf' || term === 'flower') && o.leaf !== 'none')
        tryLeaf(o.leaf, ex, ey, ed, leafMax, st.sign, st, st.t0 + st.len)
    }
    for (let i = 0; i < n; i++) field.stamp(pts[i][0], pts[i][1], ws[i] / 2)
    if (endDot) dot(ex, ey, Math.max(1.6, ew * (term === 'ball' ? 1.25 : 0.7)), st.t0 + st.len)
    // penwork beads along the outer side
    if (o.beads > 0) {
      const every = lerp(46, 7, o.beads)
      let along = 0,
        next = every
      for (let i = 1; i < n - 1; i++) {
        along += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if (along < next) continue
        next = along + every
        const sd = ks[i] >= 0 ? -1 : 1
        const r = Math.max(1.2, o.stemW * 0.7 + 1)
        const off = ws[i] / 2 + m * 0.6 + r
        const bx = pts[i][0] - Math.sin(dirs[i]) * sd * off,
          by = pts[i][1] + Math.cos(dirs[i]) * sd * off
        if (field.at(bx, by) >= r + m * 0.5) {
          items.beads.push([bx, by, r, st.t0 + along])
          field.stamp(bx, by, r)
        }
      }
    }
    // queue branches
    if (st.depth < o.maxDepth && o.branch > 0) {
      let along = 0,
        next = branchEvery * (0.35 + R() * 0.5)
      for (let i = 1; i < n - 3; i++) {
        along += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if (along < next || along > st.len - 20) continue
        next = along + branchEvery * (0.7 + R() * 0.6)
        let sd =
          Math.abs(ks[i]) < 1e-4
            ? R() < 0.5
              ? 1
              : -1
            : st.wave
              ? Math.sign(ks[i])
              : -Math.sign(ks[i])
        const a = dirs[i] + sd * o.branchAngle
        const off = ws[i] * 0.35
        queue.push({
          x: pts[i][0] - Math.sin(dirs[i]) * sd * off,
          y: pts[i][1] + Math.cos(dirs[i]) * sd * off,
          dir: a,
          w: Math.max(0.6, ws[i] * thin),
          len: st.len * lerp(0.4, 0.8, R()) * (st.depth === 0 ? 0.9 : 1),
          sign: sd,
          depth: st.depth + 1,
          grace: ws[i] + m + 3,
          parent: st,
          t0: st.t0 + along,
        })
      }
    }
  }

  const drain = () => {
    while (queue.length && !late()) {
      const s = queue.shift()
      const st = tryStem(s)
      if (st) decorate(st)
    }
    queue.length = 0
  }

  // start a stem into the gap at (bx, by): branch off the nearest existing stem when one is close,
  // otherwise grow out of whatever is nearest (frame, letter)
  const sproutInto = (bx, by, bd, w, len, depth, t0 = 0) => {
    let best = null,
      bestD = (bd * 2.4 + 40) ** 2
    for (const st of items.stems) {
      if (st.ws[0] < w * 0.5) continue
      const P = st.pts
      for (let i = 2; i < P.length - 2; i += 2) {
        const dx = P[i][0] - bx,
          dy = P[i][1] - by,
          d = dx * dx + dy * dy
        if (d < bestD) {
          bestD = d
          best = [st, i]
        }
      }
    }
    if (best) {
      const [st, i] = best
      const [px, py] = st.pts[i],
        a = st.dirs[i]
      const toGap = Math.atan2(by - py, bx - px)
      // leave along whichever way the parent runs that points more at the gap, turned towards it
      let fwd = Math.cos(toGap - a) >= 0 ? a : a + Math.PI
      let diff = Math.atan2(Math.sin(toGap - fwd), Math.cos(toGap - fwd))
      const sd = diff >= 0 ? 1 : -1
      const dir = fwd + sd * Math.min(Math.abs(diff), o.branchAngle * 1.2)
      const nx = -Math.sin(a),
        ny = Math.cos(a)
      const side = Math.cos(dir) * nx + Math.sin(dir) * ny >= 0 ? 1 : -1
      const x = px + nx * side * st.ws[i] * 0.35,
        y = py + ny * side * st.ws[i] * 0.35
      const ns = tryStem({
        x,
        y,
        dir,
        w: Math.min(w, st.ws[i] * thin),
        len,
        sign: sd,
        depth: Math.max(depth, st.depth + 1),
        grace: st.ws[i] + m + 3,
        parent: st,
        t0: st.t0 + st.acc[i],
      })
      if (ns) {
        decorate(ns)
        drain()
        return ns
      }
    }
    let x = bx,
      y = by
    for (let k = 0; k < 80; k++) {
      const c = field.at(x, y)
      if (c <= w / 2 + m + 1.5) break
      const g = field.grad(x, y)
      const st = Math.min(c - (w / 2 + m), 6)
      x -= g[0] * st
      y -= g[1] * st
    }
    const dir = Math.atan2(by - y, bx - x) + (R() - 0.5) * 0.9
    const st = tryStem({ x, y, dir, w, len, sign: 0, depth, grace: 0, t0 })
    if (st) {
      decorate(st)
      drain()
    }
    return st
  }

  seeds.forEach((sd, i) => {
    queue.push({
      x: sd.x,
      y: sd.y,
      dir: sd.dir,
      w: o.stemW * (sd.w || 1),
      len: baseLen * (sd.len || 1) * (0.8 + R() * 0.4),
      sign: sd.sign || 0,
      depth: 0,
      grace: sd.grace ?? m + o.stemW,
      t0: i * 25,
    })
  })
  drain()
  // no explicit starts: the first stems sprout into the biggest openings
  if (!seeds.length && o.mains > 0 && o.sprout > 0) {
    for (let i = 0, tries = 0; i < o.mains && tries < o.mains * 4 && !late(); tries++) {
      const b = field.biggest()
      if (b.d < sproutMin) break
      if (sproutInto(b.x, b.y, b.d, o.stemW, baseLen * (0.8 + R() * 0.4), 0, i * 30)) i++
      else field.stamp(b.x, b.y, b.d * 0.3)
    }
  }

  // fill the biggest gaps until none are left above the threshold
  if (o.density > 0) {
    let guard = 0
    while (guard++ < 3000 && !late()) {
      const b = field.biggest()
      if (b.d < fillMin) break
      let placed = false
      if (o.sprout > 0 && b.d > sproutMin && R() < o.sprout) {
        placed = !!sproutInto(
          b.x,
          b.y,
          b.d,
          o.stemW * lerp(0.55, 0.85, R()),
          clamp(b.d * lerp(4, 8, R()), 50, baseLen),
          1,
          fillT(),
        )
      }
      if (
        !placed &&
        o.flower !== 'none' &&
        !HEAD.has(o.flower) &&
        b.d > 12 + m &&
        R() < o.flowerChance
      ) {
        placed = !!tryFlower(o.flower, b.x, b.y, R() * TAU, b.d - m, fillT())
      }
      if (!placed && o.leaf !== 'none' && o.fillLeaves && b.d > 8 + m) {
        let x = b.x,
          y = b.y
        for (let k = 0; k < 60; k++) {
          const c = field.at(x, y)
          if (c <= m + 1.5) break
          const g = field.grad(x, y)
          const st = Math.min(c - m, 6)
          x -= g[0] * st
          y -= g[1] * st
        }
        const dir = Math.atan2(b.y - y, b.x - x)
        placed = !!tryLeaf(
          o.leaf,
          x,
          y,
          dir,
          Math.min(leafMax * 1.2, b.d * 2.6),
          R() < 0.5 ? 1 : -1,
          null,
          fillT(),
        )
      }
      if (!placed && o.berries > 0 && R() < o.berries && b.d > m + 3) {
        // a little cluster
        const r = clamp((b.d - m) * 0.42, 1.5, 7 * o.berrySize + 2)
        const n = b.d > r * 3 + m ? 3 + ((R() * 4) | 0) : 1
        const tc = fillT()
        for (let i = 0; i < n; i++) {
          const a = i * 2.4 + R(),
            q = i ? r * 1.15 * Math.sqrt(i) : 0
          const x = b.x + Math.cos(a) * q,
            y = b.y + Math.sin(a) * q
          if (field.at(x, y) >= r + m * 0.6) {
            items.dots.push([x, y, r, tc + i * 3])
            field.stamp(x, y, r)
            placed = true
          }
        }
      }
      if (!placed) field.stamp(b.x, b.y, Math.max(1.5, b.d * 0.3))
    }
  }

  // tiny gaps become dots
  if (o.dots > 0) {
    const dmin = lerp(30, m * 0.5 + 1.2, o.dots)
    let guard = 0
    while (guard++ < 1500 && !late()) {
      const b = field.biggest()
      if (b.d < dmin) break
      const r = clamp(b.d - m, 1.2, 4 + o.berrySize * 6)
      if (r > 1) dot(b.x, b.y, r, fillT())
      else field.stamp(b.x, b.y, b.d)
    }
  }
  return items
}

// stem centreline + widths -> outline polygon, with an optional broad-nib thick/thin
export function stemPoly(st, contrast = 0, penAngle = 0.6, endKind = 'round', minW = 0.35) {
  const { pts, ws, dirs } = st
  const n = pts.length
  const L = [],
    Rt = []
  const wid = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let w = ws[i]
    if (contrast > 0) w *= lerp(1, 0.12 + 0.88 * Math.abs(Math.sin(dirs[i] - penAngle)), contrast)
    wid[i] = Math.max(minW, w)
  }
  // gentle smoothing so the thick/thin doesn't step
  if (contrast > 0)
    for (let pass = 0; pass < 2; pass++)
      for (let i = 1; i < n - 1; i++) wid[i] = (wid[i - 1] + 2 * wid[i] + wid[i + 1]) / 4
  for (let i = 0; i < n; i++) {
    const a = dirs[i],
      nx = -Math.sin(a),
      ny = Math.cos(a),
      h = wid[i] / 2
    L.push([pts[i][0] + nx * h, pts[i][1] + ny * h])
    Rt.push([pts[i][0] - nx * h, pts[i][1] - ny * h])
  }
  const poly = [...L]
  const cap = (cx, cy, start, h) => {
    const steps = Math.max(3, Math.min(10, Math.round(h)))
    for (let k = 1; k < steps; k++) {
      const t = start - (k / steps) * Math.PI
      poly.push([cx + Math.cos(t) * h, cy + Math.sin(t) * h])
    }
  }
  const e = n - 1
  if (endKind === 'round') cap(pts[e][0], pts[e][1], dirs[e] + Math.PI / 2, wid[e] / 2)
  for (let i = n - 1; i >= 0; i--) poly.push(Rt[i])
  cap(pts[0][0], pts[0][1], dirs[0] - Math.PI / 2, wid[0] / 2)
  return poly
}

export function stemDetail(st, kind) {
  const { pts, ws, dirs, ks } = st
  const n = pts.length
  if (kind === 'inline' || kind === 'vein') {
    const a = Math.round(n * 0.06),
      b = Math.round(n * 0.94)
    if (ws[0] < 3) return []
    return [pts.slice(a, b)]
  }
  if (kind === 'hatch') {
    const out = []
    for (let i = 2; i < n - 2; i += 3) {
      const sd = ks[i] >= 0 ? 1 : -1
      const nx = -Math.sin(dirs[i]) * sd,
        ny = Math.cos(dirs[i]) * sd
      const h = ws[i] / 2
      if (h < 1.2) continue
      out.push([
        [pts[i][0] + nx * h * 0.1, pts[i][1] + ny * h * 0.1],
        [pts[i][0] + nx * h * 0.9, pts[i][1] + ny * h * 0.9],
      ])
    }
    return out
  }
  return []
}

export { circlePoly }
