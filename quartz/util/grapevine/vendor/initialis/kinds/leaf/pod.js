// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Pod: a run of dots shrinking towards the tip.

export default {
  family: 'leaf',
  name: 'pod',
  draw({ o, blade, circlePoly }) {
    // a run of dots shrinking towards the tip
    const st = blade('willow', { ...o, width: 0.04, N: 30 })
    const out = { polys: [], lines: [], dots: [], cover: [] }
    const L = o.len,
      r0 = L * 0.12 * (0.6 + o.width * 0.8)
    let along = r0 * 0.6
    for (let k = 0; k < 9; k++) {
      const t = along / L
      if (t > 1) break
      const r = r0 * (1 - 0.65 * t)
      const i = Math.min(st.axis.length - 1, Math.round(t * (st.axis.length - 1)))
      const [px, py] = st.axis[i]
      out.polys.push(circlePoly(px, py, r))
      out.cover.push([px, py, r, t])
      along += r * 2 + Math.max(1.2, r * 0.45)
    }
    return out
  },
}
