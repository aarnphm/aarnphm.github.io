// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Cinquefoil: five teardrops on a short stalk.

export default {
  family: 'leaf',
  name: 'cinquefoil',
  draw({ type, o, blade, bladeDetail, bladeCover }) {
    const out = { polys: [], lines: [], dots: [], cover: [] }
    const n = type === 'trefoil' ? 3 : 5
    const spread = type === 'trefoil' ? 0.85 : 1.25
    const stalk = o.len * 0.18
    const cx = o.x + Math.cos(o.dir) * stalk,
      cy = o.y + Math.sin(o.dir) * stalk
    for (let i = 0; i < n; i++) {
      const a = o.dir + (i / (n - 1) - 0.5) * 2 * spread
      const L = o.len * (i === (n - 1) / 2 ? 0.82 : 0.66)
      const b = blade('teardrop', {
        ...o,
        x: cx,
        y: cy,
        dir: a + o.bend * 0.3,
        len: L,
        width: Math.max(0.4, o.width * 0.95),
        bend: o.bend * 0.3,
        curl: 0,
        N: 14,
      })
      out.polys.push(...b.polys)
      out.lines.push(
        ...bladeDetail(b, o.detail === 'veins' || o.detail === 'vein' ? 'vein' : o.detail),
      )
      for (const d of bladeCover(b)) out.cover.push([d[0], d[1], d[2], 0.18 + (d[3] * L) / o.len])
    }
    const st = blade('willow', { ...o, len: stalk * 1.4, width: 0.18, curl: 0, bend: 0, N: 6 })
    out.polys.push(...st.polys)
    return out
  },
}
