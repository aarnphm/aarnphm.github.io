// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Needles: a bundle of pine needles.

export default {
  family: 'leaf',
  name: 'needles',
  draw({ o, blade, bladeCover }) {
    // a bundle of pine needles
    const out = { polys: [], lines: [], dots: [], cover: [] }
    const n = 5 + Math.round(o.width * 4)
    for (let i = 0; i < n; i++) {
      const a = o.dir + (i / (n - 1) - 0.5) * 1.1
      const b = blade('willow', {
        ...o,
        dir: a,
        len: o.len * (0.75 + 0.25 * Math.cos((i / (n - 1) - 0.5) * 3)),
        width: 0.07,
        curl: o.curl * 0.4,
        N: 10,
      })
      out.polys.push(...b.polys)
      for (const d of bladeCover(b, 3)) out.cover.push([d[0], d[1], Math.max(d[2], 1.5), d[3]])
    }
    return out
  },
}
