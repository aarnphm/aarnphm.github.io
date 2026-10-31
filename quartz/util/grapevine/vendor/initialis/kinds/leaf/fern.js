// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Fern: a rachis with leaflets either side, longest in the middle.

export default {
  family: 'leaf',
  name: 'fern',
  draw({ o, blade, bladeCover }) {
    const rach = blade('willow', { ...o, width: 0.05 })
    const out = { polys: [...rach.polys], lines: [], dots: [], cover: bladeCover(rach) }
    const N = rach.axis.length - 1
    for (let i = 2; i < N; i += 2) {
      const t = i / N
      const L = o.len * 0.32 * Math.sin(Math.PI * (0.15 + t * 0.85))
      for (const sd of [1, -1]) {
        const a = rach.dirs[i] + sd * 0.95
        const b = blade('lance', {
          x: rach.axis[i][0],
          y: rach.axis[i][1],
          dir: a,
          len: L,
          width: 0.38,
          bend: sd * 0.25,
          curl: 0,
          N: 8,
        })
        out.polys.push(...b.polys)
        for (const d of bladeCover(b, 3)) out.cover.push([d[0], d[1], d[2], t])
      }
    }
    return out
  },
}
