// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Dotted: a leaf outline traced in dots, with a dotted midrib.

export default {
  family: 'leaf',
  name: 'dotted',
  draw({ o, blade, bladeCover, circlePoly, resampleClosed }) {
    // a leaf outline traced in dots, with a dotted midrib
    const b = blade('ovate', o)
    const ds = Math.max(3, o.len * 0.08),
      r = Math.max(1.1, ds * 0.36)
    const polys = []
    for (const [x, y] of resampleClosed(b.polys[0], ds)) polys.push(circlePoly(x, y, r, 8))
    for (let i = 3; i < b.axis.length - 3; i += 3)
      polys.push(circlePoly(b.axis[i][0], b.axis[i][1], r * 0.8, 8))
    return { polys, lines: [], dots: [], cover: bladeCover(b, 2) }
  },
}
