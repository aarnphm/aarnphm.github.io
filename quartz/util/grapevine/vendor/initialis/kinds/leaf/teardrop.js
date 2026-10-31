// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Teardrop: round at the tip, tapering to the stalk.

export default {
  family: 'leaf',
  name: 'teardrop',
  draw({ o, blade, bladeDetail, bladeCover }) {
    const b = blade('teardrop', o)
    b.lines = bladeDetail(b, o.detail)
    b.cover = bladeCover(b, 2)
    return b
  },
}
