// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Comma: paisley, a fat teardrop whose tip curls round.

export default {
  family: 'leaf',
  name: 'comma',
  draw({ o, blade, bladeDetail, bladeCover }) {
    // paisley: a fat teardrop whose tip curls round
    const sd = o.curl >= 0 ? 1 : -1
    const b = blade('boteh', {
      ...o,
      width: Math.min(1.25, o.width * 1.8),
      curl: sd * 3.6,
      bend: sd * 0.5,
      N: 36,
    })
    const lines = o.detail === 'none' ? [] : bladeDetail(b, 'inline')
    const m = b.axis[Math.round(b.axis.length * 0.3)]
    return {
      polys: b.polys,
      lines,
      dots: o.detail === 'none' ? [] : [[m[0], m[1], b.W * 0.18]],
      cover: bladeCover(b, 2),
    }
  },
}
