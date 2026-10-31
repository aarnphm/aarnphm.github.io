// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Dot: a round dot on a short stalk.

export default {
  family: 'leaf',
  name: 'dot',
  draw({ o, blade, bladeCover, circlePoly, circleLine }) {
    // a round dot on a short stalk
    const r = o.len * 0.24 * (0.6 + o.width * 0.7)
    const st = blade('willow', {
      ...o,
      len: Math.max(1, o.len - r * 1.6),
      width: 0.1,
      curl: o.curl * 0.3,
      N: 8,
    })
    const [ex, ey] = st.axis[st.axis.length - 1],
      ea = st.dirs[st.dirs.length - 1]
    const cx = ex + Math.cos(ea) * r * 0.85,
      cy = ey + Math.sin(ea) * r * 0.85
    return {
      polys: [...st.polys, circlePoly(cx, cy, r)],
      lines: o.detail === 'none' ? [] : [circleLine(cx, cy, r * 0.5)],
      dots: [],
      cover: [...bladeCover(st, 3), [cx, cy, r, 0.8]],
    }
  },
}
