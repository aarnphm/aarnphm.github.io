// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Heart: a heart on its point, pointing along dir.

export default {
  family: 'leaf',
  name: 'heart',
  draw({ o, TAU }) {
    const R = o.len * 0.5
    const cx = o.x + Math.cos(o.dir) * R,
      cy = o.y + Math.sin(o.dir) * R
    const poly = []
    for (let i = 0; i < 60; i++) {
      const t = (i / 60) * TAU
      // classic heart curve, pointing along dir
      const hx = 16 * Math.sin(t) ** 3,
        hy = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)
      const u = (-hy / 17) * R,
        v = (hx / 17) * R * (0.7 + o.width * 0.4)
      const ca = Math.cos(o.dir),
        sa = Math.sin(o.dir)
      poly.push([cx + ca * u - sa * v, cy + sa * u + ca * v])
    }
    return {
      polys: [poly],
      lines: [],
      dots: [],
      cover: [
        [cx, cy, R * 0.8, 0.5],
        [cx - Math.cos(o.dir) * R * 0.6, cy - Math.sin(o.dir) * R * 0.6, R * 0.3, 0.2],
      ],
    }
  },
}
