// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Ivy: palmate, lobed, notched at the stalk.

export default {
  family: 'leaf',
  name: 'ivy',
  draw({ o, TAU }) {
    // palmate, lobed, notched at the stalk
    const lobes = o.lobes >= 4 ? 5 : 3
    const R = o.len * 0.52
    const cx = o.x + Math.cos(o.dir) * R * 0.95,
      cy = o.y + Math.sin(o.dir) * R * 0.95
    const poly = []
    const M = 90
    for (let i = 0; i < M; i++) {
      const a = o.dir + Math.PI + (i / M) * TAU // start at the notch
      const rel = a - o.dir
      const lob = Math.pow(Math.abs(Math.cos((rel * lobes) / 2)), 1.6)
      const notch = 1 - 0.55 * Math.exp(-((Math.cos(rel) + 1) * 18))
      const r = R * (0.58 + 0.42 * lob) * notch * (1 + o.width * 0.2 - 0.1)
      poly.push([cx + Math.cos(a + o.bend * 0.2) * r, cy + Math.sin(a + o.bend * 0.2) * r])
    }
    const lines = []
    if (o.detail === 'vein' || o.detail === 'veins') {
      for (let k = 0; k < lobes; k++) {
        const a = o.dir + (k / (lobes - 1) - 0.5) * (lobes === 3 ? 2.1 : 3.4)
        lines.push([
          [o.x + Math.cos(o.dir) * R * 0.5, o.y + Math.sin(o.dir) * R * 0.5],
          [cx + Math.cos(a) * R * 0.75, cy + Math.sin(a) * R * 0.75],
        ])
      }
    }
    // a ring of small discs on the outline plus the middle
    const cover = [[cx, cy, R * 0.55, 0.5]]
    for (let i = 0; i < M; i += 6) {
      const [px, py] = poly[i]
      cover.push([
        px + (cx - px) * 0.2,
        py + (cy - py) * 0.2,
        Math.hypot(px - cx, py - cy) * 0.2,
        0.4 + (0.6 * Math.hypot(px - o.x, py - o.y)) / (2 * R),
      ])
    }
    return { polys: [poly], lines, dots: [], cover }
  },
}
