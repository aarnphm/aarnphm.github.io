// A leaf, for the ornament: o has where it starts (x, y), its direction dir, len, width (0..1),
// bend, curl, and the leaf settings (lobes, depth, serr, detail). It returns { polys, lines, dots,
// cover }: cover is discs [x, y, r, t] over the whole leaf (t: how far from its base, 0..1), for
// fitting it among the rest. blade(shape, o) makes a single blade along a bending axis, where
// shape
// is a profile name or a function (t, side, o) giving its half width (0..1) at t.
// Fan: ginkgo, a wedge with a scalloped, notched edge.

export default {
  family: 'leaf',
  name: 'fan',
  draw({ o, blade, bladeCover }) {
    // ginkgo: a wedge with a scalloped, notched edge
    const st = blade('willow', { ...o, len: o.len * 0.3, width: 0.12, curl: 0, N: 6 })
    const [bx, by] = st.axis[st.axis.length - 1],
      a0 = st.dirs[st.dirs.length - 1]
    const R = o.len * 0.72,
      spread = 0.55 + o.width * 0.5
    const poly = [[bx, by]]
    for (let i = 0; i <= 40; i++) {
      const t = i / 40,
        a = a0 - spread + 2 * spread * t
      const notch = 1 - 0.25 * Math.exp(-((t - 0.5) ** 2) * 400)
      const r = R * (0.9 + 0.1 * Math.abs(Math.sin(t * Math.PI * 7))) * notch
      poly.push([bx + Math.cos(a) * r, by + Math.sin(a) * r])
    }
    const lines = []
    if (o.detail !== 'none')
      for (let k = 1; k < 6; k++) {
        const a = a0 - spread + (2 * spread * k) / 6
        lines.push([
          [bx, by],
          [bx + Math.cos(a) * R * 0.8, by + Math.sin(a) * R * 0.8],
        ])
      }
    const cover = [...bladeCover(st, 2)]
    for (let k = 0; k <= 4; k++) {
      const a = a0 - spread * 0.7 + (spread * 1.4 * k) / 4
      cover.push([bx + Math.cos(a) * R * 0.6, by + Math.sin(a) * R * 0.6, R * 0.32, 0.6])
    }
    cover.push([bx + Math.cos(a0) * R * 0.3, by + Math.sin(a0) * R * 0.3, R * 0.25, 0.4])
    return { polys: [...st.polys, poly], lines, dots: [], cover }
  },
}
