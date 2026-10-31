// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Burst: rays, long and short by turns.

export default {
  family: 'flower',
  name: 'burst',
  draw({ o, x, y, r, rot, out, blade, circlePoly, TAU, dir }) {
    const n = Math.max(12, (o.petals || 6) * 3)
    for (let i = 0; i < n; i++) {
      const a = rot + (i / n) * TAU,
        L = r * (i % 2 ? 0.7 : 1)
      const b = blade('lance', {
        x: x + Math.cos(a) * r * 0.2,
        y: y + Math.sin(a) * r * 0.2,
        dir: a,
        len: L - r * 0.2,
        width: 0.2,
        bend: 0,
        curl: 0,
        N: 8,
      })
      out.polys.push(b.polys[0])
    }
    out.polys.push(circlePoly(x, y, r * 0.18))
  },
}
