// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Phyllo: a seed head, Vogel's spiral of dots growing outward.

export default {
  family: 'flower',
  name: 'phyllo',
  draw({ x, y, r, rot, out, circlePoly, clamp }) {
    // a seed head: Vogel's spiral of dots, growing outward
    const n = clamp(Math.round((r * r) / 28), 12, 220)
    const step = r / Math.sqrt(n)
    for (let i = 1; i <= n; i++) {
      const q = Math.sqrt(i / n) * r * 0.92,
        a = rot + i * 2.39996323
      out.polys.push(
        circlePoly(
          x + Math.cos(a) * q,
          y + Math.sin(a) * q,
          step * (0.3 + 0.18 * Math.sqrt(i / n)),
          8,
        ),
      )
    }
  },
}
