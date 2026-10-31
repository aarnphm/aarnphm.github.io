// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Dotring: a disc in a ring of dots.

export default {
  family: 'flower',
  name: 'dotring',
  draw({ o, x, y, r, rot, out, circlePoly, TAU }) {
    out.polys.push(circlePoly(x, y, r * 0.32))
    const n = Math.max(6, o.petals || 8),
      d = r * 0.72,
      rr = Math.min(r * 0.24, ((Math.PI * d) / n) * 0.42)
    for (let i = 0; i < n; i++) {
      const a = rot + (i / n) * TAU
      out.polys.push(circlePoly(x + Math.cos(a) * d, y + Math.sin(a) * d, rr))
    }
  },
}
