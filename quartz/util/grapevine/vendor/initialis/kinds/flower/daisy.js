// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Daisy: many narrow petals and a seeded middle.

export default {
  family: 'flower',
  name: 'daisy',
  draw({ o, x, y, r, out, ring, circlePoly, TAU }) {
    const n = Math.max(10, Math.round((o.petals || 6) * 2.2))
    ring(n, r * 0.32, r, Math.min(0.5, 2.4 * Math.sin(Math.PI / n)), 'lance')
    out.polys.push(circlePoly(x, y, r * 0.25))
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * TAU,
        q = i ? r * 0.13 : 0
      out.dots.push([x + Math.cos(a) * q, y + Math.sin(a) * q, r * 0.035])
    }
  },
}
