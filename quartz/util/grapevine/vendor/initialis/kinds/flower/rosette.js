// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Rosette: a ring of petals round a disc.

export default {
  family: 'flower',
  name: 'rosette',
  draw({ o, x, y, r, out, ring, circlePoly }) {
    const n = o.petals || 6
    ring(n, r * 0.36, r, Math.min(1.25, 2.6 * Math.sin(Math.PI / n) * 1.25), 'petal')
    out.polys.push(circlePoly(x, y, r * 0.27))
    if (r > 10) out.lines.push([...circlePoly(x, y, r * 0.15), circlePoly(x, y, r * 0.15)[0]])
  },
}
