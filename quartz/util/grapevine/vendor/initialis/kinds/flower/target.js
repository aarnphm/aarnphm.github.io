// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Target: a disc with rings.

export default {
  family: 'flower',
  name: 'target',
  draw({ x, y, r, out, circlePoly, circleLine }) {
    out.polys.push(circlePoly(x, y, r))
    for (let f = 0.78; f > 0.2; f -= 0.24) out.lines.push(circleLine(x, y, r * f))
    out.dots.push([x, y, r * 0.1])
  },
}
