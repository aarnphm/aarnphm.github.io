// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Quatre: four broad petals.

export default {
  family: 'flower',
  name: 'quatre',
  draw({ x, y, r, rot, out, blade, circlePoly, TAU, dir }) {
    for (let i = 0; i < 4; i++) {
      const a = rot + (i / 4) * TAU
      const b = blade('petal', {
        x: x + Math.cos(a) * r * 0.12,
        y: y + Math.sin(a) * r * 0.12,
        dir: a,
        len: r * 0.88,
        width: 1.3,
        bend: 0,
        curl: 0,
        N: 14,
      })
      out.polys.push(b.polys[0])
    }
    out.polys.push(circlePoly(x, y, r * 0.18))
  },
}
