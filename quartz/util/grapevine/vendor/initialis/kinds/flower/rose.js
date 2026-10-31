// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Rose: five petals, five more drawn over them, and a knot of seeds.

export default {
  family: 'flower',
  name: 'rose',
  draw({ x, y, r, rot, out, ring, blade, circlePoly, TAU, dir }) {
    const n = 5
    ring(n, r * 0.12, r, 1.15, 'petal')
    const inner = []
    const m = 5
    for (let i = 0; i < m; i++) {
      const a = rot + ((i + 0.5) / m) * TAU
      const b = blade('petal', {
        x: x + Math.cos(a) * r * 0.08,
        y: y + Math.sin(a) * r * 0.08,
        dir: a,
        len: r * 0.55,
        width: 1.1,
        bend: 0,
        curl: 0,
        N: 12,
      })
      inner.push(b.polys[0])
    }
    // inner petals are drawn as lines over the outer ring, then a knot of seeds
    for (const p of inner) out.lines.push([...p, p[0]])
    out.polys.push(circlePoly(x, y, r * 0.2))
    for (let i = 0; i < 6; i++) {
      const a = rot + (i / 6) * TAU
      out.dots.push([x + Math.cos(a) * r * 0.1, y + Math.sin(a) * r * 0.1, r * 0.04])
    }
  },
}
