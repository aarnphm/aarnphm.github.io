// A flower, for the ornament: centred at (x, y), radius r, turned by rot; o has petals and detail.
// It adds polys, lines and dots to out (out.radius: how far it reaches). ring(n, r0, r1, width,
// shape, off) sets n blades round the middle, from r0 out to r1.
// Star: a star of o.petals points.

export default {
  family: 'flower',
  name: 'star',
  draw({ o, x, y, r, rot, out, TAU }) {
    const n = o.petals || 6,
      p = []
    for (let i = 0; i < n * 2; i++) {
      const a = rot + (i / (n * 2)) * TAU,
        q = i % 2 ? r * 0.42 : r
      p.push([x + Math.cos(a) * q, y + Math.sin(a) * q])
    }
    out.polys.push(p)
    out.dots.push([x, y, r * 0.14])
  },
}
