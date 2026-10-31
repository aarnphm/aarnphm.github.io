// A flower on the end of a stem (head: true), pointing along dir from (x, y) where the stem ends,
// size r. P(u, v) is a point u along the stem's direction and v across it; det: whether it has
// detail.
// It adds polys, lines and dots to out, and sets out.radius to how far it reaches.
// Bell: a flared bell with a clapper.

export default {
  family: 'flower',
  name: 'bell',
  head: true,
  draw({ r, out, P, det }) {
    const p = []
    const L = r * 1.8
    for (let i = 0; i <= 20; i++) {
      const t = i / 20,
        w = r * (0.25 + 0.55 * Math.pow(t, 1.6))
      p.push(P(t * L, w))
    }
    for (let k = 4; k >= 0; k--) {
      const v = (k / 4 - 0.5) * 2 * r * 0.8
      p.push(P(L + r * 0.22 * (k % 2 ? 0 : 1), v))
    }
    for (let i = 20; i >= 0; i--) {
      const t = i / 20,
        w = r * (0.25 + 0.55 * Math.pow(t, 1.6))
      p.push(P(t * L, -w))
    }
    out.polys.push(p)
    if (det)
      out.lines.push(
        [P(L * 0.25, 0), P(L * 0.95, 0)],
        [P(L * 0.3, r * 0.15), P(L * 0.95, r * 0.5)],
        [P(L * 0.3, -r * 0.15), P(L * 0.95, -r * 0.5)],
      )
    out.dots.push([...P(L + r * 0.5, 0), r * 0.1])
    out.radius = L
  },
}
