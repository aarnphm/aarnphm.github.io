// A flower on the end of a stem (head: true), pointing along dir from (x, y) where the stem ends,
// size r. P(u, v) is a point u along the stem's direction and v across it; det: whether it has
// detail.
// It adds polys, lines and dots to out, and sets out.radius to how far it reaches.
// Pomegranate: a round fruit with a crown, its seeds showing through a split.

export default {
  family: 'flower',
  name: 'pomegranate',
  head: true,
  draw({ r, out, TAU, P }) {
    const body = []
    const L = r * 1.7,
      cu = r * 0.75
    for (let i = 0; i < 40; i++) {
      const a = (i / 40) * TAU
      body.push(P(cu + Math.cos(a) * cu * 1.0, Math.sin(a) * cu * 0.92))
    }
    out.polys.push(body)
    const crown = [
      P(L - r * 0.15, -r * 0.32),
      P(L + r * 0.25, -r * 0.42),
      P(L + r * 0.05, -r * 0.1),
      P(L + r * 0.4, 0),
      P(L + r * 0.05, r * 0.1),
      P(L + r * 0.25, r * 0.42),
      P(L - r * 0.15, r * 0.32),
    ]
    out.polys.push(crown)
    // seeds show through a split
    for (let i = 0; i < 9; i++) {
      const u = cu + (((i * 7) % 9) / 9 - 0.5) * cu * 1.1,
        v = (((i * 4) % 9) / 9 - 0.5) * cu * 0.9
      out.dots.push([...P(u, v), r * 0.07])
    }
    out.radius = L + r * 0.4
  },
}
