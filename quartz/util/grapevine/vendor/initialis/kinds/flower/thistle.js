// A flower on the end of a stem (head: true), pointing along dir from (x, y) where the stem ends,
// size r. P(u, v) is a point u along the stem's direction and v across it; det: whether it has
// detail.
// It adds polys, lines and dots to out, and sets out.radius to how far it reaches.
// Thistle: a round head with a brush of spines.

export default {
  family: 'flower',
  name: 'thistle',
  head: true,
  draw({ x, y, r, out, blade, circlePoly, dir, P, det }) {
    const head = circlePoly(...P(r * 0.7, 0), r * 0.62, 28)
    out.polys.push(head)
    for (let i = -3; i <= 3; i++) {
      const a = dir + i * 0.22
      const b = blade('willow', {
        x: P(r * 0.95, 0)[0],
        y: P(r * 0.95, 0)[1],
        dir: a,
        len: r * 1.0,
        width: 0.16,
        bend: 0,
        curl: 0,
        N: 8,
      })
      out.polys.push(b.polys[0])
    }
    if (det)
      for (let i = 0; i < 4; i++)
        out.lines.push([P(r * 0.3, (i - 1.5) * r * 0.3), P(r * 1.05, (i - 1.5) * r * 0.15)])
    out.radius = r * 2
  },
}
