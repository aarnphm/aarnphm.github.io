// A flower on the end of a stem (head: true), pointing along dir from (x, y) where the stem ends,
// size r. P(u, v) is a point u along the stem's direction and v across it; det: whether it has
// detail.
// It adds polys, lines and dots to out, and sets out.radius to how far it reaches.
// Bud: a closed bud between two sepals.

export default {
  family: 'flower',
  name: 'bud',
  head: true,
  draw({ x, y, r, out, blade, dir, det }) {
    const c = blade('teardrop', { x, y, dir, len: r * 1.6, width: 0.6, bend: 0, curl: 0, N: 14 })
    out.polys.push(c.polys[0])
    for (const sd of [1, -1]) {
      const s = blade('lance', {
        x,
        y,
        dir: dir + sd * 0.55,
        len: r * 0.95,
        width: 0.35,
        bend: -sd * 0.6,
        curl: 0,
        N: 10,
      })
      out.polys.push(s.polys[0])
    }
    if (det) out.lines.push(c.axis.slice(4, 12))
    out.radius = r * 1.6
  },
}
