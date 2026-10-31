// A flower on the end of a stem (head: true), pointing along dir from (x, y) where the stem ends,
// size r. P(u, v) is a point u along the stem's direction and v across it; det: whether it has
// detail.
// It adds polys, lines and dots to out, and sets out.radius to how far it reaches.
// Tulip: three petals, the middle one in front.

export default {
  family: 'flower',
  name: 'tulip',
  head: true,
  draw({ x, y, r, out, blade, dir, det }) {
    const c = blade('teardrop', { x, y, dir, len: r * 1.9, width: 0.62, bend: 0, curl: 0, N: 16 })
    const l = blade('teardrop', {
      x,
      y,
      dir: dir - 0.42,
      len: r * 1.55,
      width: 0.55,
      bend: 0.5,
      curl: 0,
      N: 16,
    })
    const rr = blade('teardrop', {
      x,
      y,
      dir: dir + 0.42,
      len: r * 1.55,
      width: 0.55,
      bend: -0.5,
      curl: 0,
      N: 16,
    })
    out.polys.push(l.polys[0], rr.polys[0], c.polys[0])
    if (det) out.lines.push(c.axis.slice(2, 14))
    out.radius = r * 1.9
  },
}
