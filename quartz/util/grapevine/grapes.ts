import type { KindModule } from './vendor/initialis/kinds.js'

// A bunch of grapes as an initialis flower kind that sits on the end of a stem (head: true): a
// cone of overlapping round berries hanging along the stem's direction, widest near the stalk.
// Each berry is its own polygon so the painter can edge them and keep them apart.

export const GRAPES: KindModule = {
  family: 'flower',
  name: 'grapes',
  head: true,
  draw({ r, out, rot, P, circlePoly }) {
    out.kind = 'grapes'
    // A bunch the engine could only fit this small would print as a speck; leave the stem bare.
    if (r < 16) {
      out.radius = 0
      return
    }
    const b = r * 0.36
    const rows = [3, 2, 3, 2, 1]
    let u = b * 1.1
    const n = rows.length
    for (let row = 0; row < n; row++) {
      const count = rows[row]
      const shrink = 1 - 0.1 * Math.max(0, row - 2)
      const br = b * shrink
      // a little sway from the stem's own random turn, so no two bunches hang alike
      const sway = Math.sin(rot * 3 + row) * b * 0.25
      for (let k = 0; k < count; k++) {
        const v = (k - (count - 1) / 2) * br * 1.55 + sway
        out.polys.push(circlePoly(...P(u, v), br, 18))
      }
      u += br * 1.3
    }
    out.radius = u
  },
}
