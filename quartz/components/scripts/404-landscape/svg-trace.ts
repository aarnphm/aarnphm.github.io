// A point at least every scene unit along an outline, in the element's own coordinates, one list
// per subpath. The ruin is drawn with lines and circular arcs only, so the path data is walked
// directly: getPointAtLength re-measures every arc from the start of the path on each call, which
// cost 670 ms per load for the ring and the soffit.
export function trace(d: string) {
  const tok = d.match(/[a-zA-Z]|-?(?:\d+\.?\d*|\.\d+)/g) ?? []
  const chains: number[][] = []
  let pts: number[] = []
  let k = 0
  let cmd = ''
  let x = 0
  let y = 0
  let x0 = 0
  let y0 = 0
  const num = () => Number(tok[k++])
  const line = (tx: number, ty: number) => {
    const n = Math.max(1, Math.ceil(Math.hypot(tx - x, ty - y)))
    for (let s = 1; s <= n; s++) pts.push(x + ((tx - x) * s) / n, y + ((ty - y) * s) / n)
    x = tx
    y = ty
  }
  // Endpoint to centre form, for a circle with no rotation (SVG 1.1 implementation notes, F.6.5).
  const arc = (r: number, large: number, sweep: number, tx: number, ty: number) => {
    const hx = (x - tx) / 2
    const hy = (y - ty) / 2
    const h2 = hx * hx + hy * hy
    if (h2 === 0) return
    r = Math.max(r, Math.sqrt(h2))
    const c = Math.sqrt(Math.max(0, (r * r - h2) / h2)) * (large === sweep ? -1 : 1)
    const cx = c * hy + (x + tx) / 2
    const cy = -c * hx + (y + ty) / 2
    const a0 = Math.atan2(y - cy, x - cx)
    let da = Math.atan2(ty - cy, tx - cx) - a0
    if (sweep && da < 0) da += 2 * Math.PI
    if (!sweep && da > 0) da -= 2 * Math.PI
    const n = Math.max(1, Math.ceil(Math.abs(da) * r))
    for (let s = 1; s <= n; s++) {
      const a = a0 + (da * s) / n
      pts.push(cx + r * Math.cos(a), cy + r * Math.sin(a))
    }
    x = tx
    y = ty
  }
  while (k < tok.length) {
    if (/[a-zA-Z]/.test(tok[k])) cmd = tok[k++]
    const rel = cmd === cmd.toLowerCase()
    const ox = rel ? x : 0
    const oy = rel ? y : 0
    switch (cmd.toUpperCase()) {
      case 'M':
        if (pts.length) chains.push(pts)
        x = x0 = ox + num()
        y = y0 = oy + num()
        pts = [x, y]
        // Pairs after a moveto are linetos.
        cmd = rel ? 'l' : 'L'
        break
      case 'L':
        line(ox + num(), oy + num())
        break
      case 'H':
        line(ox + num(), y)
        break
      case 'V':
        line(x, oy + num())
        break
      case 'A': {
        const r = num()
        k += 2
        const large = num()
        const sweep = num()
        arc(r, large, sweep, ox + num(), oy + num())
        break
      }
      case 'Z':
        line(x0, y0)
        break
      default:
        k++
    }
  }
  if (pts.length) chains.push(pts)
  return chains
}

// The pixels each subpath passes through, in order and one pixel thick: repeats are dropped, and
// so is the corner pixel of every L-shaped step, which keeps diagonals from doubling up.
export function run(chains: number[][], m: DOMMatrix) {
  const out: number[] = []
  for (const pts of chains) {
    const start = out.length
    for (let s = 0; s < pts.length; s += 2) {
      const i = Math.floor(m.a * pts[s] + m.c * pts[s + 1] + m.e)
      const j = Math.floor(m.b * pts[s] + m.d * pts[s + 1] + m.f)
      const n = out.length - start
      if (n && out[start + n - 2] === i && out[start + n - 1] === j) continue
      if (
        n >= 4 &&
        Math.abs(i - out[start + n - 4]) === 1 &&
        Math.abs(j - out[start + n - 3]) === 1
      )
        out.length -= 2
      out.push(i, j)
    }
  }
  return out
}
