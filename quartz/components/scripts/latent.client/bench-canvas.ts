import { docOfRow, type Bench, type Runtime } from './runtime'

// Match the backing store to device pixels. device-pixel-content-box gives the exact snapped
// size (Chromium, Firefox) and re-fires when the window moves to a screen with another density;
// Safari falls back to css size × devicePixelRatio. Capped at 3× so a zoomed-in page does not
// blow the fill cost up.
export function fitBenchCanvas(rt: Runtime, canvas: HTMLCanvasElement): void {
  const apply = (cssW: number, cssH: number, devW?: number, devH?: number) => {
    const bench = rt.bench
    if (!bench || cssW <= 0 || cssH <= 0) return
    const dpr = Math.min(3, window.devicePixelRatio || 1)
    let pw = Math.round(cssW * dpr)
    let ph = Math.round(cssH * dpr)
    if (devW && devH && devW / cssW <= 3) {
      pw = devW
      ph = devH
    }
    bench.w = cssW
    bench.h = cssH
    bench.dpr = pw / cssW
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    drawBench(rt)
  }
  const observer = new ResizeObserver(entries => {
    const entry = entries[entries.length - 1]
    const box = entry.contentBoxSize?.[0]
    const dev = entry.devicePixelContentBoxSize?.[0]
    const cssW = box ? box.inlineSize : entry.contentRect.width
    const cssH = box ? box.blockSize : entry.contentRect.height
    apply(cssW, cssH, dev?.inlineSize, dev?.blockSize)
  })
  try {
    observer.observe(canvas, { box: 'device-pixel-content-box' })
  } catch {
    observer.observe(canvas)
  }
  rt.signal.addEventListener('abort', () => observer.disconnect())
}

function projected(bench: Bench, i: number, w: number, h: number): [number, number, number] {
  const c = bench.coords!
  return projectPoint(bench, c[i * 3], c[i * 3 + 1], c[i * 3 + 2], w, h)
}

/** A point of the bench's space on the canvas: [x, y, depth in -1..1]. */
function projectPoint(
  bench: Bench,
  x: number,
  y: number,
  z: number,
  w: number,
  h: number,
): [number, number, number] {
  const s = (Math.min(w, h) * 0.42 * bench.zoom) / bench.extent
  const cy = Math.cos(bench.yaw)
  const sy = Math.sin(bench.yaw)
  const cp = Math.cos(bench.pitch)
  const sp = Math.sin(bench.pitch)
  const rx = x * cy - z * sy
  const rz = x * sy + z * cy
  const ry = y * cp - rz * sp
  const depth = y * sp + rz * cp
  return [w / 2 + bench.panX + rx * s, h / 2 + bench.panY - ry * s, depth / bench.extent]
}

export function hoverBench(rt: Runtime, px: number, py: number): void {
  const bench = rt.bench
  if (!bench || !bench.coords) return
  const n = bench.coords.length / 3
  let best = -1
  let bestDist = 36
  for (let i = 0; i < n; i++) {
    const [x, y] = projected(bench, i, bench.w, bench.h)
    const d = (x - px) ** 2 + (y - py) ** 2
    if (d < bestDist) {
      bestDist = d
      best = i
    }
  }
  if (best !== bench.hover) {
    bench.hover = best
    drawBench(rt)
    const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
    if (status && best >= 0) {
      const doc = bench.level === 'docs' ? rt.docs[best] : docOfRow(rt, best)
      setBenchStatus(status, `${doc.s} · ${doc.t}`)
    }
  }
}

/**
 * The hovered point's name on the status line's one line. A name wider than the line scrolls
 * to its end and back (latent.scss), at a reading pace, so it never wraps or moves the pane.
 */
function setBenchStatus(status: HTMLElement, text: string): void {
  const span = document.createElement('span')
  span.textContent = text
  status.replaceChildren(span)
  delete status.dataset.overflow
  const style = getComputedStyle(status)
  const room = status.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
  const shift = span.getBoundingClientRect().width - room
  if (shift <= 1) return
  status.style.setProperty('--lat-shift', `${-Math.ceil(shift)}px`)
  // about 40px a second, plus the holds at either end (a fifth of the cycle each)
  status.style.setProperty('--lat-marquee', `${(shift / 40 / 0.6).toFixed(2)}s`)
  status.dataset.overflow = ''
}

export function drawBench(rt: Runtime): void {
  const bench = rt.bench
  const canvas = rt.probe.querySelector<HTMLCanvasElement>('.lat-bench-canvas')
  if (!bench || !bench.coords || !bench.kinds || !canvas) return
  if (bench.frame) return
  bench.frame = requestAnimationFrame(() => {
    bench.frame = 0
    const ctx = canvas.getContext('2d')
    if (!ctx || !bench.coords || !bench.kinds) return
    const style = getComputedStyle(canvas)
    const ink = style.getPropertyValue('--dark').trim() || '#100f0f'
    const paper = style.getPropertyValue('--light').trim() || '#fffcf0'
    const colours = [
      ink,
      style.getPropertyValue('--lat-pdf').trim() || '#fdb2a2',
      style.getPropertyValue('--lat-image').trim() || '#cdd597',
      style.getPropertyValue('--gray').trim() || '#6f6e69',
    ]
    const w = bench.w
    const h = bench.h
    // draw in css px; the transform maps them onto the device-pixel backing store
    ctx.setTransform(bench.dpr, 0, 0, bench.dpr, 0, 0)
    ctx.fillStyle = paper
    ctx.fillRect(0, 0, w, h)
    const n = bench.coords.length / 3
    const order: number[] = Array.from({ length: n }, (_, i) => i)
    const depth = new Float32Array(n)
    const xs = new Float32Array(n)
    const ys = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const [x, y, z] = projected(bench, i, w, h)
      xs[i] = x
      ys[i] = y
      depth[i] = z
      order[i] = i
    }
    order.sort((a, b) => depth[a] - depth[b])
    drawBenchAxes(ctx, bench, w, h, colours[3], style)
    const openDoc = rt.open
    const focusRow = rt.focus
    for (const i of order) {
      const t = (depth[i] + 1) / 2
      const r = bench.level === 'docs' ? 1.2 + t * 2.2 : 0.6 + t * 1.4
      ctx.globalAlpha = 0.25 + t * 0.65
      ctx.fillStyle = colours[bench.kinds![i]] ?? ink
      ctx.beginPath()
      ctx.arc(xs[i], ys[i], r, 0, Math.PI * 2)
      ctx.fill()
    }
    const mark = bench.level === 'docs' ? openDoc : focusRow
    if (mark >= 0 && mark < n) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = ink
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(xs[mark], ys[mark], 6, 0, Math.PI * 2)
      ctx.stroke()
    }
    if (bench.hover >= 0 && bench.hover < n) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = colours[1]
      ctx.beginPath()
      ctx.arc(xs[bench.hover], ys[bench.hover], 5, 0, Math.PI * 2)
      ctx.stroke()
    }
    // the three axis names on one line along the floor, in the page's mono face (the canvas
    // inherits it from latent.scss), on a paper halo so the point cloud does not run through
    // the glyphs. The variance is in the table above, so only the names
    ctx.globalAlpha = 1
    ctx.font = `${style.fontSize} ${style.fontFamily}`
    ctx.textBaseline = 'alphabetic'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 3
    ctx.strokeStyle = paper
    ctx.fillStyle = ink
    const names = bench.axes.map(
      (a, slot) => `${'xyz'[slot]} ${axisBase(bench.meta?.axes[a].name ?? '')}`,
    )
    // the names read whole on one line: the type steps down to fit (to 80 %), and only a line
    // still too long cuts its longest name
    const room = w - 12
    const widthAt = (list: string[]) =>
      list.reduce((sum, name) => sum + ctx.measureText(name).width, 0) +
      ctx.measureText('  ').width * 2
    const base = parseFloat(style.fontSize) || 11
    const size = Math.max(base * 0.8, Math.min(base, (base * room) / widthAt(names)))
    ctx.font = `${size}px ${style.fontFamily}`
    const gap = ctx.measureText('  ').width
    const fitted = [...names]
    for (let over = widthAt(fitted) - room; over > 0; over = widthAt(fitted) - room) {
      let longest = 0
      for (let i = 1; i < 3; i++)
        if (ctx.measureText(fitted[i]).width > ctx.measureText(fitted[longest]).width) longest = i
      const next = fitText(ctx, fitted[longest], ctx.measureText(fitted[longest]).width - over)
      if (next === fitted[longest]) break
      fitted[longest] = next
    }
    let x = 6
    for (const text of fitted) {
      ctx.strokeText(text, x, h - 6)
      ctx.fillText(text, x, h - 6)
      x += ctx.measureText(text).width + gap
    }
  })
}

/** "pc1 · 6.2 %" → "pc1"; the variance share is shown on the axes tab. */
export function axisBase(name: string): string {
  return name.replace(/\s+·\s+[\d.]+\s*%$/, '')
}

function fitText(ctx: CanvasRenderingContext2D, text: string, room: number): string {
  if (ctx.measureText(text).width <= room) return text
  let cut = text.length
  while (cut > 2 && ctx.measureText(`${text.slice(0, cut)}…`).width > room) cut--
  return `${text.slice(0, cut)}…`
}

/**
 * The three axes through the origin, drawn under the points: a hairline from end to end with
 * ticks at the half and full extent, and the axis letter past the positive end.
 */
function drawBenchAxes(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  w: number,
  h: number,
  colour: string,
  style: CSSStyleDeclaration,
): void {
  const e = bench.extent
  const ends: [number, number, number][] = [
    [e, 0, 0],
    [0, e, 0],
    [0, 0, e],
  ]
  const [ox, oy] = projectPoint(bench, 0, 0, 0, w, h)
  ctx.save()
  ctx.globalAlpha = 0.7
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 1
  ctx.font = `${style.fontSize} ${style.fontFamily}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ends.forEach(([ex, ey, ez], slot) => {
    const at = (t: number) => projectPoint(bench, ex * t, ey * t, ez * t, w, h)
    const [ax, ay] = at(-1)
    const [bx, by] = at(1)
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.stroke()
    // ticks perpendicular to the axis as drawn; a foreshortened axis keeps short ticks
    const len = Math.hypot(bx - ax, by - ay)
    if (len > 8) {
      const nx = -(by - ay) / len
      const ny = (bx - ax) / len
      ctx.beginPath()
      for (const t of [-1, -0.5, 0.5, 1]) {
        const [tx, ty] = at(t)
        ctx.moveTo(tx - nx * 3, ty - ny * 3)
        ctx.lineTo(tx + nx * 3, ty + ny * 3)
      }
      ctx.stroke()
    }
    // the letter just past the positive end, pushed out from the origin
    const dx = bx - ox
    const dy = by - oy
    const d = Math.hypot(dx, dy) || 1
    ctx.fillText('xyz'[slot], bx + (dx / d) * 9, by + (dy / d) * 9)
  })
  ctx.restore()
}
