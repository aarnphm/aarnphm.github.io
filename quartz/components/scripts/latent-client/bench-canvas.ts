import { titleText } from '../../../util/latent/title'
import { drawBenchLinks } from './bench-links'
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

type Screen = { coords: Float32Array; xs: Float32Array; ys: Float32Array; depth: Float32Array }

// screen positions of the last frame, reused across frames (53k chunks would otherwise be three
// fresh arrays every frame) and by hover hit-testing
let screen: Screen | null = null

/** Every point of the bench onto the canvas at once: the projectPoint math with the trig hoisted. */
function projectAll(bench: Bench, w: number, h: number): Screen {
  const c = bench.coords!
  const n = c.length / 3
  if (!screen || screen.xs.length !== n)
    screen = {
      coords: c,
      xs: new Float32Array(n),
      ys: new Float32Array(n),
      depth: new Float32Array(n),
    }
  screen.coords = c
  const { xs, ys, depth } = screen
  const s = (Math.min(w, h) * 0.42 * bench.zoom) / bench.extent
  const cy = Math.cos(bench.yaw)
  const sy = Math.sin(bench.yaw)
  const cp = Math.cos(bench.pitch)
  const sp = Math.sin(bench.pitch)
  const ox = w / 2 + bench.panX
  const oy = h / 2 + bench.panY
  const inv = 1 / bench.extent
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const x = c[j]
    const y = c[j + 1]
    const z = c[j + 2]
    const rz = x * sy + z * cy
    xs[i] = ox + (x * cy - z * sy) * s
    ys[i] = oy - (y * cp - rz * sp) * s
    depth[i] = (y * sp + rz * cp) * inv
  }
  return screen
}

// depth bands for the points: each band × source is one path and one fill, far bands first, so
// a frame is at most 48 fills whatever the level (it was one fill per point, 53k at chunks)
const BANDS = 12
const KINDS = 4
// within a band the many go under the few: pdf (44k chunks), bases, images, then notes, so the
// notes read through the pdf cloud as they did when every point was sorted by depth
const KIND_ORDER = [1, 3, 2, 0]

function drawBenchPoints(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  at: Screen,
  shown: (i: number) => boolean,
  colours: string[],
): void {
  const kinds = bench.kinds!
  const n = kinds.length
  const docs = bench.level === 'docs'
  const radius = new Float32Array(BANDS)
  for (let b = 0; b < BANDS; b++) {
    const t = (b + 0.5) / BANDS
    radius[b] = docs ? 1.2 + t * 2.2 : 0.6 + t * 1.4
  }
  // a chunk dot is 1 to 4 device px across, where an antialiased square reads as the disc
  // does; squares raster in about a third of the time of 53k arcs
  const square = !docs
  const paths: (Path2D | undefined)[] = Array.from({ length: BANDS * KINDS })
  const TAU = Math.PI * 2
  for (let i = 0; i < n; i++) {
    if (!shown(i)) continue
    const t = (at.depth[i] + 1) / 2
    const b = t <= 0 ? 0 : t >= 1 ? BANDS - 1 : (t * BANDS) | 0
    const slot = b * KINDS + kinds[i]
    const path = (paths[slot] ??= new Path2D())
    const r = radius[b]
    const x = at.xs[i]
    const y = at.ys[i]
    // same area as the disc: side r·√π
    if (square) path.rect(x - r * 0.886, y - r * 0.886, r * 1.772, r * 1.772)
    else {
      path.moveTo(x + r, y)
      path.arc(x, y, r, 0, TAU)
    }
  }
  for (let b = 0; b < BANDS; b++) {
    ctx.globalAlpha = 0.25 + ((b + 0.5) / BANDS) * 0.65
    for (const k of KIND_ORDER) {
      const path = paths[b * KINDS + k]
      if (!path) continue
      ctx.fillStyle = colours[k] ?? colours[0]
      ctx.fill(path)
    }
  }
}

export function hoverBench(rt: Runtime, px: number, py: number): void {
  const bench = rt.bench
  if (!bench || !bench.coords) return
  const n = bench.coords.length / 3
  // the last frame's screen positions hold while no frame is pending (every view change asks
  // for one, so a settled view is exactly what was drawn); otherwise project afresh
  const at =
    screen && screen.coords === bench.coords && !bench.frame
      ? screen
      : projectAll(bench, bench.w, bench.h)
  let best = -1
  let bestDist = 36
  for (let i = 0; i < n; i++) {
    if (bench.kinds && (bench.hidden >> bench.kinds[i]) & 1) continue
    const d = (at.xs[i] - px) ** 2 + (at.ys[i] - py) ** 2
    if (d < bestDist) {
      bestDist = d
      best = i
    }
  }
  if (best !== bench.hover) {
    bench.hover = best
    drawBench(rt)
    const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
    hoverCounts = ''
    if (status && best >= 0) setBenchStatus(status, hoverName(rt, bench, best))
  }
}

function hoverName(rt: Runtime, bench: Bench, i: number): string {
  const doc = bench.level === 'docs' ? rt.docs[i] : docOfRow(rt, i)
  return `${doc.s} · ${titleText(doc.t)}`
}

// what the status line last said about the hovered point's edges, so a redraw only touches the
// line when a count lands
let hoverCounts = ''

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
    const at = projectAll(bench, w, h)
    drawBenchAxes(ctx, bench, w, h, colours[3], style)
    const shown = (i: number) => !((bench.hidden >> bench.kinds![i]) & 1)
    const mark = bench.level === 'docs' ? rt.open : rt.focus
    const selected = mark >= 0 && mark < n ? mark : -1
    drawBenchPoints(ctx, bench, at, shown, colours)
    // edges are drawn for two points only (bench-links.ts): the marked one once a point has
    // been clicked, and the hovered one, which also shows its nearest chunks
    const redraw = () => drawBench(rt)
    const edgeColours = { ink, near: colours[3] }
    const hovered = bench.hover >= 0 && bench.hover < n ? bench.hover : -1
    if (bench.picked && selected !== hovered)
      drawBenchLinks(ctx, rt, bench, at, shown, selected, edgeColours, redraw)
    const counts = drawBenchLinks(ctx, rt, bench, at, shown, hovered, edgeColours, redraw, true)
    if (counts && hovered >= 0) {
      const parts = [
        counts.near === null ? null : `${counts.near} near`,
        counts.out === null ? null : `${counts.out} out`,
        counts.back === null ? null : `${counts.back} back`,
      ].filter(Boolean)
      const said = parts.join(' · ')
      const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
      if (status && said !== hoverCounts) {
        hoverCounts = said
        setBenchStatus(status, [hoverName(rt, bench, hovered), ...parts].join(' · '))
      }
    }
    if (selected >= 0 && shown(selected)) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = ink
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(at.xs[selected], at.ys[selected], 6, 0, Math.PI * 2)
      ctx.stroke()
    }
    if (bench.hover >= 0 && bench.hover < n && shown(bench.hover)) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = colours[1]
      ctx.beginPath()
      ctx.arc(at.xs[bench.hover], at.ys[bench.hover], 5, 0, Math.PI * 2)
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
