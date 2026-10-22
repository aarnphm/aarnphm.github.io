import {
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationNodeDatum,
} from 'd3'
import { debounce } from '../../util/debounce'
import { inquiriesSlug, type InquiryMapData, type InquiryNode } from '../../util/inquiry-map'
import { FullSlug, getFullSlug, resolveRelative } from '../../util/path'

type Body = SimulationNodeDatum & {
  i: number
  node: InquiryNode
  el: HTMLElement
  w: number
  h: number
  /** seed position, px */
  sx: number
  sy: number
  x: number
  y: number
}

const PAD = 4
const HALO_REACH = 180
const HALO_ALPHA = [0.55, 0.35, 0.15, 0.06]
const SAGE = '205,213,151'
const PEACH = '253,178,162'

const spread = (k: number, n: number, from: number, to: number) =>
  n <= 1 ? (from + to) / 2 : from + ((to - from) * k) / (n - 1)

// Deterministic scatter in [0, 1): the golden-ratio sequence, so a given node always lands in the
// same place and neighbours in the sequence never share a column.
const scatter = (k: number, salt: number) => (k * 0.6180339887 + salt) % 1

/**
 * Seeds positions as the hand-placed mock does: hubs down the left (what joins pages) or across the
 * top (what joins sections), one column of pages per kind, sections at the centre of what they touch.
 */
function seed(bodies: Body[], adj: number[][], w: number, h: number) {
  const isFixed = (b: Body) => b.node.cls !== 'theme'
  // hubs keep the order the data declares them in (a1..c1, automne..été)
  const hubs = bodies.filter(b => b.node.cls === 'mvt').sort((a, b) => a.i - b.i)
  const sideways = hubs.filter(
    b => adj[b.i].filter(j => bodies[j].node.cls === 'pg').length * 2 >= adj[b.i].length,
  )
  const across = hubs.filter(b => !sideways.includes(b))
  sideways.forEach((b, k) => {
    b.sx = (0.05 + 0.05 * scatter(k, 0.3)) * w
    b.sy = spread(k, sideways.length, 0.14, 0.88) * h
  })
  across.forEach((b, k) => {
    b.sx = spread(k, across.length, 0.22, 0.82) * w
    b.sy = (0.05 + 0.04 * scatter(k, 0.6)) * h
  })
  // lessons run down their own column in episode order, topics down another in the order of the
  // levels that cite them, so a reader finds ép. 14 between ép. 13 and ép. 15
  const episodeOf = (b: Body) => /episode-(\d+)$/.exec(b.node.id)?.[1]
  const pages = bodies.filter(b => b.node.cls === 'pg')
  const lessons = pages.filter(b => episodeOf(b) !== undefined)
  const topics = pages.filter(b => episodeOf(b) === undefined)
  lessons.sort((a, b) => Number(episodeOf(a)) - Number(episodeOf(b)))
  const rank = (b: Body) =>
    Math.min(...adj[b.i].filter(j => bodies[j].node.cls === 'mvt'), Infinity)
  topics.sort((a, b) => rank(a) - rank(b) || a.i - b.i)
  // order still reads top to bottom, but each page drifts sideways inside a wide band, so the
  // field looks scattered instead of two stacks
  const bands: [Body[], number, number, number][] = [
    [topics, 0.3, 0.36, 0.2],
    [lessons, 0.5, 0.42, 0.7],
  ]
  for (const [column, from, span, salt] of bands) {
    column.forEach((b, k) => {
      b.sx = (from + span * scatter(k, salt)) * w
      b.sy = (spread(k, column.length, 0.07, 0.93) + 0.03 * (scatter(k, salt + 0.4) - 0.5)) * h
    })
  }
  bodies
    .filter(b => !isFixed(b))
    .forEach((b, k) => {
      const near = adj[b.i].map(j => bodies[j]).filter(isFixed)
      b.sx = near.length
        ? near.reduce((s, o) => s + o.sx, 0) / near.length
        : (0.35 + 0.3 * ((k * 0.618) % 1)) * w
      b.sy = near.length
        ? near.reduce((s, o) => s + o.sy, 0) / near.length
        : (0.2 + 0.6 * ((k * 0.382) % 1)) * h
    })
  for (const b of bodies) {
    b.x = b.sx
    b.y = b.sy
  }
}

/** Pushes overlapping rectangles apart along the shorter overlap. */
function rectCollide(bodies: Body[], w: number, h: number) {
  return (): boolean => {
    let moved = false
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const p = bodies[i]
        const q = bodies[j]
        const dx = q.x - p.x
        const dy = q.y - p.y
        const ox = (p.w + q.w) / 2 + PAD - Math.abs(dx)
        const oy = (p.h + q.h) / 2 + PAD - Math.abs(dy)
        if (ox <= 0 || oy <= 0) continue
        moved = true
        // page capsules and hubs hold their columns; theme boxes give way first
        const wp = p.node.cls === 'theme' ? 1 : 0.15
        const wq = q.node.cls === 'theme' ? 1 : 0.15
        const share = wp / (wp + wq)
        if (ox < oy) {
          const s = (dx < 0 ? -1 : 1) * ox
          p.x -= s * share
          q.x += s * (1 - share)
        } else {
          const s = (dy < 0 ? -1 : 1) * oy
          p.y -= s * share
          q.y += s * (1 - share)
        }
      }
    }
    for (const b of bodies) {
      b.x = Math.min(w - b.w / 2, Math.max(b.w / 2, b.x))
      b.y = Math.min(h - b.h / 2, Math.max(b.h / 2, b.y))
    }
    return moved
  }
}

/** Two 45° legs and a straight run between them. */
function octilinear(x1: number, y1: number, x2: number, y2: number): string {
  const dx = x2 - x1
  const dy = y2 - y1
  const ax = Math.abs(dx)
  const ay = Math.abs(dy)
  const sx = dx < 0 ? -1 : 1
  const sy = dy < 0 ? -1 : 1
  if (ay >= ax) {
    const k = ax / 2
    return `M${x1},${y1} l${sx * k},${sy * k} V${y2 - sy * k} L${x2},${y2}`
  }
  const k = ay / 2
  return `M${x1},${y1} l${sx * k},${sy * k} H${x2 - sx * k} L${x2},${y2}`
}

function mount(root: HTMLElement, data: InquiryMapData, slug: FullSlug, signal: AbortSignal) {
  const svg = root.querySelector<SVGSVGElement>('.inquiry-map-links')!
  const halo = root.querySelector<HTMLCanvasElement>('.inquiry-map-halo')!
  const list = root.querySelector<HTMLElement>('.inquiry-map-list')!
  const hrefOf = (node: InquiryNode) => {
    const [path, hash] = node.href.split('#')
    return resolveRelative(slug, path as FullSlug) + (hash ? `#${hash}` : '')
  }

  const adj: number[][] = data.nodes.map(() => [])
  for (const [a, b] of data.links) {
    adj[a].push(b)
    adj[b].push(a)
  }
  const maxW = Math.max(1, ...data.nodes.filter(n => n.cls === 'theme').map(n => n.w))

  // the nodes come in weight order, so Tab walks the heavy ones first
  const order = data.nodes
    .map((_, i) => i)
    .sort((a, b) => data.nodes[b].w - data.nodes[a].w || a - b)
  const bodies: Body[] = []
  for (const i of order) {
    const node = data.nodes[i]
    // a node that names no page is a plain label, still focusable so the keyboard can hover it
    const el = document.createElement(node.href ? 'a' : 'span')
    el.className = `nd ${node.cls}`
    if (el instanceof HTMLAnchorElement) el.href = hrefOf(node)
    else el.tabIndex = 0
    el.dataset.id = node.id
    el.textContent = node.label
    if (node.cls === 'theme') {
      el.style.setProperty('--scale', String(0.78 + 0.5 * Math.sqrt(node.w / maxW)))
    }
    root.appendChild(el)
    bodies[i] = { i, node, el, w: 0, h: 0, sx: 0, sy: 0, x: 0, y: 0 }
  }

  const paths: SVGPathElement[] = data.links.map(() =>
    document.createElementNS('http://www.w3.org/2000/svg', 'path'),
  )
  svg.append(...paths)

  const layout = () => {
    const w = root.clientWidth
    const h = root.clientHeight
    if (w === 0 || h === 0) return
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`)
    for (const b of bodies) {
      const scale = Number(b.el.style.getPropertyValue('--scale')) || 1
      b.w = b.el.offsetWidth * scale
      b.h = b.el.offsetHeight * scale
    }
    seed(bodies, adj, w, h)
    const collide = rectCollide(bodies, w, h)
    const sim = forceSimulation<Body>(bodies)
      .force(
        'link',
        forceLink<Body, { source: number; target: number }>(
          data.links.map(([source, target]) => ({ source, target })),
        )
          .id(b => b.i)
          .distance(70)
          .strength(0.06),
      )
      .force('charge', forceManyBody<Body>().strength(-40))
      .force(
        'x',
        forceX<Body>(b => b.sx).strength(b => (b.node.cls === 'theme' ? 0.12 : 0.5)),
      )
      .force(
        'y',
        forceY<Body>(b => b.sy).strength(b => (b.node.cls === 'theme' ? 0.12 : 0.5)),
      )
      .force('collide', collide)
      .stop()
    for (let tick = 0; tick < 300; tick++) sim.tick()
    // settle until no two boxes overlap; the clamp to the frame can undo a push, so allow many passes
    for (let pass = 0; pass < 2000 && collide(); pass++);
    for (const b of bodies) {
      b.el.style.left = `${b.x}px`
      b.el.style.top = `${b.y}px`
    }
    data.links.forEach(([a, b], k) => {
      paths[k].setAttribute('d', octilinear(bodies[a].x, bodies[a].y, bodies[b].x, bodies[b].y))
    })
  }

  // hover: the node and its neighbours within three hops, direct links in black
  const hover = (id: number) => {
    const dist = new Map<number, number>([[id, 0]])
    const queue = [id]
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head]
      const d = dist.get(u)!
      if (d >= 3) continue
      for (const v of adj[u]) {
        if (!dist.has(v)) {
          dist.set(v, d + 1)
          queue.push(v)
        }
      }
    }
    root.classList.add('hov')
    for (const b of bodies) {
      b.el.classList.remove('h0', 'h1', 'h2', 'h3')
      const d = dist.get(b.i)
      if (d !== undefined) b.el.classList.add(`h${d}`)
    }
    data.links.forEach(([a, b], k) => paths[k].classList.toggle('h1', a === id || b === id))
    paintHalo(dist)
  }
  const clear = () => {
    root.classList.remove('hov')
    for (const b of bodies) b.el.classList.remove('h0', 'h1', 'h2', 'h3')
  }
  const paintHalo = (dist: Map<number, number>) => {
    const w = root.clientWidth
    const h = root.clientHeight
    const dpr = window.devicePixelRatio || 1
    halo.width = w * dpr
    halo.height = h * dpr
    const ctx = halo.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)
    for (const [i, d] of dist) {
      const b = bodies[i]
      const rgb = d === 0 ? PEACH : SAGE
      const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, HALO_REACH)
      g.addColorStop(0, `rgba(${rgb},${HALO_ALPHA[d]})`)
      g.addColorStop(1, `rgba(${rgb},0)`)
      ctx.fillStyle = g
      ctx.fillRect(b.x - HALO_REACH, b.y - HALO_REACH, 2 * HALO_REACH, 2 * HALO_REACH)
    }
  }
  for (const b of bodies) {
    b.el.addEventListener('mouseenter', () => hover(b.i), { signal })
    b.el.addEventListener('focus', () => hover(b.i), { signal })
    b.el.addEventListener('mouseleave', clear, { signal })
    b.el.addEventListener('blur', clear, { signal })
  }

  // narrow screens: the same nodes as links, one group per hub, then the rest
  const group = (title: string | undefined, ids: number[]) => {
    if (ids.length === 0) return
    const box = document.createElement('div')
    if (title) {
      const label = document.createElement('b')
      label.textContent = title
      box.append(label)
    }
    ids.forEach((i, k) => {
      if (k > 0) box.append(' · ')
      const node = data.nodes[i]
      if (node.href) {
        const a = document.createElement('a')
        a.href = hrefOf(node)
        a.textContent = node.label
        box.append(a)
      } else box.append(node.label)
    })
    list.append(box)
  }
  const seen = new Set<number>()
  // season and level groups keep the order the data declares them in
  for (const i of order.filter(k => data.nodes[k].cls === 'mvt').sort((a, b) => a - b)) {
    const members = adj[i].filter(j => data.nodes[j].cls !== 'mvt')
    members.forEach(j => seen.add(j))
    seen.add(i)
    group(data.nodes[i].label, members)
  }
  group(
    undefined,
    order.filter(i => !seen.has(i)),
  )

  layout()
  const relayout = debounce(layout, 150)
  // label widths change when the web font swaps in
  void document.fonts?.ready.then(() => {
    if (!signal.aborted) layout()
  })
  const observer = new ResizeObserver(relayout)
  observer.observe(root)
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      relayout.cancel()
    },
    { once: true },
  )
}

document.addEventListener('nav', async () => {
  const root = document.querySelector<HTMLElement>('[data-inquiry-map]')
  if (!root) return
  const controller = new AbortController()
  window.addCleanup(() => controller.abort())
  const slug = getFullSlug(window)
  try {
    const res = await fetch(`${resolveRelative(slug, inquiriesSlug(slug) as FullSlug)}.json`, {
      signal: controller.signal,
    })
    if (!res.ok) return
    const data = (await res.json()) as InquiryMapData
    if (!controller.signal.aborted) mount(root, data, slug, controller.signal)
  } catch {
    // aborted by navigation, or the file is missing
  }
})
