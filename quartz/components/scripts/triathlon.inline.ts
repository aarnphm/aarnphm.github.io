import { type Sport, SPORT_ICON, type StravaActivityDetail } from '../../plugins/stores/strava'

export {}

const SVGNS = 'http://www.w3.org/2000/svg'
const KM_TO_MI = 0.621371
const FT_PER_KM = 3280.84

const el = (tag: string, cls?: string, text?: string): HTMLElement => {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

const svg = (tag: string, attrs: Record<string, string | number>): SVGElement => {
  const e = document.createElementNS(SVGNS, tag)
  for (const k in attrs) e.setAttribute(k, String(attrs[k]))
  return e
}

const dist = (km: number, sport: Sport): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  const mi = km * KM_TO_MI
  return mi < 1 ? `${Math.round(km * FT_PER_KM)} ft` : `${mi.toFixed(1)} mi`
}
const dur = (s: number): string => {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h > 0 ? `${h}h${m.toString().padStart(2, '0')}'` : `${m}'`
}
const clock = (s: number): string =>
  `${Math.floor(s / 60)}:${Math.round(s % 60)
    .toString()
    .padStart(2, '0')}`
const rate = (sport: Sport, km: number, s: number): string => {
  const mi = km * KM_TO_MI
  if (sport === 'bike') return `${(mi / (s / 3600)).toFixed(1)} mph`
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  return `${clock(s / mi)} /mi`
}

const buildIcon = (sport: Sport): SVGElement => {
  const s = svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of SPORT_ICON[sport]) s.appendChild(svg('path', { d }))
  return s
}

const buildRoute = (route: StravaActivityDetail['route']): SVGElement => {
  const pad = 6
  const span = 100 - pad * 2
  let d = ''
  route.forEach((p, i) => {
    d += `${i ? 'L' : 'M'} ${(pad + p.x * span).toFixed(2)} ${(pad + (1 - p.y) * span).toFixed(2)} `
  })
  const s = svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(svg('path', { d, class: 'tri-route-path' }))
  return s
}

const buildElevation = (d: StravaActivityDetail): HTMLElement => {
  const w = 100
  const h = 30
  const pad = 2
  const maxD = d.route[d.route.length - 1].d || 1
  const altSpan = Math.max(1, d.maxAlt - d.minAlt)
  const px = (km: number): number => (km / maxD) * w
  const py = (alt: number): number => h - pad - ((alt - d.minAlt) / altSpan) * (h - 2 * pad)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    area += `L ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const s = svg('svg', { class: 'tri-elev', viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none' })
  s.appendChild(svg('path', { d: area, class: 'tri-elev-area' }))
  s.appendChild(svg('path', { d: line, class: 'tri-elev-line' }))
  const wrap = el('div', 'tri-elev-wrap')
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-elev-d', `+${d.elevationM} m`),
    el('span', 'tri-elev-d', `−${d.descentM} m`),
    el('span', 'tri-elev-range', `${Math.round(d.minAlt)}–${Math.round(d.maxAlt)} m`),
  )
  wrap.append(s, cap)
  return wrap
}

const buildPool = (d: StravaActivityDetail): HTMLElement => {
  const lengths = Math.max(1, Math.round((d.distanceKm * 1000) / 25))
  const wrap = el('div', 'tri-pool-wrap')
  const s = svg('svg', {
    class: 'tri-route tri-pool',
    viewBox: '0 0 100 56',
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(
    svg('rect', { x: 6, y: 12, width: 88, height: 32, rx: 16, ry: 16, class: 'tri-pool-lane' }),
  )
  s.appendChild(svg('line', { x1: 22, y1: 28, x2: 78, y2: 28, class: 'tri-pool-mid' }))
  wrap.append(s, el('span', 'tri-pool-cap', `${lengths} × 25m`))
  return wrap
}

const statRow = (label: string, value: string): HTMLElement => {
  const tr = document.createElement('tr')
  tr.append(el('th', 'tri-act-stat-k', label), el('td', 'tri-act-stat-v', value))
  return tr
}

const renderDetail = (d: StravaActivityDetail): HTMLElement => {
  const wrap = el('section', 'tri-act')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildIcon(d.sport))
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const body = document.createElement('tbody')
  body.append(
    statRow('distance', dist(d.distanceKm, d.sport)),
    statRow('time', dur(d.movingTimeS)),
    statRow(d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)),
  )
  if (d.sport !== 'swim')
    body.appendChild(statRow('elevation', `${d.elevationM.toLocaleString('en-US')} m`))
  if (d.avgHr) body.appendChild(statRow('avg hr', `${d.avgHr} bpm`))
  stats.appendChild(body)
  wrap.appendChild(stats)

  if (d.sport === 'swim') {
    const figs = el('div', 'tri-act-figs')
    figs.appendChild(buildPool(d))
    wrap.appendChild(figs)
  } else if (d.route.length >= 2) {
    const figs = el('div', 'tri-act-figs')
    figs.append(buildRoute(d.route), buildElevation(d))
    wrap.appendChild(figs)
  }
  return wrap
}

const setup = (root: HTMLElement): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let active: HTMLElement | null = null
  let activeIdx = -1
  let details: Record<string, StravaActivityDetail> | null = null
  let pinned = false
  let hideTimer = 0

  const buildCard = (bar: HTMLElement): HTMLElement => {
    const card = el('div', 'tri-pop-card')
    card.appendChild(el('div', 'tri-pop-date', bar.dataset.date ?? ''))
    const idsAttr = bar.dataset.ids
    if (!idsAttr) {
      card.appendChild(el('div', 'tri-pop-rest', 'Rest'))
    } else if (details) {
      for (const id of idsAttr.split(',')) {
        const d = details[id]
        if (d) card.appendChild(renderDetail(d))
      }
    } else {
      card.appendChild(el('div', 'tri-pop-rest', '·'))
    }
    return card
  }

  const place = (cx: number, cy: number) => {
    const r = pop.getBoundingClientRect()
    const gap = 18
    let left = cx + gap
    if (left + r.width > window.innerWidth - 8) left = cx - gap - r.width
    left = Math.max(8, left)
    let top = cy - r.height / 2
    top = Math.max(8, Math.min(top, window.innerHeight - r.height - 8))
    pop.style.left = `${left}px`
    pop.style.top = `${top}px`
  }

  const nearest = (clientX: number): number => {
    let best = Infinity
    let found = -1
    bars.forEach((bar, i) => {
      const r = bar.getBoundingClientRect()
      const d = Math.abs(r.left + r.width / 2 - clientX)
      if (d < best) {
        best = d
        found = i
      }
    })
    return found
  }

  const showFor = (idx: number, cx: number, cy: number) => {
    const bar = bars[idx]
    if (bar !== active) {
      const dir = activeIdx === -1 ? 0 : Math.sign(idx - activeIdx)
      if (active) active.classList.remove('tri-bar--active')
      active = bar
      activeIdx = idx
      bar.classList.add('tri-bar--active')
      const card = buildCard(bar)
      pop.replaceChildren(card)
      pop.scrollTop = 0
      if (!reduce)
        card.animate(
          [
            { opacity: 0, transform: `translateX(${dir * 12}px)` },
            { opacity: 1, transform: 'none' },
          ],
          { duration: 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
        )
    }
    place(cx, cy)
    root.classList.add('tri-hovering')
  }

  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    activeIdx = -1
    pinned = false
    root.classList.remove('tri-hovering')
  }

  const onMove = (event: MouseEvent) => {
    if (pinned) return
    window.clearTimeout(hideTimer)
    const idx = nearest(event.clientX)
    if (idx >= 0) showFor(idx, event.clientX, event.clientY)
  }
  const onBarsLeave = () => {
    if (!pinned) hideTimer = window.setTimeout(hide, 140)
  }
  const onPopEnter = () => {
    window.clearTimeout(hideTimer)
    pinned = true
  }
  const onPopLeave = () => {
    pinned = false
    hideTimer = window.setTimeout(hide, 140)
  }

  const path = root.dataset.detailPath
  if (path)
    fetch(path)
      .then(res => res.json())
      .then((data: Record<string, StravaActivityDetail>) => {
        details = data
        if (active) pop.replaceChildren(buildCard(active))
      })
      .catch(() => {})

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onBarsLeave)
  pop.addEventListener('mouseenter', onPopEnter)
  pop.addEventListener('mouseleave', onPopLeave)

  return () => {
    window.clearTimeout(hideTimer)
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
  }
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (!root) return
  const cleanup = setup(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
