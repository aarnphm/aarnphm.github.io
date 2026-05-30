import type { Sport, StravaActivityDetail } from '../../plugins/stores/strava'

export {}

const SVGNS = 'http://www.w3.org/2000/svg'
const KM_TO_MI = 0.621371
const EMOJI: Record<Sport, string> = { swim: '🏊', bike: '🚴', run: '🏃' }

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

const miles1 = (km: number): string => (km * KM_TO_MI).toFixed(1)
const dur = (s: number): string => {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h > 0 ? `${h}h${m.toString().padStart(2, '0')}` : `${m}m`
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

const renderReadout = (container: HTMLElement, raw: string): void => {
  container.replaceChildren()
  if (!raw || raw === 'Rest') {
    container.appendChild(el('span', 'tri-r-rest', 'Rest'))
    return
  }
  for (const row of raw.split(';')) {
    const [emoji, dist, rt] = row.split('|')
    container.append(
      el('span', 'tri-r-emoji', emoji ?? ''),
      el('span', 'tri-r-dist', dist ?? ''),
      el('span', 'tri-r-sep', '·'),
      el('span', 'tri-r-rate', rt ?? ''),
    )
  }
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

const buildElevation = (d: StravaActivityDetail): SVGElement => {
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
  return s
}

const stat = (label: string, value: string): HTMLElement => {
  const s = el('div', 'tri-act-stat')
  s.append(el('span', 'tri-act-stat-v', value), el('span', 'tri-act-stat-k', label))
  return s
}

const renderDetail = (d: StravaActivityDetail): HTMLElement => {
  const wrap = el('section', 'tri-act')
  const head = el('div', 'tri-act-head')
  head.append(el('span', 'tri-act-emoji', EMOJI[d.sport]), el('span', 'tri-act-name', d.name))
  wrap.appendChild(head)

  const stats = el('div', 'tri-act-stats')
  stats.append(
    stat('distance', `${miles1(d.distanceKm)} mi`),
    stat('time', dur(d.movingTimeS)),
    stat(d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)),
    stat('elevation', `${d.elevationM.toLocaleString('en-US')} m`),
  )
  if (d.avgHr) stats.append(stat('avg hr', `${d.avgHr} bpm`))
  wrap.appendChild(stats)

  if (d.route.length >= 2) {
    const figs = el('div', 'tri-act-figs')
    figs.append(buildRoute(d.route), buildElevation(d))
    wrap.appendChild(figs)
  }
  return wrap
}

const setup = (root: HTMLElement): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const info = root.querySelector<HTMLElement>('.tri-info')
  const readout = root.querySelector<HTMLElement>('.tri-readout')
  const dateEl = root.querySelector<HTMLElement>('.tri-date')
  const panel = root.querySelector<HTMLElement>('.tri-panel')
  const panelBody = root.querySelector<HTMLElement>('.tri-panel-body')
  const closeBtn = root.querySelector<HTMLElement>('.tri-panel-close')
  const scrim = root.querySelector<HTMLElement>('.tri-scrim')
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (
    !barsEl ||
    !info ||
    !readout ||
    !dateEl ||
    !panel ||
    !panelBody ||
    !scrim ||
    bars.length === 0
  ) {
    return null
  }

  let active: HTMLElement | null = null
  let details: Record<string, StravaActivityDetail> | null = null

  const nearest = (clientX: number): HTMLElement | null => {
    let best = Infinity
    let found: HTMLElement | null = null
    for (const bar of bars) {
      const r = bar.getBoundingClientRect()
      const d = Math.abs(r.left + r.width / 2 - clientX)
      if (d < best) {
        best = d
        found = bar
      }
    }
    return found
  }

  const activate = (bar: HTMLElement) => {
    if (bar === active) return
    if (active) active.classList.remove('tri-bar--active')
    active = bar
    bar.classList.add('tri-bar--active')
    const sr = root.getBoundingClientRect()
    const br = bar.getBoundingClientRect()
    const x = br.left - sr.left + br.width / 2
    const frac = sr.width ? x / sr.width : 0.5
    info.style.left = `${x}px`
    info.style.transform = `translateX(${frac > 0.7 ? '-100%' : frac < 0.3 ? '0' : '-50%'})`
    renderReadout(readout, bar.dataset.readout ?? '')
    dateEl.textContent = bar.dataset.date ?? ''
    root.classList.add('tri-hovering')
  }

  const onMove = (event: MouseEvent) => {
    const bar = nearest(event.clientX)
    if (bar) activate(bar)
  }
  const onScrub = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    root.classList.remove('tri-hovering')
  }

  const closePanel = () => {
    root.classList.remove('tri-panel-open')
    panel.setAttribute('aria-hidden', 'true')
    scrim.setAttribute('aria-hidden', 'true')
  }

  const showPanel = (ids: string[]) => {
    panelBody.replaceChildren()
    if (details) {
      for (const id of ids) {
        const d = details[id]
        if (d) panelBody.appendChild(renderDetail(d))
      }
    }
    root.classList.add('tri-panel-open')
    panel.setAttribute('aria-hidden', 'false')
    scrim.setAttribute('aria-hidden', 'false')
  }

  const openPanel = (bar: HTMLElement) => {
    const idsAttr = bar.dataset.ids
    if (!idsAttr) return
    const ids = idsAttr.split(',')
    if (details) {
      showPanel(ids)
      return
    }
    const path = root.dataset.detailPath
    if (!path) return
    fetch(path)
      .then(res => res.json())
      .then((data: Record<string, StravaActivityDetail>) => {
        details = data
        showPanel(ids)
      })
      .catch(() => {})
  }

  const onClick = (event: MouseEvent) => {
    const bar = nearest(event.clientX)
    if (bar) openPanel(bar)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') closePanel()
  }

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onScrub)
  barsEl.addEventListener('click', onClick)
  scrim.addEventListener('click', closePanel)
  closeBtn?.addEventListener('click', closePanel)
  document.addEventListener('keydown', onKey)

  return () => {
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onScrub)
    barsEl.removeEventListener('click', onClick)
    scrim.removeEventListener('click', closePanel)
    closeBtn?.removeEventListener('click', closePanel)
    document.removeEventListener('keydown', onKey)
  }
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (!root) return
  const cleanup = setup(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
