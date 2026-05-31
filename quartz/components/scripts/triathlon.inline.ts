import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import type { DailyPoint, StravaAnalytics } from '../../plugins/stores/strava-analytics'
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

const BATTERY = [
  'M23 10V14',
  'M1 16V8C1 6.89543 1.89543 6 3 6H18C19.1046 6 20 6.89543 20 8V16C20 17.1046 19.1046 18 18 18H3C1.89543 18 1 17.1046 1 16Z',
  'M10.1667 9L8.5 12H12.5L10.8333 15',
]

const buildBattery = (): SVGElement => {
  const s = svg('svg', { class: 'tri-ico tri-battery', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of BATTERY) s.appendChild(svg('path', { d }))
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
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
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
  s.appendChild(svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
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

const buildPower = (d: StravaActivityDetail): HTMLElement => {
  const w = 100
  const h = 30
  const pad = 2
  const maxD = d.route[d.route.length - 1].d || 1
  let maxW = 1
  for (const p of d.route) if (p.w > maxW) maxW = p.w
  const px = (km: number): number => (km / maxD) * w
  const py = (watt: number): number => h - pad - (watt / maxW) * (h - 2 * pad)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    area += `L ${px(p.d).toFixed(2)} ${py(p.w).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(p.w).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const s = svg('svg', { class: 'tri-elev', viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none' })
  s.appendChild(svg('path', { d: area, class: 'tri-elev-area' }))
  s.appendChild(svg('path', { d: line, class: 'tri-elev-line' }))
  const wrap = el('div', 'tri-elev-wrap')
  const cap = el('div', 'tri-elev-cap')
  cap.append(el('span', 'tri-elev-range', `${maxW} W peak`))
  wrap.append(s, cap)
  return wrap
}

const statRow = (label: string, value: string): HTMLElement => {
  const tr = document.createElement('tr')
  tr.append(el('th', 'tri-act-stat-k', label), el('td', 'tri-act-stat-v', value))
  return tr
}

const linkElev = (
  figs: HTMLElement,
  routeSvg: SVGElement,
  elevWrap: HTMLElement,
  route: StravaActivityDetail['route'],
): void => {
  const elevSvg = elevWrap.querySelector<SVGElement>('.tri-elev')
  const marker = routeSvg.querySelector('.tri-route-cursor')
  const cursor = elevSvg?.querySelector('.tri-elev-cursor')
  if (!elevSvg || !marker || !cursor) return
  const maxD = route[route.length - 1].d || 1
  const pad = 6
  const span = 88
  const readout = el('div', 'tri-fig-readout')
  figs.appendChild(readout)
  const onMove = (event: MouseEvent) => {
    const r = elevSvg.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (event.clientX - r.left) / r.width))
    const targetD = frac * maxD
    let i = 0
    let best = Infinity
    for (let k = 0; k < route.length; k++) {
      const dd = Math.abs(route[k].d - targetD)
      if (dd < best) {
        best = dd
        i = k
      }
    }
    const p = route[i]
    const x = ((p.d / maxD) * 100).toFixed(2)
    cursor.setAttribute('x1', x)
    cursor.setAttribute('x2', x)
    marker.setAttribute('cx', (pad + p.x * span).toFixed(2))
    marker.setAttribute('cy', (pad + (1 - p.y) * span).toFixed(2))
    const j0 = Math.max(0, i - 2)
    const j1 = Math.min(route.length - 1, i + 2)
    const dKm = route[j1].d - route[j0].d
    const grade = dKm > 0 ? ((route[j1].alt - route[j0].alt) / (dKm * 1000)) * 100 : 0
    const g = Math.round(grade * 10) / 10
    readout.textContent = `${(p.d * KM_TO_MI).toFixed(2)} mi · ${Math.round(p.alt)} m · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%`
    figs.classList.add('tri-figs--hover')
  }
  const onLeave = () => figs.classList.remove('tri-figs--hover')
  elevSvg.addEventListener('mousemove', onMove)
  elevSvg.addEventListener('mouseleave', onLeave)
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
  if (d.avgHr) body.appendChild(statRow('avg hr', `${d.avgHr} bpm`))
  stats.appendChild(body)
  wrap.appendChild(stats)

  if (d.sport === 'swim') {
    const figs = el('div', 'tri-act-figs')
    figs.appendChild(buildPool(d))
    wrap.appendChild(figs)
  } else if (d.route.length >= 2) {
    const figs = el('div', 'tri-act-figs')
    const routeSvg = buildRoute(d.route)
    const elev = buildElevation(d)
    figs.append(routeSvg, elev)
    linkElev(figs, routeSvg, elev, d.route)
    wrap.appendChild(figs)
  }

  const moreRows: [string, string][] = []
  if (d.deviceWatts && d.npWatts != null) moreRows.push(['NP', `${d.npWatts} W`])
  if (d.avgWatts != null)
    moreRows.push([d.deviceWatts ? 'avg power' : 'est power', `${d.avgWatts} W`])
  if (d.deviceWatts && d.maxWatts != null) moreRows.push(['max power', `${d.maxWatts} W`])
  if (d.kilojoules != null) moreRows.push(['energy', `${d.kilojoules} kJ`])
  if (d.avgCadence != null)
    moreRows.push(['cadence', `${d.avgCadence} ${d.sport === 'run' ? 'spm' : 'rpm'}`])
  if (d.maxHr != null) moreRows.push(['max hr', `${d.maxHr} bpm`])
  if (d.sufferScore != null) moreRows.push(['effort', `${d.sufferScore}`])
  if (d.avgTemp != null) moreRows.push(['temp', `${d.avgTemp}°C`])
  const hasPowerStream = d.deviceWatts && d.route.some(p => p.w > 0)

  if (moreRows.length > 0 || hasPowerStream) {
    const toggle = el('button', 'tri-act-toggle')
    toggle.setAttribute('type', 'button')
    const more = el('div', 'tri-act-more')
    if (moreRows.length > 0) {
      const mt = el('table', 'tri-act-stats')
      const mb = document.createElement('tbody')
      for (const [k, v] of moreRows) mb.appendChild(statRow(k, v))
      mt.appendChild(mb)
      more.appendChild(mt)
    }
    if (hasPowerStream) more.appendChild(buildPower(d))
    wrap.append(toggle, more)
  }
  return wrap
}

const setup = (root: HTMLElement): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const location = root.dataset.location ?? 'Toronto'
  let active: HTMLElement | null = null
  let activeIdx = -1
  let details: Record<string, StravaActivityDetail> | null = null
  let pinned = false
  let locked = false
  let hideTimer = 0

  const scroller = el('div', 'tri-pop-scroll')
  pop.appendChild(scroller)
  const updateOverflow = () => {
    pop.classList.toggle(
      'tri-pop--more',
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 4,
    )
  }
  const setLocked = (on: boolean) => {
    locked = on
    barsEl.classList.toggle('tri-bars--locked', on)
  }

  let audio: AudioContext | null = null
  let lastDrop = 0
  try {
    audio = new AudioContext()
    if (navigator.userActivation?.hasBeenActive) void audio.resume()
  } catch {
    audio = null
  }
  const armAudio = () => {
    if (audio && audio.state === 'suspended') void audio.resume()
  }
  const raindrop = (idx: number) => {
    if (!audio || audio.state !== 'running') return
    const t = audio.currentTime
    if (t - lastDrop < 0.05) return
    lastDrop = t
    const base = 560 + (idx % 8) * 28
    const osc = audio.createOscillator()
    const gain = audio.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(base * 1.8, t)
    osc.frequency.exponentialRampToValueAtTime(base, t + 0.08)
    gain.gain.setValueAtTime(0.0001, t)
    gain.gain.exponentialRampToValueAtTime(0.05, t + 0.004)
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.13)
    osc.connect(gain)
    gain.connect(audio.destination)
    osc.start(t)
    osc.stop(t + 0.15)
  }
  window.addEventListener('pointerdown', armAudio)
  window.addEventListener('keydown', armAudio)

  const buildCard = (bar: HTMLElement): HTMLElement => {
    const card = el('div', 'tri-pop-card')
    const idsAttr = bar.dataset.ids
    const head = el('div', 'tri-pop-head')
    head.appendChild(el('span', 'tri-pop-date', bar.dataset.date ?? ''))
    if (idsAttr) {
      const first = details?.[idsAttr.split(',')[0]]
      head.appendChild(el('span', 'tri-pop-loc', first?.location ?? location))
    }
    card.appendChild(head)
    if (!idsAttr) {
      const rest = el('div', 'tri-pop-rest')
      rest.append(buildBattery(), el('span', 'tri-pop-rest-label', 'rest'))
      card.appendChild(rest)
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
      raindrop(idx)
      const card = buildCard(bar)
      scroller.replaceChildren(card)
      scroller.scrollTop = 0
      updateOverflow()
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

  const setExpanded = (on: boolean) => {
    for (const a of pop.querySelectorAll('.tri-act')) a.classList.toggle('tri-act--expanded', on)
    updateOverflow()
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    activeIdx = -1
    pinned = false
    setLocked(false)
    root.classList.remove('tri-hovering')
  }

  const onMove = (event: MouseEvent) => {
    if (pinned || locked) return
    window.clearTimeout(hideTimer)
    const idx = nearest(event.clientX)
    if (idx >= 0) showFor(idx, event.clientX, event.clientY)
  }
  const onBarsLeave = () => {
    if (!pinned && !locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onPopEnter = () => {
    window.clearTimeout(hideTimer)
    pinned = true
  }
  const onPopLeave = () => {
    pinned = false
    if (!locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onBarsClick = (event: MouseEvent) => {
    const idx = nearest(event.clientX)
    if (idx < 0) return
    if (locked && bars[idx] === active) {
      setLocked(false)
      setExpanded(false)
    } else {
      showFor(idx, event.clientX, event.clientY)
      setLocked(true)
      setExpanded(true)
    }
  }
  const onToggle = (event: MouseEvent) => {
    const btn = (event.target as HTMLElement | null)?.closest('.tri-act-toggle')
    btn?.closest('.tri-act')?.classList.toggle('tri-act--expanded')
  }
  const dismiss = () => {
    if (!locked) return
    setLocked(false)
    setExpanded(false)
    hide()
  }
  const onDocClick = (event: MouseEvent) => {
    const t = event.target as Node
    if (locked && !barsEl.contains(t) && !pop.contains(t)) dismiss()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') dismiss()
  }

  const path = root.dataset.detailPath
  if (path)
    fetch(path)
      .then(res => res.json())
      .then((data: Record<string, StravaActivityDetail>) => {
        details = data
        if (active) {
          scroller.replaceChildren(buildCard(active))
          if (locked) setExpanded(true)
          updateOverflow()
        }
      })
      .catch(() => {})

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onBarsLeave)
  barsEl.addEventListener('click', onBarsClick)
  pop.addEventListener('mouseenter', onPopEnter)
  pop.addEventListener('mouseleave', onPopLeave)
  pop.addEventListener('click', onToggle)
  scroller.addEventListener('scroll', updateOverflow, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    window.clearTimeout(hideTimer)
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    barsEl.removeEventListener('click', onBarsClick)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
    pop.removeEventListener('click', onToggle)
    scroller.removeEventListener('scroll', updateOverflow)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('pointerdown', armAudio)
    window.removeEventListener('keydown', armAudio)
    void audio?.close()
  }
}

const setupCalc = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-calc-btn')
  const calc = root.querySelector<HTMLElement>('.tri-calc')
  const scrim = root.querySelector<HTMLElement>('.tri-calc-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-calc-close')
  if (!btn || !calc || !scrim) return null

  const parseClock = (s: string): number => {
    const parts = s.split(':').map(Number)
    return parts.length === 2 ? (parts[0] || 0) * 60 + (parts[1] || 0) : Number(s) || 0
  }
  const fmt = (sec: number): string => {
    const t = Math.round(sec)
    const h = Math.floor(t / 3600)
    const m = Math.floor((t % 3600) / 60)
    const s = t % 60
    return h > 0
      ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
      : `${m}:${String(s).padStart(2, '0')}`
  }
  const inputVal = (k: string): string =>
    calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)?.value ?? ''
  const setResult = (leg: string, sec: number): void => {
    const e = calc.querySelector<HTMLElement>(`.tri-calc-r[data-leg="${leg}"]`)
    if (e) e.textContent = fmt(sec)
  }

  const compute = () => {
    const swimKm = Number(calc.dataset.swim) || 0
    const bikeKm = Number(calc.dataset.bike) || 0
    const runKm = Number(calc.dataset.run) || 0
    const swimSec = ((swimKm * 1000) / 100) * parseClock(inputVal('swim'))
    const t1 = parseClock(inputVal('t1'))
    const mph = Number(inputVal('bike')) || 0
    const bikeSec = mph > 0 ? ((bikeKm * 0.621371) / mph) * 3600 : 0
    const t2 = parseClock(inputVal('t2'))
    const runSec = runKm * 0.621371 * parseClock(inputVal('run'))
    setResult('swim', swimSec)
    setResult('t1', t1)
    setResult('bike', bikeSec)
    setResult('t2', t2)
    setResult('run', runSec)
    setResult('total', swimSec + t1 + bikeSec + t2 + runSec)
  }

  const open = () => {
    root.classList.add('tri-calc-open')
    calc.setAttribute('aria-hidden', 'false')
    scrim.setAttribute('aria-hidden', 'false')
    compute()
  }
  const close = () => {
    root.classList.remove('tri-calc-open')
    calc.setAttribute('aria-hidden', 'true')
    scrim.setAttribute('aria-hidden', 'true')
  }
  const onCalcClick = (event: MouseEvent) => {
    const p = (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-calc-preset')
    if (!p) return
    calc.dataset.swim = p.dataset.swim ?? ''
    calc.dataset.bike = p.dataset.bike ?? ''
    calc.dataset.run = p.dataset.run ?? ''
    for (const x of calc.querySelectorAll('.tri-calc-preset'))
      x.classList.toggle('tri-calc-preset--on', x === p)
    compute()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', open)
  closeBtn?.addEventListener('click', close)
  scrim.addEventListener('click', close)
  calc.addEventListener('click', onCalcClick)
  calc.addEventListener('input', compute)
  document.addEventListener('keydown', onKey)
  calc.querySelectorAll('.tri-calc-preset')[1]?.classList.add('tri-calc-preset--on')

  return () => {
    btn.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    scrim.removeEventListener('click', close)
    calc.removeEventListener('click', onCalcClick)
    calc.removeEventListener('input', compute)
    document.removeEventListener('keydown', onKey)
  }
}

const setupGear = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-gear-btn')
  const wrap = root.querySelector<HTMLElement>('.tri-gear-wrap')
  const panel = root.querySelector<HTMLElement>('.tri-gear')
  if (!btn || !wrap || !panel) return null

  const close = () => {
    wrap.classList.remove('tri-gear-open')
    panel.setAttribute('aria-hidden', 'true')
  }
  const onBtn = () => {
    const open = wrap.classList.toggle('tri-gear-open')
    panel.setAttribute('aria-hidden', open ? 'false' : 'true')
  }
  const onDocClick = (event: MouseEvent) => {
    if (!wrap.contains(event.target as Node)) close()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', onBtn)
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', onBtn)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
  }
}

const ANA_W = 100
const ANA_H = 30
const RACE_LABEL: Record<string, string> = {
  sprint: 'sprint',
  olympic: 'olympic',
  '70.3': '70.3',
  ironman: 'ironman',
}

const clampN = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))
const polyD = (pts: [number, number][]): string =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')
const signed = (n: number): string => (n > 0 ? `+${n}` : `${n}`)
const hms = (sec: number): string => {
  const t = Math.max(0, Math.round(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`
}
const anaTitle = (text: string): HTMLElement => el('div', 'tri-ana-block-title', text)
const bySport = <T extends { sport: Sport }>(arr: T[], sport: Sport): T | undefined =>
  arr.find(x => x.sport === sport)
const thLabel = (th: { paceLabel: string; unit: string }): string =>
  th.unit === 'km/h' ? `${th.paceLabel} km/h` : `${th.paceLabel}${th.unit.slice(1)}`
const buildIconLeg = (sport: Sport): HTMLElement => {
  const wrap = el('span', `tri-ana-ico tri-leg-${sport}`)
  wrap.appendChild(buildIcon(sport))
  return wrap
}
const trendDir = (invert: boolean, slope: number | null): number => {
  if (slope == null || slope === 0) return 0
  return (invert ? slope < 0 : slope > 0) ? 1 : -1
}

const buildHeadline = (data: StravaAnalytics): HTMLElement => {
  const wrap = el('div', 'tri-ana-head')
  wrap.appendChild(el('div', 'tri-ana-head-line', data.headline || 'training analytics'))
  const r = data.risk
  const stats = el('div', 'tri-ana-head-stats')
  const stat = (k: string, v: string, sub: string, cls: string): HTMLElement => {
    const c = el('div', 'tri-ana-stat')
    c.append(
      el('span', `tri-ana-stat-v ${cls}`, v),
      el('span', 'tri-ana-stat-k', k),
      el('span', 'tri-ana-stat-sub', sub),
    )
    return c
  }
  stats.append(
    stat('fitness', `${Math.round(r.ctl)}`, 'CTL', ''),
    stat('form', signed(Math.round(r.tsb)), r.tsbZone, `tri-zone-${r.tsbZone}`),
    stat(
      'load',
      r.acwrState === 'building' ? 'base' : (r.acwr?.toFixed(2) ?? '—'),
      r.acwrState,
      `tri-acwr-${r.acwrState}`,
    ),
  )
  wrap.appendChild(stats)
  return wrap
}

const buildGauge = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-gauge')
  block.appendChild(anaTitle('form · ramp'))
  const r = data.risk
  const W = 100
  const H = 12
  const lo = -40
  const hi = 25
  const xf = (v: number): number => ((clampN(v, lo, hi) - lo) / (hi - lo)) * W
  const s = svg('svg', {
    class: 'tri-gauge-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('rect', { x: 0, y: 4, width: xf(-10), height: 4, class: 'tri-gauge-zone tri-gauge-z-low' }),
  )
  s.appendChild(
    svg('rect', {
      x: xf(-10),
      y: 4,
      width: xf(5) - xf(-10),
      height: 4,
      class: 'tri-gauge-zone tri-gauge-z-mid',
    }),
  )
  s.appendChild(
    svg('rect', {
      x: xf(5),
      y: 4,
      width: W - xf(5),
      height: 4,
      class: 'tri-gauge-zone tri-gauge-z-high',
    }),
  )
  const k42 = data.meta.method.k42
  const k7 = data.meta.method.k7
  let c = r.ctl
  let a = r.atl
  for (let i = 0; i < 14; i++) {
    c = c * (1 - k42)
    a = a * (1 - k7)
  }
  const projTsb = c - a
  const track = el('div', 'tri-gauge-track')
  const proj = el('span', 'tri-gauge-proj')
  proj.style.left = `${clampN(xf(projTsb), 1.5, 98.5)}%`
  const needle = el('span', 'tri-gauge-needle')
  needle.style.left = `${clampN(xf(r.tsb), 1.5, 98.5)}%`
  track.append(s, proj, needle)
  block.appendChild(track)
  const chips = el('div', 'tri-gauge-chips')
  chips.append(
    el(
      'span',
      `tri-ana-chip tri-acwr-${r.acwrState}`,
      r.acwrState === 'building'
        ? 'ACWR building base'
        : `ACWR ${r.acwr?.toFixed(2)} ${r.acwrState}`,
    ),
    el('span', 'tri-ana-chip', `ramp ${signed(Math.round((r.rampWeek || 0) * 100))}%`),
    el(
      'span',
      'tri-ana-chip',
      r.monotony != null ? `monotony ${r.monotony.toFixed(2)}` : 'monotony —',
    ),
    el('span', 'tri-ana-chip', r.strain != null ? `strain ${Math.round(r.strain)}` : 'strain —'),
  )
  block.appendChild(chips)
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(
    el(
      'span',
      'tri-ana-k',
      `form ${signed(Math.round(r.tsb))} → ${signed(Math.round(projTsb))} after 14d rest`,
    ),
  )
  block.appendChild(cap)
  return block
}

const buildPmc = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-pmc')
  block.appendChild(anaTitle('fitness · fatigue · form'))
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  let maxFit = 1
  for (const d of daily) {
    if (d.ctl > maxFit) maxFit = d.ctl
    if (d.atl > maxFit) maxFit = d.atl
  }
  let maxTsb = 8
  for (const d of daily) {
    const ab = Math.abs(d.tsb)
    if (ab > maxTsb) maxTsb = ab
  }
  const fitBot = 20
  const fitTop = 3
  const tsbBase = 27
  const tsbAmp = 5.5
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const yFit = (v: number): number => fitBot - (v / maxFit) * (fitBot - fitTop)
  const yTsb = (v: number): number => tsbBase - (v / maxTsb) * tsbAmp
  let warmEnd = 0
  while (warmEnd < n && daily[warmEnd].warmup) warmEnd++
  const ctlPts = daily.map((d, i) => [x(i), yFit(d.ctl)] as [number, number])
  const atlPts = daily.map((d, i) => [x(i), yFit(d.atl)] as [number, number])
  const tsbPts = daily.map((d, i) => [x(i), yTsb(d.tsb)] as [number, number])
  const s = svg('svg', {
    class: 'tri-ana-svg tri-pmc-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const areaInner = ctlPts.map(([px, py]) => `L ${px.toFixed(2)} ${py.toFixed(2)}`).join(' ')
  s.appendChild(
    svg('path', { d: `M 0 ${fitBot} ${areaInner} L ${ANA_W} ${fitBot} Z`, class: 'tri-elev-area' }),
  )
  s.appendChild(svg('line', { x1: 0, y1: tsbBase, x2: ANA_W, y2: tsbBase, class: 'tri-ana-zero' }))
  s.appendChild(svg('path', { d: polyD(tsbPts), class: 'tri-ana-tsb' }))
  if (warmEnd > 1)
    s.appendChild(
      svg('path', { d: polyD(ctlPts.slice(0, warmEnd)), class: 'tri-elev-line tri-pmc-warm' }),
    )
  s.appendChild(
    svg('path', {
      d: polyD(ctlPts.slice(Math.max(0, warmEnd - 1))),
      class: 'tri-elev-line tri-pmc-ctl',
    }),
  )
  s.appendChild(svg('path', { d: polyD(atlPts), class: 'tri-pmc-atl' }))
  s.appendChild(svg('line', { x1: ANA_W, y1: 0, x2: ANA_W, y2: ANA_H, class: 'tri-pmc-now' }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
  block.appendChild(s)
  const r = data.risk
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', `CTL ${Math.round(r.ctl)}`),
    el('span', 'tri-ana-k', `ATL ${Math.round(r.atl)}`),
    el('span', `tri-ana-k tri-zone-${r.tsbZone}`, `TSB ${signed(Math.round(r.tsb))}`),
  )
  block.appendChild(cap)
  return block
}

const buildCtlSport = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-ctlsport')
  block.appendChild(anaTitle('fitness by discipline'))
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  let mx = 1
  for (const d of daily) mx = Math.max(mx, d.swimCtl, d.bikeCtl, d.runCtl)
  const top = 3
  const bot = 28
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const y = (v: number): number => bot - (v / mx) * (bot - top)
  const s = svg('svg', {
    class: 'tri-ana-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const series: { sp: Sport; get: (d: DailyPoint) => number }[] = [
    { sp: 'swim', get: d => d.swimCtl },
    { sp: 'bike', get: d => d.bikeCtl },
    { sp: 'run', get: d => d.runCtl },
  ]
  for (const { sp, get } of series) {
    s.appendChild(
      svg('path', {
        d: polyD(daily.map((d, i) => [x(i), y(get(d))])),
        class: `tri-elev-line tri-line-${sp}`,
      }),
    )
  }
  block.appendChild(s)
  const cap = el('div', 'tri-elev-cap')
  for (const sp of ['swim', 'bike', 'run'] as Sport[]) {
    const days = bySport(data.thresholds, sp)?.staleDays ?? 0
    const leg = el('span', `tri-ana-leg tri-leg-${sp}`)
    leg.append(buildIcon(sp), el('span', 'tri-ana-k', days > 45 ? `${days}d stale` : `${days}d`))
    cap.appendChild(leg)
  }
  block.appendChild(cap)
  return block
}

const buildWeekly = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-weekly')
  block.appendChild(anaTitle('weekly load'))
  const wk = data.weekly
  if (!wk.length) {
    block.appendChild(el('div', 'tri-ana-empty', 'no weeks'))
    return block
  }
  let mx = 1
  for (const w of wk) if (w.load > mx) mx = w.load
  const n = wk.length
  const H = 32
  const bot = H - 0.5
  const s = svg('svg', {
    class: 'tri-ana-svg tri-ana-weekly-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  wk.forEach((w, i) => {
    if (w.load <= 0) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (w.load / mx) * (H - 2)
    const spike = w.ramp != null && w.ramp > 0.1
    s.appendChild(
      svg('rect', {
        x: i + 0.12,
        y: bot - h,
        width: 0.76,
        height: h,
        class: spike ? 'tri-seg--load tri-seg--spike' : 'tri-seg--load',
      }),
    )
  })
  block.appendChild(s)
  const active = wk.filter(w => w.load > 0).length
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', `${active} active wk`),
    el('span', 'tri-ana-k', `peak ${Math.round(mx)}/wk`),
  )
  block.appendChild(cap)
  return block
}

const buildReadiness = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-readiness')
  block.appendChild(anaTitle('race readiness'))
  if (!data.races.length) {
    block.appendChild(el('div', 'tri-ana-empty', '—'))
    return block
  }
  for (const r of data.races) {
    const row = el('div', 'tri-rdy-row')
    row.appendChild(el('span', 'tri-rdy-label', RACE_LABEL[r.distance] ?? r.distance))
    const track = el('div', 'tri-rdy-bar')
    const score = clampN(r.score, 0, 100)
    const fill = el('div', 'tri-rdy-fill')
    fill.style.width = `${Math.max(2, score)}%`
    const bw = Math.min(40, r.bandPct)
    const band = el('div', 'tri-rdy-band')
    band.style.left = `${clampN(score - bw / 2, 0, 100 - bw)}%`
    band.style.width = `${bw}%`
    track.append(fill, band)
    row.appendChild(track)
    const meta = el('span', 'tri-rdy-meta')
    meta.append(
      el('span', `tri-rdy-bind tri-leg-${r.bindingLeg}`, r.bindingLeg),
      el('span', 'tri-rdy-time', hms(r.predictedTotalS)),
    )
    row.appendChild(meta)
    block.appendChild(row)
  }
  return block
}

const buildTrendPanel = (data: StravaAnalytics, sport: Sport): HTMLElement => {
  const tr = bySport(data.trends, sport)
  const th = bySport(data.thresholds, sport)
  const wrap = el('div', `tri-trend-panel${tr?.stale ? ' tri-trend-stale' : ''}`)
  const head = el('div', 'tri-trend-head')
  head.append(buildIconLeg(sport), el('span', 'tri-trend-unit', th ? thLabel(th) : sport))
  if (th) head.appendChild(el('span', `tri-ana-conf tri-conf-${th.conf}`, th.conf))
  wrap.appendChild(head)
  if (!tr || tr.method === 'none') {
    const msg =
      tr?.note ||
      (th && th.staleDays > 45 ? `stale · last ${th.staleDays}d ago` : 'not enough data')
    wrap.appendChild(el('div', 'tri-trend-note', msg))
    return wrap
  }
  const fc = tr.forecast
  if (fc.length >= 1 && tr.level != null) {
    const ys = [tr.level, ...fc.flatMap(f => [f.value, f.lo, f.hi])]
    const ymin = Math.min(...ys)
    const ymax = Math.max(...ys)
    const yspan = Math.max(1e-6, ymax - ymin)
    const top = 4
    const bot = 24
    const m = fc.length
    const xOf = (i: number): number => (i < 0 ? 0 : 8 + (i / Math.max(1, m - 1)) * (ANA_W - 8))
    const Y = (v: number): number =>
      tr.invert
        ? top + ((v - ymin) / yspan) * (bot - top)
        : bot - ((v - ymin) / yspan) * (bot - top)
    const s = svg('svg', {
      class: 'tri-ana-svg tri-trend-svg',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    if (m >= 2) {
      const hiPts = fc.map((f, i) => `${xOf(i).toFixed(2)} ${Y(f.hi).toFixed(2)}`)
      const loPts = fc.map(
        (_, i) => `${xOf(m - 1 - i).toFixed(2)} ${Y(fc[m - 1 - i].lo).toFixed(2)}`,
      )
      s.appendChild(
        svg('path', {
          d: `M ${hiPts.join(' L ')} L ${loPts.join(' L ')} Z`,
          class: `tri-trend-band tri-fill-${sport}`,
        }),
      )
    }
    const linePts: [number, number][] = [
      [xOf(-1), Y(tr.level)],
      ...fc.map((f, i) => [xOf(i), Y(f.value)] as [number, number]),
    ]
    s.appendChild(svg('path', { d: polyD(linePts), class: `tri-trend-proj tri-line-${sport}` }))
    const track = el('div', 'tri-trend-track')
    const dot = el('span', `tri-trend-dot tri-bg-${sport}`)
    dot.style.left = `${clampN((xOf(-1) / ANA_W) * 100, 2, 98)}%`
    dot.style.top = `${(Y(tr.level) / ANA_H) * 100}%`
    track.append(s, dot)
    wrap.appendChild(track)
  }
  const dir = trendDir(tr.invert, tr.slopePerWeek)
  const note = el('div', 'tri-trend-note')
  note.append(
    el(
      'span',
      `tri-trend-dir tri-dir-${dir > 0 ? 'up' : dir < 0 ? 'down' : 'flat'}`,
      dir > 0 ? 'faster' : dir < 0 ? 'slower' : 'flat',
    ),
    el('span', 'tri-ana-k', `${tr.method} · n=${tr.sampleSize}`),
  )
  wrap.appendChild(note)
  return wrap
}

const buildTrend = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-trend')
  block.appendChild(anaTitle('pace trend + forecast'))
  for (const sport of ['swim', 'bike', 'run'] as Sport[])
    block.appendChild(buildTrendPanel(data, sport))
  return block
}

const buildActions = (data: StravaAnalytics): HTMLElement => {
  const block = el('div', 'tri-ana-actions')
  block.appendChild(anaTitle('things to improve'))
  const banner = el('div', 'tri-actions-head')
  banner.append(
    el('span', 'tri-actions-weak', 'weakest'),
    buildIconLeg(data.weakestSport),
    el('span', 'tri-ana-k', data.weakestSport),
  )
  block.appendChild(banner)
  if (data.actions.length) {
    const tbl = el('table', 'tri-act-stats')
    const body = document.createElement('tbody')
    data.actions.forEach((a, i) => {
      const tr = document.createElement('tr')
      tr.append(
        el('th', 'tri-act-stat-k', `${i + 1}. ${a.text}`),
        el('td', 'tri-act-stat-v', a.value),
      )
      body.appendChild(tr)
    })
    tbl.appendChild(body)
    block.appendChild(tbl)
  }
  const chips = el('div', 'tri-gauge-chips')
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const th = bySport(data.thresholds, sport)
    if (th)
      chips.appendChild(
        el('span', `tri-ana-chip tri-chip-${sport}`, `${sport} ${thLabel(th)} ${th.conf}`),
      )
  }
  block.appendChild(chips)
  return block
}

const ANALYTICS_BUILDERS: Record<string, (data: StravaAnalytics) => HTMLElement> = {
  gauge: buildGauge,
  pmc: buildPmc,
  'ctl-sport': buildCtlSport,
  weekly: buildWeekly,
  readiness: buildReadiness,
  trend: buildTrend,
  actions: buildActions,
}

const wireScrub = (panel: HTMLElement, pop: HTMLElement, daily: DailyPoint[]): (() => void) => {
  const svgEl = panel.querySelector<SVGElement>('.tri-pmc-svg')
  const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
  if (!svgEl || !cursor || daily.length < 2) return () => {}
  const onMove = (event: MouseEvent) => {
    const r = svgEl.getBoundingClientRect()
    const frac = clampN((event.clientX - r.left) / r.width, 0, 1)
    const d = daily[Math.round(frac * (daily.length - 1))]
    cursor.setAttribute('x1', `${(frac * ANA_W).toFixed(2)}`)
    cursor.setAttribute('x2', `${(frac * ANA_W).toFixed(2)}`)
    pop.replaceChildren(
      el('span', 'tri-pop-date', d.date),
      el('span', 'tri-ana-k', `CTL ${Math.round(d.ctl)}`),
      el('span', 'tri-ana-k', `ATL ${Math.round(d.atl)}`),
      el('span', 'tri-ana-k', `TSB ${signed(Math.round(d.tsb))}`),
    )
    pop.style.left = `${Math.min(event.clientX + 14, window.innerWidth - 150)}px`
    pop.style.top = `${r.top - 4}px`
    panel.classList.add('tri-ana-scrubbing')
  }
  const onLeave = () => panel.classList.remove('tri-ana-scrubbing')
  svgEl.addEventListener('mousemove', onMove)
  svgEl.addEventListener('mouseleave', onLeave)
  return () => {
    svgEl.removeEventListener('mousemove', onMove)
    svgEl.removeEventListener('mouseleave', onLeave)
  }
}

const setupAnalytics = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-analytics-btn')
  const panel = root.querySelector<HTMLElement>('.tri-analytics')
  const scrim = root.querySelector<HTMLElement>('.tri-analytics-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-ana-close')
  const headline = root.querySelector<HTMLElement>('.tri-ana-headline')
  const pop = root.querySelector<HTMLElement>('.tri-ana-pop')
  if (!btn || !panel || !scrim) return null

  let loaded = false
  let scrubCleanup: (() => void) | null = null

  const render = (data: StravaAnalytics) => {
    if (headline) headline.replaceChildren(buildHeadline(data))
    for (const block of Array.from(panel.querySelectorAll<HTMLElement>('.tri-ana-block'))) {
      const build = ANALYTICS_BUILDERS[block.dataset.chart ?? '']
      if (build) block.replaceChildren(build(data))
    }
    if (pop) scrubCleanup = wireScrub(panel, pop, data.daily)
  }
  const load = () => {
    if (loaded) return
    loaded = true
    const path = root.dataset.analyticsPath
    if (!path) return
    fetch(path)
      .then(res => res.json())
      .then((data: StravaAnalytics) => render(data))
      .catch(() => {})
  }
  const open = () => {
    root.classList.add('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'false')
    scrim.setAttribute('aria-hidden', 'false')
    load()
  }
  const close = () => {
    root.classList.remove('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'true')
    scrim.setAttribute('aria-hidden', 'true')
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', open)
  closeBtn?.addEventListener('click', close)
  scrim.addEventListener('click', close)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    scrim.removeEventListener('click', close)
    document.removeEventListener('keydown', onKey)
    scrubCleanup?.()
  }
}

const setupCheat = (root: HTMLElement): (() => void) | null => {
  const unit = root.querySelector<HTMLButtonElement>('.tri-cheat-unit')
  const cells = root.querySelectorAll<HTMLElement>('.tri-cheat td[data-km]')
  if (!unit || cells.length === 0) return null

  const target = root.querySelector<HTMLElement>('.tri-cheat-target')
  let ann: RoughAnnotation | null = null
  let showTimer = 0
  if (target) {
    const color = getComputedStyle(root).getPropertyValue('--tri-accent').trim() || '#fc4c02'
    ann = annotate(target, {
      type: 'circle',
      color,
      strokeWidth: 1.6,
      padding: 5,
      animationDuration: 800,
      iterations: 2,
    })
    const a = ann
    showTimer = window.setTimeout(() => a.show(), 200)
  }

  let mi = false
  const onClick = () => {
    mi = !mi
    unit.textContent = mi ? 'mi' : 'km'
    for (const c of cells) {
      if (!mi) {
        c.textContent = c.dataset.km ?? ''
      } else {
        const v = Number(c.dataset.km) * KM_TO_MI
        c.textContent = v < 10 ? v.toFixed(2) : v.toFixed(1)
      }
    }
  }
  unit.addEventListener('click', onClick)
  return () => {
    window.clearTimeout(showTimer)
    unit.removeEventListener('click', onClick)
    ann?.remove()
  }
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (!root) return
  const cleanup = setup(root)
  if (cleanup) window.addCleanup?.(cleanup)
  const calcCleanup = setupCalc(root)
  if (calcCleanup) window.addCleanup?.(calcCleanup)
  const gearCleanup = setupGear(root)
  if (gearCleanup) window.addCleanup?.(gearCleanup)
  const cheatCleanup = setupCheat(root)
  if (cheatCleanup) window.addCleanup?.(cheatCleanup)
  const anaCleanup = setupAnalytics(root)
  if (anaCleanup) window.addCleanup?.(anaCleanup)
})
