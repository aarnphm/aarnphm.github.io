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
    figs.append(buildRoute(d.route), buildElevation(d))
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
  let hideTimer = 0

  let audio: AudioContext | null = null
  let lastDrop = 0
  const armAudio = () => {
    if (!audio) audio = new AudioContext()
    if (audio.state === 'suspended') void audio.resume()
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
      raindrop(idx)
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
  const onToggle = (event: MouseEvent) => {
    const btn = (event.target as HTMLElement | null)?.closest('.tri-act-toggle')
    btn?.closest('.tri-act')?.classList.toggle('tri-act--expanded')
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
  pop.addEventListener('click', onToggle)

  return () => {
    window.clearTimeout(hideTimer)
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
    pop.removeEventListener('click', onToggle)
    window.removeEventListener('pointerdown', armAudio)
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

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (!root) return
  const cleanup = setup(root)
  if (cleanup) window.addCleanup?.(cleanup)
  const calcCleanup = setupCalc(root)
  if (calcCleanup) window.addCleanup?.(calcCleanup)
  const gearCleanup = setupGear(root)
  if (gearCleanup) window.addCleanup?.(gearCleanup)
})
