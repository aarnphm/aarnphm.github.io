import { writePetSpot } from '../../util/pet-spot'
import { readVisitorDaylightCell } from '../../util/visitor-daylight'
import { readStoredTrailWeather } from '../../util/visitor-trail-weather'
import { type Site, type Weather, fetchSite, guessSite } from './404-almanac'
import { makeClock } from './404-clock'
import { setupLandscape, type Landscape } from './404-landscape'

const SVG_NS = 'http://www.w3.org/2000/svg'
const GRAVITY = 900
const FLY_COLORS = ['nf-fly-rose', 'nf-fly-sage', 'nf-fly-gold']

type Letter = { el: SVGTextElement; x: number; y: number; gone: boolean }
type Falling = {
  el: SVGElement
  x: number
  y: number
  vx: number
  vy: number
  rot: number
  vr: number
  floor: number
  sunk: number
  letter: boolean
}
type Fly = {
  g: SVGGElement
  halo: SVGRectElement
  core: SVGRectElement
  x: number
  y: number
  a: number
  v: number
  phase: number
  freq: number
}

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
) {
  const el = document.createElementNS(SVG_NS, tag)
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v))
  return el
}

function missingPath(): string {
  let path = location.pathname
  try {
    path = decodeURIComponent(path)
  } catch {}
  return path.length > 1 ? path.replace(/\/$/, '') : path
}

function fillTrigrams(s: string, out = new Set<string>()): Set<string> {
  out.clear()
  const padded = `  ${s} `
  for (let i = 0; i < padded.length - 2; i++) out.add(padded.slice(i, i + 3))
  return out
}

function dice(a: Set<string>, b: Set<string>): number {
  let hit = 0
  for (const g of a) if (b.has(g)) hit++
  return (2 * hit) / (a.size + b.size || 1)
}

const norm = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/\.html?$/, '')
    .replace(/[^a-z0-9/]+/g, '-')
    .replace(/^[-/]+|[-/]+$/g, '')

const lastSegment = (s: string) => s.slice(s.lastIndexOf('/') + 1)

// Departures board: nearest pages by trigram similarity, or random wandering when nothing is close.
// Home stays pinned as the last departure.
async function fillDepartures(list: HTMLElement, path: string, isActive: () => boolean) {
  const data = await fetchData.catch(() => undefined)
  if (!data || !isActive()) return
  const query = norm(path)
  const qAll = fillTrigrams(query)
  const qLast = fillTrigrams(lastSegment(query))
  const candTri = new Set<string>()
  const top: { slug: string; title: string; score: number }[] = []
  const validSlugs: string[] = []
  for (const slug in data) {
    if (slug.startsWith('tags/') || slug === '404') continue
    const d = data[slug]
    if (!d || d.protected) continue
    validSlugs.push(slug)
    const s = norm(slug)
    const s1 = dice(qAll, fillTrigrams(s, candTri))
    const s2 = dice(qLast, fillTrigrams(lastSegment(s), candTri)) * 0.95
    const s3 = dice(qLast, fillTrigrams(norm(d.title), candTri)) * 0.9
    const score = Math.max(s1, s2, s3)
    if (score <= 0.3) continue
    if (top.length < 3) {
      top.push({ slug, title: d.title, score })
      top.sort((a, b) => b.score - a.score)
    } else if (score > top[2].score) {
      top[2] = { slug, title: d.title, score }
      top.sort((a, b) => b.score - a.score)
    }
  }
  const wander = top.length === 0
  const picks = wander
    ? Array.from({ length: Math.min(3, validSlugs.length) }, () => {
        const slug = validSlugs[Math.floor(Math.random() * validSlugs.length)]
        return { slug, title: data[slug].title, score: 0 }
      })
    : top
  const rows = picks.map((p, i) => {
    const li = document.createElement('li')
    const a = document.createElement('a')
    a.className = 'nf-row'
    a.dataset.noPopover = ''
    a.href = '/' + p.slug.replace(/(^|\/)index$/, '')
    const label = document.createElement('span')
    label.textContent = p.title || lastSegment(p.slug)
    const lead = document.createElement('i')
    lead.className = 'nf-lead'
    const dock = document.createElement('span')
    dock.textContent = wander ? 'errance' : `quai ${i + 1}`
    a.append(label, lead, dock)
    li.append(a)
    return li
  })
  while (list.children.length > 1) list.firstElementChild?.remove()
  list.prepend(...rows)
}

// The WMO weather codes as a station announcer would read them.
const CONDITIONS: Record<number, string> = {
  0: 'dégagé',
  1: 'peu nuageux',
  2: 'nuageux',
  3: 'couvert',
  45: 'brouillard',
  48: 'brouillard givrant',
  51: 'bruine',
  53: 'bruine',
  55: 'bruine',
  56: 'bruine verglaçante',
  57: 'bruine verglaçante',
  61: 'pluie',
  63: 'pluie',
  65: 'forte pluie',
  66: 'pluie verglaçante',
  67: 'pluie verglaçante',
  71: 'neige',
  73: 'neige',
  75: 'forte neige',
  77: 'grains de neige',
  80: 'averses',
  81: 'averses',
  82: 'fortes averses',
  85: 'averses de neige',
  86: 'averses de neige',
  95: 'orage',
  96: 'orage de grêle',
  99: 'orage de grêle',
}
const conditions = (w: Weather) => CONDITIONS[w.code] ?? '—'

const pad2 = (n: number) => String(n).padStart(2, '0')
const hhmm = (d: Date, seconds = false) =>
  [d.getHours(), d.getMinutes(), ...(seconds ? [d.getSeconds()] : [])].map(pad2).join(':')

let cleanupPrev: (() => void) | null = null

document.addEventListener('nav', () => {
  cleanupPrev?.()
  cleanupPrev = null
  if (document.body.dataset.slug !== '404') return
  const scene = document.querySelector<HTMLElement>('.nf-scene')
  if (!scene) return
  const path = missingPath()
  if (window.plausible) window.plausible('404', { props: { path } })

  let active = true
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const timers = new Set<number>()
  const later = (fn: () => void, ms: number) => {
    const id = window.setTimeout(() => {
      timers.delete(id)
      if (active) fn()
    }, ms)
    timers.add(id)
  }

  const pathEl = scene.querySelector<HTMLElement>('[data-nf-path]')
  if (pathEl) pathEl.textContent = path

  const departures = scene.querySelector<HTMLElement>('[data-nf-departures]')
  const boardDetails = scene.querySelector<HTMLDetailsElement>('.nf-board details')
  let departuresLoaded = false
  const ensureDepartures = () => {
    if (departuresLoaded || !departures) return
    if (boardDetails && !boardDetails.open) return
    departuresLoaded = true
    void fillDepartures(departures, path, () => active)
  }
  if (boardDetails) {
    boardDetails.addEventListener('toggle', ensureDepartures)
    if (boardDetails.open) ensureDepartures()
  } else {
    ensureDepartures()
  }

  const arrival = scene.querySelector<HTMLElement>('[data-nf-arrival]')
  if (arrival) arrival.textContent = `arr ${hhmm(new Date())}`
  const dateEl = scene.querySelector<HTMLElement>('[data-nf-date]')
  if (dateEl) {
    // The weekday sits in its own span so it can give way when the weather needs the room.
    const now = new Date()
    const weekday = document.createElement('span')
    weekday.className = 'nf-weekday'
    weekday.textContent = `${new Intl.DateTimeFormat('fr', { weekday: 'short' }).format(now)} `
    dateEl.replaceChildren(
      weekday,
      new Intl.DateTimeFormat('fr', { day: 'numeric', month: 'short' }).format(now),
    )
  }

  // The sky follows the visitor: the landscape starts from a guess at where they are and takes the
  // worker's answer, with their weather, as soon as it comes.
  const font = document.fonts.load("11px 'Departure Mono'").catch(() => undefined)
  const locating = new AbortController()
  void fetchSite(locating.signal).then(async site => {
    if (!site || !active) return
    land?.place(site)
    const placeRow = scene.querySelector<HTMLElement>('[data-nf-place-row]')
    const placeEl = scene.querySelector<HTMLElement>('[data-nf-place]')
    const tempEl = scene.querySelector<HTMLElement>('[data-nf-temp]')
    if (site.place && placeRow && placeEl) {
      placeEl.textContent = site.place
      if (site.weather && tempEl) {
        tempEl.textContent = `${Math.round(site.weather.temp)}°C`
        tempEl.hidden = false
      }
      placeRow.hidden = false
    }
    const weatherEl = scene.querySelector<HTMLAnchorElement>('[data-nf-weather]')
    if (!weatherEl || !site.weather) return
    weatherEl.textContent = site.place
      ? conditions(site.weather)
      : `${Math.round(site.weather.temp)}° · ${conditions(site.weather)}`
    weatherEl.hidden = false
    await font
    const weekday = dateEl?.querySelector<HTMLElement>('.nf-weekday')
    if (dateEl && weekday && dateEl.scrollWidth > dateEl.clientWidth) weekday.hidden = true
  })
  const hands = makeClock(reduce)
  hands.step(0)

  const fx = scene.querySelector<SVGSVGElement>('.nf-fx')!
  const carving = fx.querySelector<SVGGElement>('.nf-carving')!
  const fallingLayer = fx.querySelector<SVGGElement>('.nf-falling')!
  const flyLayer = fx.querySelector<SVGGElement>('.nf-flies')!
  const door = scene.querySelector<SVGAElement>('.nf-door')!
  const layers = Array.from(scene.querySelectorAll<HTMLElement | SVGSVGElement>('.nf-layer'))
  const depths = layers.map(l => Number(l.dataset.depth ?? 0))
  const [tx, ty, trot, tw] = (scene.dataset.tablet ?? '0,0,0,200').split(',').map(Number)
  const water = Number(scene.dataset.water ?? 870)
  const cos = Math.cos((trot * Math.PI) / 180)
  const sin = Math.sin((trot * Math.PI) / 180)
  const toWorld = (x: number, y: number) => ({
    x: tx + x * cos - y * sin,
    y: ty + x * sin + y * cos,
  })

  // Carve the missing path into the fallen tablet, one glyph per element so each can crumble.
  const carved = path.length > 34 ? '…' + path.slice(-33) : path
  const chars = Array.from(carved)
  // Departure Mono advances 7 of every 11 units.
  const fontSize = Math.max(11, Math.min(28, (tw - 36) / (chars.length * 0.64)))
  const advance = fontSize * 0.64
  const letters: Letter[] = chars.map((ch, i) => {
    const x = (i - (chars.length - 1) / 2) * advance
    const y = fontSize * 0.35
    const el = svgEl('text', { x, y, 'text-anchor': 'middle', class: 'nf-glyph' })
    el.style.fontSize = `${fontSize}px`
    el.textContent = ch
    return { el, x, y, gone: ch.trim() === '' }
  })
  carving.replaceChildren(...letters.map(l => l.el))

  const falling: Falling[] = []

  function drop(letter: Letter, push = 0) {
    if (letter.gone) return
    letter.gone = true
    letter.el.classList.add('is-gone')
    const p = toWorld(letter.x, letter.y - fontSize * 0.35)
    const el = svgEl('text', { 'text-anchor': 'middle', class: 'nf-glyph nf-glyph-falling' })
    el.style.fontSize = `${fontSize}px`
    el.textContent = letter.el.textContent
    fallingLayer.append(el)
    falling.push({
      el,
      x: p.x,
      y: p.y,
      vx: push + (Math.random() - 0.5) * 60,
      vy: -60 - Math.random() * 120,
      rot: trot,
      vr: (Math.random() - 0.5) * 420,
      floor: water + 14 + Math.random() * 40,
      sunk: 0,
      letter: true,
    })
    if (letters.every(l => l.gone)) later(recarve, 3600)
  }

  function recarve() {
    letters.forEach((l, i) => {
      if (l.el.textContent!.trim() === '') return
      l.el.style.transitionDelay = `${i * 70}ms`
      l.el.classList.remove('is-gone')
      l.gone = false
    })
    later(() => letters.forEach(l => (l.el.style.transitionDelay = '')), letters.length * 70 + 800)
    later(crumble, 5200)
  }

  function crumble() {
    const left = letters.filter(l => !l.gone)
    if (left.length === 0) return
    drop(left[Math.floor(Math.random() * left.length)])
    later(crumble, 1300 + Math.random() * 2200)
  }

  // Droplets are one buffer pixel square and hop from pixel to pixel, like everything printed.
  function splash(x: number, y: number) {
    land?.splash(x, y, 34 + Math.random() * 20)
    const s = land?.pixel()?.px ?? 3
    for (let k = 0; k < 4; k++) {
      const el = svgEl('rect', { x: 0, y: 0, width: s, height: s, class: 'nf-droplet' })
      fallingLayer.append(el)
      falling.push({
        el,
        x,
        y: y - 2,
        vx: (Math.random() - 0.5) * 160,
        vy: -140 - Math.random() * 160,
        rot: 0,
        vr: 0,
        floor: y + 2,
        sunk: 0,
        letter: false,
      })
    }
  }

  // Fireflies drift on a heading random walk, get herded back into the scene, and follow the cursor.
  // Each is a lit pixel in a three-pixel halo, snapped to the canvases' grid.
  const flies: Fly[] = Array.from({ length: 20 }, (_, i) => {
    const g = svgEl('g', { class: 'nf-fly' })
    const halo = svgEl('rect', { class: `nf-fly-halo ${FLY_COLORS[i % 3]}` })
    const core = svgEl('rect', { class: `nf-fly-core ${FLY_COLORS[i % 3]}` })
    g.append(halo, core)
    flyLayer.append(g)
    return {
      g,
      halo,
      core,
      x: 240 + Math.random() * 1120,
      y: 380 + Math.random() * 560,
      a: Math.random() * Math.PI * 2,
      v: 18 + Math.random() * 22,
      phase: Math.random() * Math.PI * 2,
      freq: 0.6 + Math.random() * 1.4,
    }
  })

  const pointer = { x: -1e4, y: -1e4, px: 0, py: 0, nx: 0, ny: 0, vx: 0 }
  const parallax = { x: 0, y: 0 }

  const onMove = (e: PointerEvent) => {
    pointer.nx = e.clientX / window.innerWidth - 0.5
    pointer.ny = e.clientY / window.innerHeight - 0.5
    const m = fx.getScreenCTM()
    if (!m) return
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse())
    pointer.vx = p.x - pointer.x
    pointer.x = p.x
    pointer.y = p.y
    if (reduce) return void land?.still(pointer)
    for (const l of letters) {
      if (l.gone) continue
      const w = toWorld(l.x, l.y - fontSize * 0.35)
      if (Math.hypot(w.x - p.x, w.y - p.y) < Math.max(14, fontSize * 0.7)) {
        drop(l, Math.max(-240, Math.min(240, pointer.vx * 12)))
      }
    }
  }
  const onLeave = () => {
    pointer.x = pointer.y = -1e4
    if (reduce) land?.still(pointer)
  }
  window.addEventListener('pointermove', onMove, { passive: true })
  document.documentElement.addEventListener('pointerleave', onLeave)

  // Reveal the synchronously painted landscape on the next frame. Weather and the board's
  // font update independently, so their network requests do not hold the route handoff.
  requestAnimationFrame(() => active && scene.classList.add('is-ready'))

  const cell = readVisitorDaylightCell(document)
  const cached = readStoredTrailWeather(window)
  const baseSite = cell
    ? { lat: cell.lat, lon: cell.lon, place: null, weather: null }
    : cached?.lat !== null && cached?.lat !== undefined && cached.lon !== null
      ? { lat: cached.lat, lon: cached.lon, place: null, weather: null }
      : guessSite()
  const initialWeather: Weather | null = cached
    ? {
        code: cached.code,
        cloud: cached.cloud,
        wind: cached.windKmh,
        temp: cached.temperatureC ?? 12,
        snow: cached.snowCm,
      }
    : null
  const seededSite: Site = { ...baseSite, weather: initialWeather }

  const land: Landscape | null = setupLandscape(scene, reduce, water, hands, seededSite)
  // The gate's hands read the same time as the board. Without motion they only move once a second.
  const clockEl = scene.querySelector<HTMLElement>('[data-nf-clock]')
  const tick = () => {
    if (clockEl) clockEl.textContent = hhmm(new Date(), true)
    if (reduce) {
      hands.step(0)
      land?.refresh()
    }
    later(tick, 1000 - (Date.now() % 1000))
  }
  tick()

  const portalOn = () => {
    land?.portal('hot')
    hands.mode('hot')
  }
  const portalOff = () => {
    land?.portal('idle')
    hands.mode('idle')
  }
  door.addEventListener('pointerenter', portalOn)
  door.addEventListener('pointerleave', portalOff)
  door.addEventListener('focus', portalOn)
  door.addEventListener('blur', portalOff)

  // SVG anchors bypass the SPA router, so the door walks you in and then navigates itself.
  const onDoor = (e: MouseEvent) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
    e.preventDefault()
    const home = new URL('/', location.href)
    const go = () => (window.spaNavigate ? window.spaNavigate(home) : location.assign(home))
    const box = door.querySelector('circle')!.getBoundingClientRect()
    // The pet goes through with the reader, and comes out on the other side from the middle.
    writePetSpot({ x: box.left + box.width / 2, y: box.top + box.height / 2 })
    if (reduce) return void go()
    const stage = scene.querySelector<HTMLElement>('.nf-stage')!
    stage.style.transformOrigin = `${box.left + box.width / 2}px ${box.top + box.height / 2}px`
    land?.portal('enter')
    hands.mode('enter')
    scene.classList.add('is-entering')
    later(go, 820)
  }
  door.addEventListener('click', onDoor)

  // A click on one of the little people gets a "?" out of them. It is caught on the way down, so a
  // figure standing in the opening answers instead of letting the door take you home.
  const onHail = (e: MouseEvent) => {
    if (e.button !== 0 || (e.target as Element).closest('.nf-board')) return
    if (!land?.hail(e.clientX, e.clientY)) return
    e.preventDefault()
    e.stopPropagation()
  }
  scene.addEventListener('click', onHail, true)

  // A press on the cat pats it. A touch that starts on it strokes it, so it does not move the page.
  const onPat = (e: PointerEvent) => {
    if (e.button !== 0 || (e.target as Element).closest('.nf-board')) return
    land?.pat(e.clientX, e.clientY)
  }
  const onTouch = (e: TouchEvent) => {
    const t = e.touches[0]
    if (e.touches.length === 1 && land?.onCat(t.clientX, t.clientY)) e.preventDefault()
  }
  scene.addEventListener('pointerdown', onPat)
  scene.addEventListener('touchstart', onTouch, { passive: false })

  // The button over the cat is the keyboard's hand: focus lays it on the cat's cheek, a held arrow
  // strokes along its back, and Enter or Space pats. A pointer goes through the button to the cat.
  const petter = scene.querySelector<HTMLButtonElement>('.nf-pet')
  let pawDir = 0
  const onPaw = (e: KeyboardEvent) => {
    const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!dir) return
    e.preventDefault()
    if (e.type === 'keydown') {
      // Without motion each press moves the hand once.
      if (e.repeat) return
      pawDir = dir
    } else if (pawDir === dir) pawDir = 0
    land?.paw(true, pawDir)
  }
  const onPawFocus = () => land?.paw(true, (pawDir = 0))
  const onPawBlur = () => land?.paw(false, (pawDir = 0))
  const onPawPress = () => land?.patPaw()
  petter?.addEventListener('keydown', onPaw)
  petter?.addEventListener('keyup', onPaw)
  petter?.addEventListener('focus', onPawFocus)
  petter?.addEventListener('blur', onPawBlur)
  petter?.addEventListener('click', onPawPress)

  let raf = 0
  let last = performance.now()
  let clock = 0
  let pitch = 0
  const onGrid = (v: number, origin: number, px: number) =>
    px ? origin + Math.floor((v - origin) / px) * px : v
  const placeFlies = () => {
    const grid = land?.pixel()
    if (grid && grid.px !== pitch) {
      pitch = grid.px
      for (const fly of flies) {
        for (const [el, n] of [
          [fly.halo, 3],
          [fly.core, 1],
        ] as const) {
          el.setAttribute('x', (-(n >> 1) * pitch).toFixed(2))
          el.setAttribute('y', (-(n >> 1) * pitch).toFixed(2))
          el.setAttribute('width', (n * pitch).toFixed(2))
          el.setAttribute('height', (n * pitch).toFixed(2))
        }
      }
    }
    for (const fly of flies) {
      const x = onGrid(fly.x, grid?.x0 ?? 0, pitch)
      const y = onGrid(fly.y, grid?.y0 ?? 0, pitch)
      fly.g.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)})`)
    }
  }

  function frame(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000)
    last = now
    clock += dt

    parallax.x += (pointer.nx - parallax.x) * 0.05
    parallax.y += (pointer.ny - parallax.y) * 0.05
    layers.forEach((layer, i) => {
      const d = depths[i]
      layer.style.transform = `translate3d(${(-parallax.x * d * 64).toFixed(2)}px, ${(-parallax.y * d * 30).toFixed(2)}px, 0)`
    })

    for (const fly of flies) {
      fly.a += (Math.random() - 0.5) * 3 * dt
      const cx = fly.x < 200 ? 1 : fly.x > 1400 ? -1 : 0
      const cy = fly.y < 360 ? 1 : fly.y > 960 ? -1 : 0
      if (cx || cy) fly.a += Math.sin(Math.atan2(cy, cx) - fly.a) * 2.5 * dt
      let vx = Math.cos(fly.a) * fly.v
      let vy = Math.sin(fly.a) * fly.v
      const dx = pointer.x - fly.x
      const dy = pointer.y - fly.y
      const dist = Math.hypot(dx, dy)
      if (dist < 260 && dist > 30) {
        vx += (dx / dist) * 34
        vy += (dy / dist) * 34
      }
      fly.x += vx * dt
      fly.y += vy * dt
      const glow = Math.max(0, Math.sin(fly.phase + clock * fly.freq))
      fly.g.style.opacity = (0.15 + 0.85 * glow ** 3).toFixed(3)
    }
    placeFlies()

    for (let i = falling.length - 1; i >= 0; i--) {
      const p = falling[i]
      if (p.y < p.floor) {
        p.vy += GRAVITY * dt
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.rot += p.vr * dt
        if (p.y >= p.floor) {
          if (p.letter) splash(p.x, p.floor)
          p.vx *= 0.1
          p.vy = 14
          p.vr *= 0.1
        }
      } else {
        p.sunk += dt
        p.y += p.vy * dt
        p.rot += p.vr * dt
        p.el.style.opacity = Math.max(0, 1 - p.sunk * (p.letter ? 0.7 : 6)).toFixed(3)
        if (p.sunk > (p.letter ? 1.5 : 0.2)) {
          p.el.remove()
          falling.splice(i, 1)
          continue
        }
      }
      const grid = p.letter ? null : land?.pixel()
      const x = grid ? onGrid(p.x, grid.x0, grid.px) : p.x
      const y = grid ? onGrid(p.y, grid.y0, grid.px) : p.y
      p.el.setAttribute(
        'transform',
        `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${p.rot.toFixed(1)})`,
      )
    }

    hands.step(dt)
    land?.step(dt, clock, pointer)
    raf = requestAnimationFrame(frame)
  }

  const start = () => {
    if (raf || !active || document.hidden) return
    last = performance.now()
    raf = requestAnimationFrame(frame)
  }
  const stop = () => {
    cancelAnimationFrame(raf)
    raf = 0
  }
  const onVisibility = () => (document.hidden ? stop() : start())

  if (reduce) {
    placeFlies()
  } else {
    document.addEventListener('visibilitychange', onVisibility)
    start()
    later(crumble, 4200)
  }

  const cleanup = () => {
    if (!active) return
    active = false
    if (cleanupPrev === cleanup) cleanupPrev = null
    locating.abort()
    land?.dispose()
    stop()
    timers.forEach(id => clearTimeout(id))
    window.removeEventListener('pointermove', onMove)
    document.documentElement.removeEventListener('pointerleave', onLeave)
    document.removeEventListener('visibilitychange', onVisibility)
    door.removeEventListener('pointerenter', portalOn)
    door.removeEventListener('pointerleave', portalOff)
    door.removeEventListener('focus', portalOn)
    door.removeEventListener('blur', portalOff)
    door.removeEventListener('click', onDoor)
    scene.removeEventListener('click', onHail, true)
    scene.removeEventListener('pointerdown', onPat)
    scene.removeEventListener('touchstart', onTouch)
    boardDetails?.removeEventListener('toggle', ensureDepartures)
    petter?.removeEventListener('keydown', onPaw)
    petter?.removeEventListener('keyup', onPaw)
    petter?.removeEventListener('focus', onPawFocus)
    petter?.removeEventListener('blur', onPawBlur)
    petter?.removeEventListener('click', onPawPress)
  }
  cleanupPrev = cleanup
  window.addCleanup(cleanup)
})
