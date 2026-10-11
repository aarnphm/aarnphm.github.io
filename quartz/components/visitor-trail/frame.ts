import { playhtml } from 'playhtml'
import { mountAlpineJourneys } from './alpine-journey'
import { trailBackpacker } from './backpacker'
import { trailCampsite } from './campsite'
import { mountGroundDepth } from './depth-order'
import { footprint, footprintTransform, type FootprintKind } from './footprints'
import { forestControlsMarkup, mountForestControls } from './forest-controls'
import { mountHikerJourney } from './hiker-journey'
import { trailHinterland } from './hinterland'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { clearSpace, trailIds, trailLayout, type TrailId } from './layout'
import { trailMountainBackdrop } from './mountain-backdrop'
import { trailPasture } from './pasture'
import { mountPastureWanderers } from './pasture-wander'
import { mountTrailPerspective } from './perspective'
import { trailRandom } from './random'
import { defaultForestSettings, trailTrees } from './scenery'
import { trailTerrain } from './terrain'
import { mountMapReaderJourneys } from './wayfarer-journey'
import { mountTrailWeather } from './weather'

type TrailData = { stones: Record<string, number>; routes?: Record<string, TrailId> }
type TrailSample = { x: number; y: number; position: number }
type TrailGeometry = {
  id: TrailId
  length: number
  samples: TrailSample[]
  delay: number
  duration: number
}

const document = parent.document
const host = window.frameElement?.parentElement
if (!host?.matches('.homepage-trail-scene') || parent === window)
  throw new Error('The trail needs its homepage host')
const controller = new AbortController()
const { signal } = controller
const scene = document.createElement('div')
scene.className = 'trail-scene'
const clip = `trail-clear-${crypto.randomUUID()}`
scene.innerHTML = `
    <svg class="trail-canvas" aria-hidden="true">
      <defs>
        <clipPath id="${clip}"><path data-trail-clear clip-rule="evenodd"/></clipPath>
        <clipPath id="${clip}-backdrop"><path data-trail-backdrop-clear clip-rule="evenodd"/></clipPath>
        <g data-pet-walk-masks></g>
        ${trailIds.map(id => `<mask id="${clip}-${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="100%" height="100%"><path class="trail-reveal" data-trail-reveal="${id}" pathLength="1"/></mask>`).join('')}
      </defs>
      ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
      <g data-pet-walk-paths></g>
      <g clip-path="url(#${clip})">
      <g data-trail-backdrop-container></g>
      <g class="trail-ground"></g>
      <g class="trail-hinterland"></g>
      <g class="trail-weather" aria-hidden="true"></g>
      <g class="trail-steps"></g>
      <g class="trail-hiker-tracks"></g>
      <g class="trail-reader-tracks"></g>
      <g class="trail-landmarks"></g>
      <g class="trail-fauna"></g>
      <g class="trail-wayfarers"></g>
      ${trailBackpacker()}
      <g class="trail-depth-objects"></g>
      <g data-trail-stones>${trailIds.map(id => `<g data-trail-stone-route="${id}" mask="url(#${clip}-${id})"></g>`).join('')}</g>
      </g>
    </svg>
    <div class="trail-stone" role="slider" aria-label="Your stone on the shared trail" tabindex="-1"
      aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-describedby="trail-hint" aria-disabled="true"
      aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home End">
      <span class="trail-stone-owner" aria-hidden="true">you</span>
      <svg viewBox="-4 -4 20 20" aria-hidden="true"><circle class="trail-stone-ring" cx="6" cy="6" r="9"/><path d="M2 8 4 3 8 2 11 7 9 10 3 10Z"/></svg>
    </div>
    <div class="trail-campsite">
      <button class="trail-camp-toggle" type="button" aria-label="Trail controls" aria-expanded="false" aria-controls="trail-camp-tools" aria-describedby="trail-hint">
        <svg viewBox="0 0 128 96" aria-hidden="true"><g class="trail-camp" transform="translate(64 64)">${trailCampsite()}</g></svg>
      </button>
      <div class="trail-camp-tools" id="trail-camp-tools" hidden>
        <span class="trail-status" id="trail-hint">drag between trails; arrow keys walk</span>
        <svg class="trail-hint-fumes" viewBox="0 0 240 180" aria-hidden="true">
          <defs>
            <path id="${clip}-hint-walk" d="M72 91C97 65 131 69 157 83S192 103 216 82"/>
            <path id="${clip}-hint-drag" d="M142 157C166 136 199 136 222 149S237 161 242 153"/>
          </defs>
          <g class="trail-hint-wisp trail-hint-wisp-middle">
            <text><textPath href="#${clip}-hint-walk" startOffset="50%" text-anchor="middle">arrow keys walk</textPath></text>
          </g>
          <g class="trail-hint-wisp trail-hint-wisp-low">
            <text><textPath href="#${clip}-hint-drag" startOffset="50%" text-anchor="middle">drag your stone</textPath></text>
          </g>
        </svg>
        <div class="trail-tools">
          <button type="button" data-trail-retry hidden>reconnect</button>
        </div>
        <span class="trail-status" data-trail-count role="status">connecting</span>
      </div>
    </div>
    ${forestControlsMarkup(clip)}`
host.append(scene)
const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
const stones = scene.querySelector<SVGGElement>('[data-trail-stones]')
const stone = scene.querySelector<HTMLElement>('.trail-stone')
const count = scene.querySelector<HTMLElement>('[data-trail-count]')
const retry = scene.querySelector<HTMLButtonElement>('[data-trail-retry]')
const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
const campToggle = scene.querySelector<HTMLButtonElement>('.trail-camp-toggle')
const campTools = scene.querySelector<HTMLElement>('.trail-camp-tools')
if (!svg || !stones || !stone || !count || !retry || !campsite || !campToggle || !campTools)
  throw new Error('Shared trail markup is incomplete')

let campTimer = 0
let campPinned = false
let forest = { ...defaultForestSettings }
let forestFrame = 0
let forestLayout: Landscape | undefined
let forestObstacles: Point[] = []
const drawForest = (sort = true) => {
  if (host.hidden || !forestLayout || !scene.isConnected) return
  groundDepth.restore()
  const landmarks = scene.querySelector('.trail-landmarks')
  if (landmarks) landmarks.innerHTML = trailTrees(forestLayout, sceneSeed, forest, forestObstacles)
  const foothillTrees = scene.querySelector('.trail-foothill-trees')
  if (foothillTrees && landmarks)
    foothillTrees.replaceChildren(...landmarks.querySelectorAll('[data-tree-depth="foothill"]'))
  weather.refreshForest()
  pastureJourneys.refreshForest()
  if (sort) groundDepth.refresh()
}
const showTools = (open: boolean) => {
  parent.clearTimeout(campTimer)
  campTools.hidden = !open
  campToggle.setAttribute('aria-expanded', String(open))
  scene.dataset.campTools = open ? 'open' : 'closed'
}
const forestControls = mountForestControls(
  scene,
  signal,
  settings => {
    forest = settings
    if (forestFrame) return
    forestFrame = parent.requestAnimationFrame(() => {
      forestFrame = 0
      drawForest()
    })
  },
  () => {
    campPinned = false
    showTools(false)
  },
)
stone.addEventListener('focus', forestControls.close, { signal })
const leaveCamp = () => {
  parent.clearTimeout(campTimer)
  campTimer = parent.setTimeout(() => {
    if (campPinned || campsite.contains(document.activeElement)) return
    showTools(false)
  }, 180)
}
campsite.addEventListener(
  'pointerenter',
  event => {
    if (event.pointerType !== 'touch' && !forestControls.isOpen()) showTools(true)
  },
  { signal },
)
campsite.addEventListener('pointerleave', leaveCamp, { signal })
campsite.addEventListener(
  'focusin',
  event => {
    if (!forestControls.isOpen() && (event.target === campToggle || event.target === retry))
      showTools(true)
  },
  { signal },
)
campsite.addEventListener('focusout', leaveCamp, { signal })
campToggle.addEventListener(
  'click',
  () => {
    forestControls.close()
    campPinned = !campPinned
    showTools(campPinned)
  },
  { signal },
)
campsite.addEventListener(
  'keydown',
  event => {
    if (event.key !== 'Escape') return
    if (forestControls.isOpen()) {
      event.preventDefault()
      event.stopPropagation()
      forestControls.close()
      forestControls.toggle.focus({ preventScroll: true })
      return
    }
    if (campTools.hidden) return
    event.preventDefault()
    event.stopPropagation()
    campPinned = false
    campToggle.focus({ preventScroll: true })
    showTools(false)
  },
  { signal },
)
retry.addEventListener('click', () => location.reload(), { signal })

let ownPosition = 0.1
let ownRoute: TrailId = 'main'
let sceneSeed = Math.floor(Math.random() * 4294967296)
const openingOrder = (seed: number) => {
  const random = trailRandom(seed ^ 0x68bc21eb)
  const order: TrailId[] = trailIds.filter(id => id !== 'main')
  for (let index = order.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1))
    ;[order[index], order[other]] = [order[other], order[index]]
  }
  order.unshift('main')
  return order
}
let revealOrder = openingOrder(sceneSeed)
let geometries: TrailGeometry[] = []
let redrawStones = () => {}
let layoutSignature = ''
let clearSignature = ''
let obstacles: { left: number; right: number; top: number; bottom: number }[] = []
const point = (position: number, route: TrailId = ownRoute) => {
  const geometry = geometries.find(geometry => geometry.id === route)
  if (!geometry) return { x: 0, y: 0 }
  const { length, samples } = geometry
  const index = Math.min(samples.length - 1, Math.floor((position * length) / 4))
  const a = samples[index]
  const b = samples[Math.min(index + 1, samples.length - 1)]
  if (!a || !b) return { x: 0, y: 0 }
  const fraction =
    a.position === b.position ? 0 : (position - a.position) / (b.position - a.position)
  return { x: a.x + (b.x - a.x) * fraction, y: a.y + (b.y - a.y) * fraction }
}
const journey = mountHikerJourney(scene, host, signal)
const readerJourneys = mountMapReaderJourneys(scene, host, signal)
const pastureJourneys = mountPastureWanderers(scene, host, signal)
const alpineJourneys = mountAlpineJourneys(scene, host, signal)
const perspective = mountTrailPerspective(scene, host, signal)
const weather = mountTrailWeather(scene, host, signal)
const groundDepth = mountGroundDepth(scene, host, signal)
const placeOwnStone = (position: number, route: TrailId = ownRoute) => {
  ownPosition = position
  ownRoute = route
  stone.dataset.route = route
  const p = point(position)
  stone.style.left = `${p.x}px`
  stone.style.top = `${p.y}px`
  const geometry = geometries.find(geometry => geometry.id === route)
  stone.style.setProperty(
    '--trail-arrival',
    `${Math.round((geometry?.delay ?? 0) + position * (geometry?.duration ?? 0))}ms`,
  )
  // A slider's generous hit area must leave the adjacent text selectable.
  const cuts = obstacles.filter(
    rect =>
      rect.left < p.x + 22 &&
      rect.right > p.x - 22 &&
      rect.top < p.y + 22 &&
      rect.bottom > p.y - 22,
  )
  const outline = clearSpace(
    44,
    44,
    cuts.map(rect => ({
      left: rect.left - p.x + 22,
      right: rect.right - p.x + 22,
      top: rect.top - p.y + 22,
      bottom: rect.bottom - p.y + 22,
    })),
  )
  stone.style.clipPath = `path(evenodd, '${outline}')`
  stone.setAttribute('aria-valuenow', String(Math.round(position * 100)))
  stone.setAttribute(
    'aria-valuetext',
    `${route} trail, ${Math.round(position * 100)} percent along the path`,
  )
}
const drawScene = () => {
  if (host.hidden || !scene.isConnected) return
  const layout = trailLayout(scene, sceneSeed)
  if (!layout) return
  const signature = `${sceneSeed}|${layout.width}|${layout.height}|${layout.bottom}|${JSON.stringify(layout.readingRoutes)}|${JSON.stringify(layout.routes)}`
  const clearance = `${layout.clear}|${layout.backdropClear}`
  if (signature === layoutSignature && clearance === clearSignature) return
  const { width, height } = layout
  obstacles = layout.obstacles
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  svg.querySelector('[data-trail-clear]')?.setAttribute('d', layout.clear)
  svg.querySelector('[data-trail-backdrop-clear]')?.setAttribute('d', layout.backdropClear)
  clearSignature = clearance
  if (signature === layoutSignature) {
    placeOwnStone(ownPosition)
    return
  }
  groundDepth.restore()
  layoutSignature = signature
  const readingPaths = svg.querySelector('[data-pet-walk-paths]')
  if (readingPaths)
    readingPaths.innerHTML = layout.readingRoutes
      .map(
        (route, index) => `<path class="trail-path" data-pet-walk-path="${index}" d="${route}"/>`,
      )
      .join('')
  const readingMasks = svg.querySelector('[data-pet-walk-masks]')
  if (readingMasks)
    readingMasks.innerHTML = layout.readingRoutes
      .map(
        (route, index) =>
          `<mask id="${clip}-reading-${index}" maskUnits="userSpaceOnUse" x="0" y="0" width="100%" height="100%"><path class="trail-reveal" data-pet-walk-reveal="${index}" pathLength="1" d="${route}" style="--trail-duration:4200ms;--trail-delay:${index * 550}ms"/></mask>`,
      )
      .join('')
  geometries = trailIds.flatMap(id => {
    const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
    if (!path) return []
    path.setAttribute('d', layout.routes[id])
    const length = path.getTotalLength()
    const delay = revealOrder.indexOf(id) * 500
    const duration = Math.round(Math.max(3000, Math.min(4200, length * 2.8)))
    const reveal = svg.querySelector<SVGPathElement>(`[data-trail-reveal="${id}"]`)
    reveal?.setAttribute('d', layout.routes[id])
    reveal?.style.setProperty('--trail-delay', `${delay}ms`)
    reveal?.style.setProperty('--trail-duration', `${duration}ms`)
    const samples = Array.from({ length: Math.ceil(length / 4) + 1 }, (_, index) => {
      const position = Math.min(1, (index * 4) / length)
      const p = path.getPointAtLength(position * length)
      return { position, x: p.x, y: p.y }
    })
    return [{ id, length, samples, delay, duration }]
  })
  const steps = scene.querySelector('.trail-steps')
  if (steps) {
    const kinds: FootprintKind[] = ['boot', 'paw', 'hoof', 'blob']
    const offset = Math.floor(trailRandom(sceneSeed ^ 0x2c681f4b)() * kinds.length)
    steps.innerHTML = geometries
      .map(({ id, length }, index) => {
        const kind = kinds[(index + offset) % kinds.length]
        const marks = []
        for (let distance = 6, step = 0; distance < length - 6; distance += 12, step++) {
          const p = point(distance / length, id)
          const ahead = point(Math.min(1, (distance + 1) / length), id)
          const angle = Math.atan2(ahead.y - p.y, ahead.x - p.x)
          const scale = trailGroundScale(p.y, layout.landscape)
          marks.push(
            footprint(
              kind,
              `${footprintTransform(p.x, p.y, angle, step % 2 ? 1 : -1)} scale(${scale.toFixed(3)})`,
            ),
          )
        }
        return `<g data-route="${id}" mask="url(#${clip}-${id})">${marks.join('')}</g>`
      })
      .join('')
    for (const readingPath of svg.querySelectorAll<SVGPathElement>('[data-pet-walk-path]')) {
      const length = readingPath.getTotalLength()
      const marks = []
      for (let distance = 6; distance < length - 6; distance += 12) {
        const p = readingPath.getPointAtLength(distance)
        const ahead = readingPath.getPointAtLength(Math.min(length, distance + 1))
        marks.push(
          footprint(
            'blob',
            footprintTransform(p.x, p.y, Math.atan2(ahead.y - p.y, ahead.x - p.x), 1),
          ),
        )
      }
      steps.insertAdjacentHTML(
        'beforeend',
        `<g data-route="reading" mask="url(#${clip}-reading-${readingPath.dataset.petWalkPath})">${marks.join('')}</g>`,
      )
    }
  }
  campsite.style.left = `${layout.camp.x}px`
  campsite.style.top = `${layout.camp.y}px`
  forestControls.toggle.style.left = `${layout.camp.x - 92}px`
  forestControls.toggle.style.top = `${layout.camp.y + 10}px`
  scene.style.setProperty(
    '--trail-camp-arrival',
    `${Math.max(...geometries.map(({ delay, duration }) => delay + duration))}ms`,
  )
  const hinterland = scene.querySelector('.trail-hinterland')
  if (hinterland) hinterland.innerHTML = trailHinterland(width, layout.camp, sceneSeed)
  const ground = scene.querySelector('.trail-ground')
  if (ground) ground.innerHTML = trailTerrain(layout.landscape, sceneSeed)
  const backdrop = scene.querySelector('[data-trail-backdrop-container]')
  if (backdrop)
    backdrop.innerHTML = trailMountainBackdrop(layout.landscape, sceneSeed, `${clip}-backdrop`)
  const pasture = trailPasture(layout.landscape, geometries, sceneSeed)
  const fauna = scene.querySelector('.trail-fauna')
  if (fauna) fauna.innerHTML = pasture.markup
  forestLayout = layout.landscape
  forestObstacles = [
    ...geometries.flatMap(route => route.samples.filter((_, index) => index % 4 === 0)),
    ...pasture.sites,
  ]
  drawForest(false)
  placeOwnStone(ownPosition)
  redrawStones()
  const routes = geometries.map(({ id, length, delay, duration }) => ({
    id,
    length,
    readyAfter: delay + duration,
    point: (distance: number) => point(Math.max(0, Math.min(1, distance / length)), id),
  }))
  journey.update(routes, layout.landscape, sceneSeed)
  readerJourneys.update(routes, layout.landscape, sceneSeed)
  pastureJourneys.update(layout.landscape, sceneSeed)
  alpineJourneys.update(layout.landscape, sceneSeed)
  perspective.update()
  weather.update(layout.landscape, sceneSeed)
  groundDepth.refresh()
  scene.dataset.trailRendered = 'true'
}
host.addEventListener('trail:reflow', drawScene, { signal })
drawScene()

const unavailable = () => {
  count.textContent = 'trail unavailable'
  retry.hidden = false
  scene.dataset.trailState = 'unavailable'
}
let timeout = 0
let connectionStarted = false
let disposeConnection = () => {}
const dispose = () => {
  if (signal.aborted) return
  controller.abort()
  parent.clearTimeout(campTimer)
  parent.clearTimeout(timeout)
  if (forestFrame) parent.cancelAnimationFrame(forestFrame)
  disposeConnection()
  scene.remove()
}
host.addEventListener('trail:dispose', dispose, { once: true, signal })
window.addEventListener('pagehide', dispose, { once: true })

const connect = async () => {
  await playhtml.init({ room: '/shared-trail-v1', host: 'api.playhtml.fun', onError: unavailable })
  if (signal.aborted || !scene.isConnected) return
  parent.clearTimeout(timeout)
  const id = playhtml.users.me.pid
  const channel = playhtml.createPageData<TrailData>('mountain-trail-v1', {
    stones: {},
    routes: {},
  })
  let position = 0.1
  let route: TrailId = 'main'
  let pointer: number | null = null
  let frame = 0
  const readPosition = (value: unknown) =>
    typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null
  const readRoute = (value: unknown): TrailId => trailIds.find(id => id === value) ?? 'main'

  const render = (data: TrailData) => {
    if (
      host.hidden ||
      !data ||
      typeof data !== 'object' ||
      !data.stones ||
      typeof data.stones !== 'object'
    )
      return
    const entries = Object.entries(data.stones).filter(([, value]) => readPosition(value) !== null)
    const fragments = new Map(trailIds.map(id => [id, document.createDocumentFragment()]))
    for (const [visitor, value] of entries) {
      if (visitor === id) continue
      const position = readPosition(value)
      if (position === null) continue
      const visitorRoute = readRoute(data.routes?.[visitor])
      const p = point(position, visitorRoute)
      const pebble = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      pebble.setAttribute('d', 'M-3 1-2-2 1-3 3 0 2 2-2 2Z')
      pebble.setAttribute('transform', `translate(${p.x} ${p.y})`)
      pebble.dataset.route = visitorRoute
      fragments.get(visitorRoute)?.append(pebble)
    }
    for (const [route, fragment] of fragments)
      stones.querySelector(`[data-trail-stone-route="${route}"]`)?.replaceChildren(fragment)
    const saved = readPosition(data.stones[id])
    if (saved !== null && pointer === null && frame === 0) {
      position = saved
      if (data.routes?.[id] !== undefined) route = readRoute(data.routes[id])
      placeOwnStone(position, route)
    }
    const label = `${entries.length} ${entries.length === 1 ? 'stone' : 'stones'}`
    if (count.textContent !== label) count.textContent = label
    retry.hidden = true
    scene.dataset.trailState = 'ready'
  }
  const publish = () =>
    channel.setData(data => {
      data.stones[id] = position
      data.routes ??= {}
      data.routes[id] = route
    })
  const move = (next: number, nextRoute: TrailId = route) => {
    position = Math.max(0, Math.min(1, next))
    route = nextRoute
    placeOwnStone(position, route)
    // Coalesce pointer traffic to one shared update per animation frame.
    if (!frame)
      frame = parent.requestAnimationFrame(() => {
        frame = 0
        publish()
      })
  }
  const initial = readPosition(channel.getData().stones[id])
  if (initial === null) {
    // A stable identity spreads first visits along the trail and deduplicates concurrent tabs.
    const hash = [...id].reduce((total, letter) => (total * 31 + letter.charCodeAt(0)) >>> 0, 0)
    position = 0.08 + (hash % 840) / 1000
    route = trailIds[hash % trailIds.length]
    publish()
  }
  channel.onUpdate(data => {
    // Concurrent first visits can replace the initial map. Restore only our entry on sync.
    if (
      readPosition(data.stones[id]) === null ||
      (pointer === null && frame === 0 && route !== 'main' && data.routes?.[id] === undefined)
    )
      publish()
    render(data)
  })
  redrawStones = () => render(channel.getData())
  render(channel.getData())
  stone.setAttribute('aria-disabled', 'false')
  stone.tabIndex = 0

  const nearest = (target: { x: number; y: number }, routes: TrailId[]) => {
    let best = { position, route }
    let distance = Infinity
    for (const id of routes) {
      const geometry = geometries.find(geometry => geometry.id === id)
      if (!geometry) continue
      for (const sample of geometry.samples) {
        const squared = (sample.x - target.x) ** 2 + (sample.y - target.y) ** 2
        if (squared < distance) {
          distance = squared
          best = { position: sample.position, route: id }
        }
      }
    }
    return best
  }
  stone.addEventListener('keydown', event => {
    if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) return
    const length = geometries.find(geometry => geometry.id === route)?.length
    if (!length) return
    const steps: Record<string, number> = {
      ArrowRight: 16 / length,
      ArrowUp: 16 / length,
      ArrowLeft: -16 / length,
      ArrowDown: -16 / length,
      PageUp: 160 / length,
      PageDown: -160 / length,
    }
    if (event.key === 'Home' || event.key === 'End' || event.key in steps) {
      event.preventDefault()
      event.stopPropagation()
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : position + steps[event.key]
      if (next < 0 && (route === 'grove' || route === 'switchback')) {
        const stem = nearest(point(0, route), ['main'])
        const main = geometries.find(geometry => geometry.id === 'main')
        if (main) move(stem.position + (next * length) / main.length, 'main')
      } else move(next)
      stone.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    }
  })
  const project = (event: PointerEvent) => {
    const matrix = svg.getScreenCTM()
    if (!matrix) return
    const target = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    const best = nearest(target, [route, ...trailIds.filter(id => id !== route)])
    move(best.position, best.route)
  }
  stone.addEventListener('pointerdown', event => {
    if (pointer !== null || event.button !== 0) return
    pointer = event.pointerId
    stone.setPointerCapture(pointer)
    stone.dataset.dragging = 'true'
    event.preventDefault()
    stone.focus({ preventScroll: true })
  })
  stone.addEventListener('pointermove', event => {
    if (event.pointerId === pointer) project(event)
  })
  const finish = (event: PointerEvent) => {
    if (event.pointerId !== pointer) return
    if (stone.hasPointerCapture(event.pointerId)) stone.releasePointerCapture(event.pointerId)
    pointer = null
    delete stone.dataset.dragging
    if (frame) {
      parent.cancelAnimationFrame(frame)
      frame = 0
    }
    publish()
  }
  stone.addEventListener('pointerup', finish)
  stone.addEventListener('pointercancel', finish)
  stone.addEventListener('lostpointercapture', finish)
  disposeConnection = () => {
    if (frame) {
      parent.cancelAnimationFrame(frame)
      frame = 0
      publish()
    }
    channel.destroy()
  }
}
const startConnection = () => {
  if (host.hidden || connectionStarted || signal.aborted) return
  connectionStarted = true
  scene.dataset.trailState = 'connecting'
  timeout = parent.setTimeout(unavailable, 12000)
  void connect().catch(unavailable)
}
host.addEventListener(
  'trail:show',
  () => {
    const nextSeed = Math.floor(Math.random() * 4294967296)
    sceneSeed = nextSeed === sceneSeed ? (sceneSeed + 1) >>> 0 : nextSeed
    revealOrder = openingOrder(sceneSeed)
    weather.resume()
    perspective.resume()
    groundDepth.resume()
    startConnection()
  },
  { signal },
)
host.addEventListener(
  'trail:hide',
  () => {
    journey.pause()
    readerJourneys.pause()
    pastureJourneys.pause()
    alpineJourneys.pause()
    perspective.pause()
    groundDepth.pause()
    weather.pause()
    campPinned = false
    showTools(false)
    forestControls.close()
  },
  { signal },
)
startConnection()
