import { playhtml } from 'playhtml'
import { readTrunks, type CollisionActor } from './actor-spacing'
import { mountAlpineJourneys } from './alpine-journey'
import { trailBackpacker } from './backpacker'
import { mountCampCommunity } from './camp-community'
import { placeCampHints } from './camp-hints'
import { campSites } from './camp-layout'
import { connectCampPresence } from './camp-presence'
import { defaultCampSettings } from './camp-settings'
import { trailCampsite } from './campsite'
import { mountGroundDepth } from './depth-order'
import { footprint, footprintTransform, type FootprintKind } from './footprints'
import { forestControlsMarkup, mountForestControls } from './forest-controls'
import { mountHikerJourney } from './hiker-journey'
import { trailHinterland } from './hinterland'
import { mountTrailHolidays } from './holidays'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { trailIds, trailLayout, type TrailId } from './layout'
import { trailMountainBackdrop } from './mountain-backdrop'
import { navigableTrail, type NavigableTrail } from './obstacle-route'
import { mountParkPatrol } from './park-patrol'
import { defaultAnimalSettings, trailPasture } from './pasture'
import { mountPastureWanderers } from './pasture-wander'
import { mountTrailPerspective } from './perspective'
import { petExpedition } from './pet-expedition'
import { trailRandom } from './random'
import {
  createTreePainter,
  defaultForestSettings,
  trailFoothillWoodland,
  trailTreeSprites,
} from './scenery'
import { mountSharedGrove } from './shared-grove'
import { trailTerrain } from './terrain'
import { defaultTravelerSettings } from './traveler-settings'
import { mountVisitorJourneys } from './visitor-journey'
import { mountMapReaderJourneys } from './wayfarer-journey'
import { mountTrailWeather } from './weather'

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
        <path id="${clip}-hint-visitors" data-camp-hint-route="grove"/>
        <g data-trail-navigation></g>
        <g data-pet-walk-masks></g>
        ${trailIds.map(id => `<mask id="${clip}-${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="100%" height="100%"><path class="trail-reveal" data-trail-reveal="${id}" pathLength="1"/></mask>`).join('')}
      </defs>
      ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
      <g data-pet-walk-paths></g>
      <path class="trail-path" data-pet-expedition/>
      <g clip-path="url(#${clip})">
      <g data-trail-backdrop-container></g>
      <g class="trail-woodland" data-trail-depth="0.8"></g>
      <g class="trail-ground"></g>
      <g class="trail-hinterland"></g>
      <g class="trail-weather" aria-hidden="true"></g>
      <g class="trail-steps"></g>
      <g class="trail-hiker-tracks"></g>
      <g class="trail-reader-tracks"></g>
      <g class="trail-camp-tracks"></g>
      <g class="trail-visitor-tracks"></g>
      <g class="trail-ranger-tracks"></g>
      <g class="trail-park-patrol"></g>
      <g class="trail-landmarks"></g>
      <g class="trail-fauna"></g>
      <g class="trail-holidays" aria-hidden="true"></g>
      <g class="trail-wayfarers"></g>
      <g class="trail-community"></g>
      <g class="trail-visitors"></g>
      ${trailBackpacker()}
      <g class="trail-depth-objects"></g>
      </g>
      <g class="trail-hint-fumes" clip-path="url(#${clip})">
        <text dy="-9"><textPath href="#${clip}-hint-visitors" startOffset="50%" text-anchor="middle">meet at the fire</textPath></text>
      </g>
    </svg>
    <div class="trail-visitor-labels"></div>
    <div class="trail-shared-cursors" aria-hidden="true"></div>
    <div class="trail-campsite">
      <button class="trail-camp-toggle" type="button" aria-label="Trail controls" aria-expanded="false" aria-controls="trail-camp-tools" aria-describedby="trail-hint">
        <svg viewBox="0 0 128 96" aria-hidden="true"><g class="trail-camp" transform="translate(64 64)">${trailCampsite(false)}</g></svg>
      </button>
      <div class="trail-camp-tools" id="trail-camp-tools" hidden>
        <span class="trail-status" id="trail-hint">walk to camp together</span>
        <div class="trail-tools">
          <button type="button" data-trail-retry hidden>reconnect</button>
        </div>
        <span class="trail-status" data-trail-count role="status">connecting</span>
      </div>
    </div>
    ${forestControlsMarkup(clip)}`
host.append(scene)
const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
const count = scene.querySelector<HTMLElement>('[data-trail-count]')
const retry = scene.querySelector<HTMLButtonElement>('[data-trail-retry]')
const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
const campToggle = scene.querySelector<HTMLButtonElement>('.trail-camp-toggle')
const campTools = scene.querySelector<HTMLElement>('.trail-camp-tools')
if (!svg || !count || !retry || !campsite || !campToggle || !campTools)
  throw new Error('Shared trail markup is incomplete')

let campTimer = 0
let hintFrame = 0
let campPinned = false
campsite.dataset.revealed = String(parent.matchMedia('(prefers-reduced-motion: reduce)').matches)
campsite.addEventListener(
  'animationend',
  event => {
    if (event.target === campsite && event.animationName === 'trail-arrive')
      campsite.dataset.revealed = 'true'
  },
  { signal },
)
let forest = { ...defaultForestSettings }
let forestFrame = 0
let animals = { ...defaultAnimalSettings }
let animalFrame = 0
let travelers = { ...defaultTravelerSettings }
let travelerFrame = 0
let camps = { ...defaultCampSettings }
let campFrame = 0
let campRegenerate = false
let forestLayout: Landscape | undefined
let forestObstacles: Point[] = []
const paintTree = createTreePainter()
const paintWoodland = createTreePainter()
const forestSprites = (landscape: Landscape) =>
  trailTreeSprites(
    landscape,
    sceneSeed,
    forest,
    forestObstacles,
    campSites(landscape, sceneSeed, camps).flatMap(site => [
      { ...site, radius: 102 },
      ...site.tents.map(tent => ({ ...tent, radius: 38 })),
    ]),
    paintTree,
  )
const placeForest = (markup: string, sort = true) => {
  const landmarks = scene.querySelector('.trail-landmarks')
  if (landmarks) landmarks.innerHTML = markup
  const woodland = scene.querySelector('.trail-woodland')
  if (woodland && forestLayout)
    woodland.innerHTML = trailFoothillWoodland(
      forestLayout,
      sceneSeed,
      forest,
      forestObstacles,
      paintWoodland,
      [...(landmarks?.querySelectorAll<SVGGElement>('[data-tree-id]') ?? [])].map(tree => ({
        x: Number(tree.dataset.groundX),
        y: Number(tree.dataset.groundY),
      })),
    )
  const foothillTrees = scene.querySelector('.trail-foothill-trees')
  if (foothillTrees && landmarks)
    foothillTrees.replaceChildren(...landmarks.querySelectorAll('[data-tree-depth="foothill"]'))
  weather.refreshForest()
  holidays.refreshTrees()
  grove.refresh()
  pastureJourneys.refreshForest()
  if (sort) groundDepth.refresh()
}
const drawForest = (sort = true) => {
  if (host.hidden || !forestLayout || !scene.isConnected) return
  groundDepth.restore()
  placeForest([...forestSprites(forestLayout)].join(''), sort)
}
const drawAnimals = () => {
  if (host.hidden || !forestLayout || !scene.isConnected) return
  groundDepth.restore()
  const fauna = scene.querySelector('.trail-fauna')
  if (fauna) fauna.innerHTML = trailPasture(forestLayout, geometries, sceneSeed, animals).markup
  pastureJourneys.update(forestLayout, sceneSeed)
  groundDepth.refresh()
}
const holidays = mountTrailHolidays(scene, signal, () => {
  if (!forestLayout || host.hidden || scene.dataset.trailRendered !== 'true') return
  groundDepth.restore()
  holidays.draw(forestLayout, geometries, sceneSeed, camps)
  pastureJourneys.update(forestLayout, sceneSeed)
  groundDepth.refresh()
})
let navigationGeometry: TrailGeometry[] | null = null
let navigationSignature = ''
let navigationRoutes: NavigableTrail[] = []
let navigationBodies: CollisionActor[] = []
const journeyRoutes = () => {
  if (!forestLayout) return []
  const landscape = forestLayout
  const solids = new Map(
    readTrunks(scene)
      .filter(solid => solid.id.startsWith('trunk-'))
      .map(solid => [solid.id, solid]),
  )
  for (const site of campSites(landscape, sceneSeed, camps)) {
    const hearth = {
      ...site,
      y: site.y + 16,
      id: `hearth-${site.id}`,
      kind: 'trunk',
      radius: 16,
    } satisfies CollisionActor
    solids.set(hearth.id, hearth)
    for (const [index, tent] of site.tents.entries()) {
      const body = {
        ...tent,
        id: `tent-${site.id}-${index}`,
        kind: 'trunk',
        radius: 28,
      } satisfies CollisionActor
      solids.set(body.id, body)
    }
  }
  const bodies = [...solids.values()]
  const signature = bodies.map(body => `${body.id}:${body.x}:${body.y}:${body.radius}`).join('|')
  if (navigationGeometry === geometries && navigationSignature === signature)
    return navigationRoutes
  navigationGeometry = geometries
  navigationSignature = signature
  navigationBodies = bodies
  navigationRoutes = geometries.flatMap(({ id, length, delay, duration }) => {
    const route = navigableTrail(
      {
        id,
        length,
        readyAfter: delay + duration,
        point: (distance: number) => point(Math.max(0, Math.min(1, distance / length)), id),
      },
      landscape,
      bodies,
    )
    return route ? [route] : []
  })
  const paths = scene.querySelector<SVGGElement>('[data-trail-navigation]')
  if (paths) {
    paths.dataset.navigationSolids = JSON.stringify(
      bodies.map(({ id, x, y, radius }) => ({ id, x, y, radius })),
    )
    paths.innerHTML = navigationRoutes
      .map(
        route =>
          `<path data-navigation-route="${route.id}" data-detours="${route.detours}" d="${route.d}"/>`,
      )
      .join('')
  }
  return navigationRoutes
}
const drawTravelers = () => {
  if (host.hidden || !forestLayout || !scene.isConnected) return
  groundDepth.restore()
  const routes = journeyRoutes()
  journey.update(routes, forestLayout, sceneSeed, travelers.walkers > 0)
  readerJourneys.update(
    routes,
    forestLayout,
    sceneSeed,
    Math.max(0, travelers.walkers - 1),
    navigationBodies,
  )
  alpineJourneys.update(forestLayout, sceneSeed, travelers.climbers)
  parkPatrol.update(forestLayout, sceneSeed, routes, travelers.rangers)
  visitorJourneys.update(routes, forestLayout, sceneSeed, camps)
  groundDepth.refresh()
}
const showTools = (open: boolean) => {
  parent.clearTimeout(campTimer)
  parent.cancelAnimationFrame(hintFrame)
  hintFrame = 0
  campTools.hidden = !open
  campToggle.setAttribute('aria-expanded', String(open))
  scene.dataset.campTools = open ? 'open' : 'closed'
  if (open) {
    placeCampHints(scene)
    hintFrame = parent.requestAnimationFrame(() => {
      hintFrame = 0
      if (!signal.aborted && scene.dataset.campTools === 'open') placeCampHints(scene)
    })
  }
}
void document.fonts.ready.then(() => {
  if (!signal.aborted && scene.dataset.campTools === 'open') placeCampHints(scene)
})
const forestControls = mountForestControls(
  scene,
  signal,
  settings => {
    forest = settings
    if (forestFrame) return
    forestFrame = parent.requestAnimationFrame(() => {
      forestFrame = 0
      if (redrawPendingScene()) return
      drawForest()
      if (forestLayout) community.update(forestLayout, sceneSeed, camps, journeyRoutes())
      drawTravelers()
    })
  },
  settings => {
    animals = settings
    if (animalFrame) return
    animalFrame = parent.requestAnimationFrame(() => {
      animalFrame = 0
      if (redrawPendingScene()) return
      drawAnimals()
    })
  },
  settings => {
    travelers = settings
    if (travelerFrame) return
    travelerFrame = parent.requestAnimationFrame(() => {
      travelerFrame = 0
      if (redrawPendingScene()) return
      drawTravelers()
    })
  },
  settings => {
    campRegenerate ||=
      camps.sites !== settings.sites ||
      camps.tents !== settings.tents ||
      camps.builders !== settings.builders
    camps = settings
    if (campFrame) return
    campFrame = parent.requestAnimationFrame(() => {
      campFrame = 0
      if (redrawPendingScene()) {
        campRegenerate = false
        return
      }
      if (!forestLayout || host.hidden) return
      groundDepth.restore()
      if (campRegenerate) drawForest(false)
      campRegenerate = false
      community.update(forestLayout, sceneSeed, camps, journeyRoutes())
      visitorJourneys.update(journeyRoutes(), forestLayout, sceneSeed, camps)
      drawTravelers()
      holidays.draw(forestLayout, geometries, sceneSeed, camps)
      pastureJourneys.update(forestLayout, sceneSeed)
      groundDepth.refresh()
    })
  },
  selection => holidays.select(selection),
  () => {
    campPinned = false
    showTools(false)
  },
)
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
    campsite.dataset.revealed = 'true'
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

const routeRevealDelay = 1400
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
let layoutSignature = ''
let clearSignature = ''
const point = (position: number, route: TrailId) => {
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
const grove = mountSharedGrove(scene, host, signal, () => pastureJourneys.refreshForest())
const refreshStructures = () => {
  // Keep generation holders intact until every owner has replaced its previous sprites.
  if (scene.dataset.trailRendered === 'true') groundDepth.refresh()
}
const community = mountCampCommunity(scene, host, signal, refreshStructures)
const parkPatrol = mountParkPatrol(scene, host, signal)
const visitorJourneys = mountVisitorJourneys(scene, host, signal, refreshStructures)
scene.querySelector('[data-camp-pack]')?.addEventListener(
  'click',
  () => {
    if (scene.dataset.trailRendered === 'true') community.packUp(forestControls.destination.value())
  },
  { signal },
)
let rendering: AbortController | null = null
function redrawPendingScene() {
  if (scene.dataset.trailRendered !== 'false') return false
  layoutSignature = ''
  void drawScene()
  return true
}
const cancelRendering = () => {
  rendering?.abort()
  rendering = null
}
signal.addEventListener('abort', cancelRendering, { once: true })
const yieldRender = (signal: AbortSignal, paint = false) =>
  new Promise<boolean>(resolve => {
    if (signal.aborted) return resolve(false)
    let frame = 0
    let task = 0
    const finish = () => {
      signal.removeEventListener('abort', cancel)
      resolve(!signal.aborted)
    }
    const cancel = () => {
      parent.cancelAnimationFrame(frame)
      parent.clearTimeout(task)
      finish()
    }
    signal.addEventListener('abort', cancel, { once: true })
    if (paint)
      frame = parent.requestAnimationFrame(() => {
        task = parent.setTimeout(finish, 0)
      })
    else task = parent.setTimeout(finish, 0)
  })
const drawScene = async () => {
  if (host.hidden || !scene.isConnected) return
  const layout = trailLayout(scene, sceneSeed)
  if (!layout) return
  const signature = `${sceneSeed}|${layout.width}|${layout.height}|${layout.bottom}|${JSON.stringify(layout.readingRoutes)}|${JSON.stringify(layout.routes)}`
  const clearance = `${layout.clear}|${layout.backdropClear}`
  if (signature === layoutSignature && clearance === clearSignature) return
  const { width, height } = layout
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  svg.querySelector('[data-trail-clear]')?.setAttribute('d', layout.clear)
  svg.querySelector('[data-trail-backdrop-clear]')?.setAttribute('d', layout.backdropClear)
  clearSignature = clearance
  if (signature === layoutSignature) return
  cancelRendering()
  rendering = new AbortController()
  const renderSignal = rendering.signal
  scene.dataset.trailRendered = 'false'
  journey.pause()
  readerJourneys.pause()
  pastureJourneys.pause()
  alpineJourneys.pause()
  community.pause()
  visitorJourneys.pause()
  parkPatrol.pause()
  groundDepth.restore()
  layoutSignature = signature
  campsite.style.left = `${layout.camp.x}px`
  campsite.style.top = `${layout.camp.y}px`
  forestControls.toggle.style.left = `${layout.camp.x - 92}px`
  forestControls.toggle.style.top = `${layout.camp.y + 10}px`
  scene.style.setProperty('--trail-camp-arrival', '0ms')
  // Paint the camp before backdrop, route sampling, and forest generation occupy the main thread.
  if (!(await yieldRender(renderSignal, true))) return
  const backdrop = scene.querySelector('[data-trail-backdrop-container]')
  if (backdrop) {
    const entrances = trailIds.flatMap(id => {
      const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
      if (!path) return []
      path.setAttribute('d', layout.routes[id])
      const { x, y } = path.getPointAtLength(0)
      return [{ x, y }]
    })
    backdrop.innerHTML = trailMountainBackdrop(
      layout.landscape,
      sceneSeed,
      `${clip}-backdrop`,
      entrances,
    )
  }
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
          `<mask id="${clip}-reading-${index}" maskUnits="userSpaceOnUse" x="0" y="0" width="100%" height="100%"><path class="trail-reveal" data-pet-walk-reveal="${index}" pathLength="1" d="${route}" style="--trail-duration:4200ms;--trail-delay:${routeRevealDelay + index * 550}ms"/></mask>`,
      )
      .join('')
  geometries = []
  for (const id of trailIds) {
    const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
    if (!path) continue
    path.setAttribute('d', layout.routes[id])
    const length = path.getTotalLength()
    const delay = routeRevealDelay + revealOrder.indexOf(id) * 500
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
    geometries.push({ id, length, samples, delay, duration })
    if (!(await yieldRender(renderSignal))) return
  }
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
      if (!(await yieldRender(renderSignal))) return
    }
  }
  if (!(await yieldRender(renderSignal))) return
  const hinterland = scene.querySelector('.trail-hinterland')
  if (hinterland) hinterland.innerHTML = trailHinterland(width, layout.camp, sceneSeed)
  const ground = scene.querySelector('.trail-ground')
  if (ground) ground.innerHTML = trailTerrain(layout.landscape, sceneSeed)
  if (!(await yieldRender(renderSignal))) return
  const expedition = petExpedition(scene, layout.landscape)
  const petPath = svg.querySelector<SVGPathElement>('[data-pet-expedition]')
  if (petPath && expedition) {
    petPath.setAttribute('d', expedition.d)
    petPath.dataset.petExpedition = String(sceneSeed)
    petPath.dataset.petScaleStops = JSON.stringify(expedition.scaleStops)
    petPath.dataset.petAlpineSpan = JSON.stringify(expedition.alpineSpan)
    petPath.dataset.petHandoff = JSON.stringify(expedition.handoff)
    petPath.dataset.petMountainTrack = expedition.mountainTrack
    petPath.dataset.petValleyRoute = expedition.valleyRoute
  } else {
    petPath?.removeAttribute('d')
  }
  if (!(await yieldRender(renderSignal))) return
  const pasture = trailPasture(layout.landscape, geometries, sceneSeed, animals)
  const fauna = scene.querySelector('.trail-fauna')
  if (fauna) fauna.innerHTML = pasture.markup
  forestLayout = layout.landscape
  forestObstacles = [
    ...geometries.flatMap(route => route.samples.filter((_, index) => index % 4 === 0)),
    ...pasture.sites,
  ]
  const sprites: string[] = []
  let batch = performance.now()
  for (const sprite of forestSprites(layout.landscape)) {
    sprites.push(sprite)
    if (performance.now() - batch >= 8) {
      if (!(await yieldRender(renderSignal))) return
      batch = performance.now()
    }
  }
  placeForest(sprites.join(''), false)
  if (!(await yieldRender(renderSignal))) return
  const routes = journeyRoutes()
  community.update(layout.landscape, sceneSeed, camps, routes)
  visitorJourneys.update(routes, layout.landscape, sceneSeed, camps)
  parkPatrol.update(layout.landscape, sceneSeed, routes, travelers.rangers)
  journey.update(routes, layout.landscape, sceneSeed, travelers.walkers > 0)
  readerJourneys.update(
    routes,
    layout.landscape,
    sceneSeed,
    Math.max(0, travelers.walkers - 1),
    navigationBodies,
  )
  holidays.draw(layout.landscape, geometries, sceneSeed, camps)
  pastureJourneys.update(layout.landscape, sceneSeed)
  alpineJourneys.update(layout.landscape, sceneSeed, travelers.climbers)
  perspective.update()
  weather.update(layout.landscape, sceneSeed)
  groundDepth.refresh()
  scene.dataset.trailRendered = 'true'
  placeCampHints(scene)
  rendering = null
  host.dispatchEvent(new Event('trail:ready'))
}
host.addEventListener(
  'trail:reflow',
  () => {
    void drawScene()
  },
  { signal },
)
void drawScene()
document.addEventListener(
  'visibilitychange',
  () => {
    if (document.hidden) {
      if (rendering && scene.dataset.trailRendered !== 'true') {
        cancelRendering()
        layoutSignature = ''
      }
    } else if (scene.dataset.trailRendered !== 'true') void drawScene()
  },
  { signal },
)

const unavailable = () => {
  count.textContent = 'trail unavailable'
  retry.hidden = false
  scene.dataset.trailState = 'unavailable'
  const connection = scene.querySelector<HTMLElement>('[data-camp-connection]')
  if (connection) connection.textContent = 'connection unavailable · reconnect at the fire'
  const visitors = document.querySelector<HTMLElement>('[data-trail-visitors]')
  if (visitors) {
    visitors.dataset.state = 'unavailable'
    visitors.textContent = '·'
  }
}
let timeout = 0
let connectionStarted = false
const dispose = () => {
  if (signal.aborted) return
  controller.abort()
  parent.clearTimeout(campTimer)
  parent.cancelAnimationFrame(hintFrame)
  parent.clearTimeout(timeout)
  if (forestFrame) parent.cancelAnimationFrame(forestFrame)
  if (animalFrame) parent.cancelAnimationFrame(animalFrame)
  if (travelerFrame) parent.cancelAnimationFrame(travelerFrame)
  if (campFrame) parent.cancelAnimationFrame(campFrame)
  scene.remove()
}
host.addEventListener('trail:dispose', dispose, { once: true, signal })
window.addEventListener('pagehide', dispose, { once: true })

const connect = async () => {
  await playhtml.init({
    room: '/shared-trail-v1',
    host: 'api.playhtml.fun',
    cursors: { enabled: false },
    onError: unavailable,
  })
  if (signal.aborted || !scene.isConnected) return
  parent.clearTimeout(timeout)
  const id = playhtml.users.me.pid
  grove.connect(id)
  connectCampPresence(scene, host, signal, forestControls.destination, visitorJourneys.setVisitors)
  count.textContent = 'visitors walk to camp'
  retry.hidden = true
  scene.dataset.trailState = 'ready'
}
const startConnection = () => {
  if (connectionStarted || signal.aborted) return
  connectionStarted = true
  scene.dataset.trailState = 'connecting'
  timeout = parent.setTimeout(unavailable, 12000)
  void connect().catch(unavailable)
}
host.addEventListener(
  'trail:show',
  () => {
    campsite.dataset.revealed = String(
      parent.matchMedia('(prefers-reduced-motion: reduce)').matches,
    )
    const nextSeed = Math.floor(Math.random() * 4294967296)
    sceneSeed = nextSeed === sceneSeed ? (sceneSeed + 1) >>> 0 : nextSeed
    revealOrder = openingOrder(sceneSeed)
    weather.resume()
    perspective.resume()
    startConnection()
  },
  { signal },
)
host.addEventListener(
  'trail:hide',
  () => {
    cancelRendering()
    journey.pause()
    readerJourneys.pause()
    pastureJourneys.pause()
    alpineJourneys.pause()
    perspective.pause()
    groundDepth.pause()
    weather.pause()
    community.pause()
    visitorJourneys.pause()
    campPinned = false
    showTools(false)
    forestControls.close()
  },
  { signal },
)
startConnection()
