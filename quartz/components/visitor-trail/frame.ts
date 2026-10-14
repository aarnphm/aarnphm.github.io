import { playhtml } from 'playhtml'
import { svgPathGeometry } from '../../util/svg-geometry'
import { readTrunks, type CollisionActor } from './actor-spacing'
import { mountAlpineJourneys } from './alpine-journey'
import { trailBackpacker } from './backpacker'
import { mountTrailBlazes } from './blazes'
import { mountCampCommunity } from './camp-community'
import { placeCampHints } from './camp-hints'
import { campSites } from './camp-layout'
import { connectCampPresence } from './camp-presence'
import { defaultCampSettings } from './camp-settings'
import { trailCampsite } from './campsite'
import { mountCrownWind } from './crown-wind'
import { mountGroundDepth } from './depth-order'
import { footprint, footprintTransform, type FootprintKind } from './footprints'
import { forestControlsMarkup, mountForestControls } from './forest-controls'
import { createForestSprites } from './forest-sprites'
import { mountHikerJourney } from './hiker-journey'
import { trailHinterland } from './hinterland'
import { mountTrailHolidays } from './holidays'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { trailIds, trailLayout, type TrailId } from './layout'
import { mountLeafFall } from './leaf-fall'
import { trailMountainBackdrop } from './mountain-backdrop'
import { navigableTrail, type NavigableTrail } from './obstacle-route'
import { mountParkPatrol } from './park-patrol'
import { defaultAnimalSettings, trailPasture } from './pasture'
import { mountPastureWanderers } from './pasture-wander'
import { mountTrailPerspective } from './perspective'
import { petExpedition } from './pet-expedition'
import { trailRandom } from './random'
import { mountSceneCompositor } from './scene-compositor'
import { mountSceneReveal, trailOpeningDuration } from './scene-reveal'
import { mountSceneSurface } from './scene-surface'
import {
  createTreePainter,
  defaultForestSettings,
  planTrailTrees,
  type TreePlacement,
  trailFoothillWoodland,
  trailTreeSprites,
} from './scenery'
import { mountSharedGrove } from './shared-grove'
import { trailTerrain, trailTerraces } from './terrain'
import { defaultTravelerSettings } from './traveler-settings'
import { prepareTreeSprites } from './tree-workers'
import { mountVisitorJourneys } from './visitor-journey'
import { mountMapReaderJourneys } from './wayfarer-journey'
import { mountTrailWeather } from './weather'
import { mountWorldPatches } from './world-patches'

type TrailSample = { x: number; y: number; position: number }
type TrailGeometry = { id: TrailId; length: number; samples: TrailSample[] }

const document = parent.document
const host = window.frameElement?.parentElement
if (!host?.matches('.homepage-trail-scene') || parent === window)
  throw new Error('The trail needs its homepage host')
const controller = new AbortController()
const { signal } = controller
const scene = document.createElement('div')
scene.className = 'trail-scene'
scene.dataset.trailRendered = 'false'
scene.dataset.trailVisible = 'false'
scene.dataset.campPositioned = 'false'
const clip = `trail-clear-${crypto.randomUUID()}`
scene.innerHTML = `
    <svg class="trail-canvas" aria-hidden="true">
      <defs>
        <clipPath id="${clip}-clock"><path data-trail-clock-clear clip-rule="evenodd"/></clipPath>
        <clipPath id="${clip}"><path data-trail-clear clip-rule="evenodd"/></clipPath>
        <clipPath id="${clip}-backdrop"><path data-trail-backdrop-clear clip-rule="evenodd"/></clipPath>
        <path id="${clip}-hint-visitors" data-camp-hint-route="grove"/>
        <g data-trail-navigation></g>
      </defs>
      <g class="trail-clock-source" data-trail-model clip-path="url(#${clip}-clock)">
      ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
      <g data-pet-walk-paths></g>
      <path class="trail-path" data-pet-expedition/>
      </g>
      <g clip-path="url(#${clip})">
      <rect class="trail-paper" data-trail-layer="paper" width="100%" height="100%" fill="var(--light)"/>
      <g clip-path="url(#${clip}-clock)">
      <g class="trail-sky" aria-hidden="true"></g>
      <g data-trail-backdrop-container></g>
      </g>
      <g class="trail-ground-base" data-trail-layer="ground-base"></g>
      <g clip-path="url(#${clip}-clock)">
      <g class="trail-woodland" data-trail-depth="0.8"></g>
      <g class="trail-ground" data-trail-layer></g>
      <g class="trail-hinterland" data-trail-layer="hinterland"></g>
      <g class="trail-weather" aria-hidden="true"></g>
      <g class="trail-steps" data-trail-layer="steps"></g>
      <g class="trail-steps trail-camp-approach" data-trail-layer="camp-approach"></g>
      <g class="trail-hiker-tracks"></g>
      <g class="trail-reader-tracks"></g>
      <g class="trail-camp-tracks"></g>
      <g class="trail-visitor-tracks"></g>
      <g class="trail-ranger-tracks"></g>
      <g class="trail-park-patrol"></g>
      <g class="trail-landmarks"></g>
      <g class="trail-extension-forest"></g>
      <g class="trail-fauna"></g>
      <g class="trail-holidays" aria-hidden="true"></g>
      <g class="trail-wayfarers"></g>
      <g class="trail-community"></g>
      <g class="trail-visitors"></g>
      ${trailBackpacker()}
      <g class="trail-depth-objects"></g>
      </g>
      </g>
      <g class="trail-hint-fumes" clip-path="url(#${clip})">
        <g clip-path="url(#${clip}-clock)">
        <text dy="-9"><textPath href="#${clip}-hint-visitors" startOffset="50%" text-anchor="middle">meet at the fire</textPath></text>
        </g>
      </g>
    </svg>
    <div class="trail-visitor-labels"></div>
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
const styleReady = mountSceneSurface(host, scene, signal)
const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
const count = scene.querySelector<HTMLElement>('[data-trail-count]')
const retry = scene.querySelector<HTMLButtonElement>('[data-trail-retry]')
const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
const campToggle = scene.querySelector<HTMLButtonElement>('.trail-camp-toggle')
const campTools = scene.querySelector<HTMLElement>('.trail-camp-tools')
if (!svg || !count || !retry || !campsite || !campToggle || !campTools)
  throw new Error('Shared trail markup is incomplete')
const opening = mountSceneReveal(scene, host, signal)

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
let forestRendering: AbortController | null = null
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
const forestNodes = createForestSprites(document)
const woodlandNodes = createForestSprites(document)
let treePlan: TreePlacement[] | null = null
const forestClearings = (landscape: Landscape) =>
  campSites(landscape, sceneSeed, camps).flatMap(site => [
    { ...site, radius: 102 },
    ...site.tents.map(tent => ({ ...tent, radius: 38 })),
  ])
const forestSprites = (landscape: Landscape, nearFirst = false, settings = forest) => {
  const clearings = forestClearings(landscape)
  treePlan ??= planTrailTrees(landscape, sceneSeed, forestObstacles, clearings)
  return trailTreeSprites(
    landscape,
    sceneSeed,
    settings,
    forestObstacles,
    clearings,
    paintTree,
    nearFirst,
    treePlan,
  )
}
const finishForest = (sort = true) => {
  const landmarks = scene.querySelector('.trail-landmarks')
  const woodland = scene.querySelector('.trail-woodland')
  if (woodland && forestLayout) {
    const wanted = new Set<string>()
    for (const sprite of trailFoothillWoodland(
      forestLayout,
      sceneSeed,
      forest,
      forestObstacles,
      paintWoodland,
      treePlan ?? [],
    )) {
      wanted.add(sprite.id)
      woodlandNodes.update(sprite, woodland)
    }
    woodlandNodes.retain(wanted)
  }
  const foothillTrees = scene.querySelector('.trail-foothill-trees')
  if (foothillTrees && landmarks)
    foothillTrees.append(...landmarks.querySelectorAll('[data-tree-depth="foothill"]'))
  weather.refreshForest()
  crowns.refresh()
  holidays.refreshTrees()
  grove.refresh()
  pastureJourneys.refreshForest()
  if (sort) groundDepth.refresh()
}
const drawForest = async (sort = true, replan = false) => {
  forestRendering?.abort()
  if (host.hidden || !forestLayout || !scene.isConnected) return false
  const generation = new AbortController()
  forestRendering = generation
  if (replan) treePlan = null
  const landmarks = scene.querySelector('.trail-landmarks')
  if (!landmarks) return false
  const wanted = new Set<string>()
  let batch = performance.now()
  for (const sprite of forestSprites(forestLayout, false, { ...forest })) {
    wanted.add(sprite.id)
    forestNodes.update(sprite, landmarks)
    if (performance.now() - batch >= 4) {
      if (!(await yieldRender(generation.signal, true))) return false
      batch = performance.now()
    }
  }
  if (generation.signal.aborted || host.hidden) return false
  forestNodes.retain(wanted)
  finishForest(sort)
  worldPatches.refreshForest()
  forestRendering = null
  return true
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
  navigationRoutes = geometries.flatMap(({ id, length }) => {
    const route = navigableTrail(
      {
        id,
        length,
        readyAfter: trailOpeningDuration,
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
  groundDepth.restore(
    '.trail-map-reader, .trail-backpacker, [data-alpine-traveler], [data-park-ranger], [data-trail-visitor]',
  )
  const routes = journeyRoutes()
  journey.update(routes, forestLayout, sceneSeed, travelers.walkers > 0)
  readerJourneys.update(
    routes,
    forestLayout,
    sceneSeed,
    Math.max(0, travelers.walkers - 1),
    navigationBodies,
  )
  alpineJourneys.update(forestLayout, sceneSeed, travelers.climbers, travelers.alpinists)
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
    forestFrame = parent.requestAnimationFrame(async () => {
      forestFrame = 0
      if (redrawPendingScene()) return
      if (!(await drawForest())) return
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
    campFrame = parent.requestAnimationFrame(async () => {
      campFrame = 0
      if (redrawPendingScene()) {
        campRegenerate = false
        return
      }
      if (!forestLayout || host.hidden) return
      groundDepth.restore()
      if (campRegenerate && !(await drawForest(false, true))) return
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
    if (campPinned || campsite.matches(':focus-within')) return
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

let sceneSeed = Math.floor(Math.random() * 4294967296)
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
const crowns = mountCrownWind(scene, signal)
if (!mountSceneCompositor(scene, host, signal, opening)) opening.useFallback()
mountTrailBlazes(scene, signal)
mountLeafFall(scene, signal)
const groundDepth = mountGroundDepth(scene, host, signal)
const grove = mountSharedGrove(scene, host, signal, () => pastureJourneys.refreshForest())
const refreshStructures = () => {
  // Keep generation holders intact until every owner has replaced its previous sprites.
  if (scene.dataset.trailRendered === 'true' && forestLayout) {
    groundDepth.refresh()
  }
}
const community = mountCampCommunity(scene, host, signal, refreshStructures)
const parkPatrol = mountParkPatrol(scene, host, signal)
const visitorJourneys = mountVisitorJourneys(scene, host, signal, refreshStructures)
const worldPatches = mountWorldPatches(
  scene,
  host,
  signal,
  () => forest,
  () => forestObstacles,
  () => {
    weather.refreshForest()
    crowns.refresh()
    holidays.refreshTrees()
    grove.refresh()
    pastureJourneys.refreshForest()
    groundDepth.refresh()
  },
)
let retainedLayout: ReturnType<typeof trailLayout> = null
let retainedSeed = 0
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
  forestRendering?.abort()
  forestRendering = null
  rendering?.abort()
  rendering = null
}
signal.addEventListener('abort', cancelRendering, { once: true })
let generationSlice = 0
const yieldRender = (signal: AbortSignal, paint = false) => {
  if (!paint && performance.now() - generationSlice < 8) return Promise.resolve(!signal.aborted)
  return new Promise<boolean>(resolve => {
    if (signal.aborted) return resolve(false)
    let frame = 0
    const task = new MessageChannel()
    const finish = () => {
      task.port1.close()
      task.port2.close()
      signal.removeEventListener('abort', cancel)
      generationSlice = performance.now()
      resolve(!signal.aborted)
    }
    const cancel = () => {
      parent.cancelAnimationFrame(frame)
      finish()
    }
    signal.addEventListener('abort', cancel, { once: true })
    task.port1.onmessage = finish
    if (paint)
      frame = parent.requestAnimationFrame(() => {
        task.port2.postMessage(null)
      })
    else task.port2.postMessage(null)
  })
}
const drawScene = async () => {
  await styleReady
  if (host.hidden || !scene.isConnected) return
  const mode = scene.clientWidth < 700 ? 'narrow' : 'wide'
  const world =
    retainedSeed === sceneSeed && host.dataset.trailWorld === mode
      ? (retainedLayout ?? undefined)
      : undefined
  const layout = trailLayout(scene, sceneSeed, world)
  if (!layout) return
  const signature = `${sceneSeed}|${layout.width}|${layout.height}|${layout.bottom}|${JSON.stringify(layout.readingRoutes)}|${JSON.stringify(layout.routes)}`
  const clearance = `${layout.clear}|${layout.backdropClear}|${JSON.stringify(layout.clockExclusion)}`
  if (signature === layoutSignature && clearance === clearSignature) return
  const { width, height } = layout
  // Keep source coordinates at one CSS pixel while the host changes around the old scene.
  svg.style.width = `${width}px`
  svg.style.height = `${height}px`
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  svg.querySelector('[data-trail-clock-clear]')?.setAttribute('d', layout.clockClear)
  svg.querySelector('[data-trail-clear]')?.setAttribute('d', layout.clear)
  svg.querySelector('[data-trail-backdrop-clear]')?.setAttribute('d', layout.backdropClear)
  clearSignature = clearance
  if (signature === layoutSignature) return
  if (world) {
    layoutSignature = signature
    const readingPaths = svg.querySelector('[data-pet-walk-paths]')
    if (
      readingPaths &&
      JSON.stringify(layout.readingRoutes) !== JSON.stringify(world.readingRoutes)
    ) {
      readingPaths.innerHTML = layout.readingRoutes
        .map(
          (route, index) => `<path class="trail-path" data-pet-walk-path="${index}" d="${route}"/>`,
        )
        .join('')
      retainedLayout = layout
      const steps = scene.querySelector('.trail-steps')
      for (const old of steps?.querySelectorAll('[data-route="reading"]') ?? []) old.remove()
      for (const path of svg.querySelectorAll<SVGPathElement>('[data-pet-walk-path]')) {
        const geometry = svgPathGeometry(path)
        const marks: string[] = []
        for (let distance = 6; distance < geometry.getTotalLength() - 6; distance += 12) {
          const p = geometry.getPointAtLength(distance)
          if (p.y < layout.landscape.alpine.top) continue
          const next = geometry.getPointAtLength(distance + 1)
          marks.push(
            footprint(
              'blob',
              footprintTransform(p.x, p.y, Math.atan2(next.y - p.y, next.x - p.x), 1),
            ),
          )
        }
        steps?.insertAdjacentHTML('beforeend', `<g data-route="reading">${marks.join('')}</g>`)
      }
    }
    worldPatches.update(width, height)
    host.dataset.trailWorldHeight = String(
      Math.max(Number(host.dataset.trailWorldHeight) || 0, height),
    )
    const rocksX = Math.max(8, Math.min(layout.camp.x - 142, width - 64))
    forestControls.toggle.style.left = `${rocksX}px`
    const rocks = forestControls.toggle.querySelector<SVGGElement>(
      '[data-camp-solid="trunk-settings-rocks"]',
    )
    if (rocks) rocks.dataset.groundX = String(rocksX + 22)
    placeCampHints(scene)
    host.dispatchEvent(new Event('trail:ready'))
    return
  }
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
  if (scene.dataset.trailOpening !== 'complete') {
    for (const holder of svg.querySelectorAll(
      '[data-trail-backdrop-container], .trail-woodland, .trail-ground-base, .trail-ground, .trail-hinterland, .trail-steps, .trail-landmarks, .trail-foothill-trees, .trail-foothill-objects, .trail-depth-objects, .trail-fauna, .trail-wayfarers, .trail-community, .trail-visitors, .trail-park-patrol, .trail-hiker-tracks, .trail-reader-tracks, .trail-camp-tracks, .trail-visitor-tracks, .trail-ranger-tracks',
    ))
      holder.replaceChildren()
  }
  retainedLayout = null
  delete host.dataset.trailWorld
  delete host.dataset.trailWorldHeight
  worldPatches.reset(layout.landscape, sceneSeed, `${clip}-backdrop`)
  layoutSignature = signature
  weather.update(layout.landscape, sceneSeed)
  campsite.style.left = `${layout.camp.x}px`
  campsite.style.top = `${layout.camp.y}px`
  const rocksX = Math.max(8, layout.camp.x - 142)
  forestControls.toggle.style.left = `${rocksX}px`
  forestControls.toggle.style.top = `${layout.camp.y + 2}px`
  const rocks = forestControls.toggle.querySelector<SVGGElement>(
    '[data-camp-solid="trunk-settings-rocks"]',
  )
  if (rocks) {
    rocks.dataset.groundX = String(rocksX + 22)
    rocks.dataset.groundY = String(layout.camp.y + 46)
  }
  scene.style.setProperty('--trail-camp-arrival', '0ms')
  scene.dataset.campPositioned = 'true'
  scene.dataset.trailVisible = 'true'
  scene.dataset.trailAmbientReady = 'false'
  opening.update(layout.camp, layout)
  const kinds: FootprintKind[] = ['boot', 'paw', 'hoof', 'blob']
  const offset = Math.floor(trailRandom(sceneSeed ^ 0x2c681f4b)() * kinds.length)
  const main = svg.querySelector<SVGPathElement>('[data-trail-route="main"]')
  if (main) {
    main.setAttribute('d', layout.routes.main)
    const geometry = svgPathGeometry(main)
    const length = geometry.getTotalLength()
    const marks: string[] = []
    for (
      let distance = Math.max(6, Math.ceil((length - 120 - 6) / 12) * 12 + 6);
      distance < length - 6;
      distance += 12
    ) {
      const p = geometry.getPointAtLength(distance)
      const ahead = geometry.getPointAtLength(distance + 1)
      const step = Math.round((distance - 6) / 12)
      marks.push(
        footprint(
          kinds[offset],
          `${footprintTransform(p.x, p.y, Math.atan2(ahead.y - p.y, ahead.x - p.x), step % 2 ? 1 : -1)} scale(${trailGroundScale(p.y, layout.landscape).toFixed(3)})`,
        ),
      )
    }
    const approach = scene.querySelector('.trail-camp-approach')
    if (approach) approach.innerHTML = marks.join('')
  }
  const base = scene.querySelector('.trail-ground-base')
  if (base) base.innerHTML = trailTerraces(layout.landscape, sceneSeed)
  // Paint the camp before backdrop, route sampling, and forest generation occupy the main thread.
  if (!(await yieldRender(renderSignal, true))) return
  const backdrop = scene.querySelector('[data-trail-backdrop-container]')
  if (backdrop) {
    const entrances = trailIds.flatMap(id => {
      const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
      if (!path) return []
      path.setAttribute('d', layout.routes[id])
      const { x, y } = svgPathGeometry(path).getPointAtLength(0)
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
  geometries = []
  for (const id of trailIds) {
    const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
    if (!path) continue
    path.setAttribute('d', layout.routes[id])
    const geometry = svgPathGeometry(path)
    const length = geometry.getTotalLength()
    const samples = Array.from({ length: Math.ceil(length / 4) + 1 }, (_, index) => {
      const position = Math.min(1, (index * 4) / length)
      const p = geometry.getPointAtLength(position * length)
      return { position, x: p.x, y: p.y }
    })
    geometries.push({ id, length, samples })
    if (!(await yieldRender(renderSignal))) return
  }
  const steps = scene.querySelector('.trail-steps')
  if (steps) {
    steps.innerHTML = geometries
      .map(({ id, length }, index) => {
        const kind = kinds[(index + offset) % kinds.length]
        const marks = []
        for (let distance = 6, step = 0; distance < length - 6; distance += 12, step++) {
          if (id === 'main' && distance >= length - 120) continue
          const p = point(distance / length, id)
          // The sky above the ranges stays open; footprints start where the routes reach them.
          if (p.y < layout.landscape.alpine.top) continue
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
        return `<g data-route="${id}">${marks.join('')}</g>`
      })
      .join('')
    for (const readingPath of svg.querySelectorAll<SVGPathElement>('[data-pet-walk-path]')) {
      const geometry = svgPathGeometry(readingPath)
      const length = geometry.getTotalLength()
      const marks = []
      for (let distance = 6; distance < length - 6; distance += 12) {
        const p = geometry.getPointAtLength(distance)
        if (p.y < layout.landscape.alpine.top) continue
        const ahead = geometry.getPointAtLength(Math.min(length, distance + 1))
        marks.push(
          footprint(
            'blob',
            footprintTransform(p.x, p.y, Math.atan2(ahead.y - p.y, ahead.x - p.x), 1),
          ),
        )
      }
      steps.insertAdjacentHTML('beforeend', `<g data-route="reading">${marks.join('')}</g>`)
      if (!(await yieldRender(renderSignal))) return
    }
  }
  if (!(await yieldRender(renderSignal))) return
  const hinterland = scene.querySelector('.trail-hinterland')
  if (hinterland) hinterland.innerHTML = trailHinterland(width, layout.camp, sceneSeed)
  const ground = scene.querySelector('.trail-ground')
  if (ground)
    ground.innerHTML = trailTerrain(layout.landscape, sceneSeed, Object.values(layout.routes))
  if (!(await yieldRender(renderSignal))) return
  const expedition = petExpedition(scene, layout.landscape)
  const petPath = svg.querySelector<SVGPathElement>('[data-pet-expedition]')
  if (petPath && expedition) {
    petPath.setAttribute('d', expedition.d)
    petPath.dataset.petExpedition = String(sceneSeed)
    petPath.dataset.petScaleStops = JSON.stringify(expedition.scaleStops)
    petPath.dataset.petAlpineSpan = JSON.stringify(expedition.alpineSpan)
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
  treePlan = planTrailTrees(
    layout.landscape,
    sceneSeed,
    forestObstacles,
    forestClearings(layout.landscape),
  )
  const sprites = [...forestSprites(layout.landscape, true)]
  const visibleTrees = new Set(sprites.map(sprite => sprite.id))
  const painting = prepareTreeSprites(
    layout.landscape,
    treePlan.filter(placement => visibleTrees.has(placement.id)),
    forest,
    renderSignal,
  )
  community.update(layout.landscape, sceneSeed, camps, [])
  holidays.draw(layout.landscape, geometries, sceneSeed, camps)
  if (!(await yieldRender(renderSignal))) return
  forestNodes.clear()
  woodlandNodes.clear()
  const landmarks = scene.querySelector('.trail-landmarks')
  landmarks?.replaceChildren()
  const forestDraft = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  const prepared = await painting
  if (renderSignal.aborted) return
  let batch = performance.now()
  for (const sprite of sprites) {
    const markup = prepared?.get(sprite.id)
    forestNodes.update(markup ? { ...sprite, render: () => markup } : sprite, forestDraft)
    if (performance.now() - batch >= 16) {
      landmarks?.prepend(...forestDraft.childNodes)
      opening.generatedThrough(Math.max(sprite.bottom, layout.landscape.alpine.base + 168))
      if (!(await yieldRender(renderSignal))) return
      batch = performance.now()
    }
  }
  // Publish complete batches from camp to ridge, with farther trees behind earlier ones.
  landmarks?.prepend(...forestDraft.childNodes)
  finishForest(false)
  opening.generatedThrough(null)
  if (!(await yieldRender(renderSignal))) return
  const routes = journeyRoutes()
  community.update(layout.landscape, sceneSeed, camps, routes)
  if (!(await yieldRender(renderSignal))) return
  visitorJourneys.update(routes, layout.landscape, sceneSeed, camps)
  if (!(await yieldRender(renderSignal))) return
  parkPatrol.update(layout.landscape, sceneSeed, routes, travelers.rangers)
  if (!(await yieldRender(renderSignal))) return
  journey.update(routes, layout.landscape, sceneSeed, travelers.walkers > 0)
  if (!(await yieldRender(renderSignal))) return
  readerJourneys.update(
    routes,
    layout.landscape,
    sceneSeed,
    Math.max(0, travelers.walkers - 1),
    navigationBodies,
  )
  if (!(await yieldRender(renderSignal))) return
  pastureJourneys.update(layout.landscape, sceneSeed)
  if (!(await yieldRender(renderSignal))) return
  alpineJourneys.update(layout.landscape, sceneSeed, travelers.climbers, travelers.alpinists)
  if (!(await yieldRender(renderSignal))) return
  perspective.update()
  groundDepth.refresh()
  scene.dataset.trailRendered = 'true'
  scene.dataset.trailVisible = 'true'
  placeCampHints(scene)
  rendering = null
  retainedLayout = layout
  retainedSeed = sceneSeed
  host.dataset.trailWorld = mode
  host.dataset.trailWorldHeight = String(height)
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
    scene.dataset.trailRendered = 'false'
    scene.dataset.trailVisible = 'false'
    scene.dataset.campPositioned = 'false'
    opening.reset()
    campsite.dataset.revealed = String(
      parent.matchMedia('(prefers-reduced-motion: reduce)').matches,
    )
    const nextSeed = Math.floor(Math.random() * 4294967296)
    sceneSeed = nextSeed === sceneSeed ? (sceneSeed + 1) >>> 0 : nextSeed
    weather.resume()
    perspective.resume()
    startConnection()
  },
  { signal },
)
host.addEventListener(
  'trail:hide',
  () => {
    scene.dataset.trailVisible = 'false'
    opening.reset()
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
