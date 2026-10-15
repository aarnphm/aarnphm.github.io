import { playhtml } from 'playhtml'
import { svgPathGeometry } from '../../util/svg-geometry'
import { trailSceneHeight } from '../../util/trail-scene'
import { readTrunks, type CollisionActor } from './actor-spacing'
import { mountAlpineJourneys } from './alpine-journey'
import { trailBackpacker } from './backpacker'
import { mountTrailBlazes } from './blazes'
import { mountCampCommunity } from './camp-community'
import { campSites } from './camp-layout'
import { connectCampPresence } from './camp-presence'
import { trailCampsite } from './campsite'
import { mountCrownWind } from './crown-wind'
import { mountGroundDepth } from './depth-order'
import { footprint, footprintTransform, type FootprintKind } from './footprints'
import { createForestSprites } from './forest-sprites'
import { mountHikerJourney } from './hiker-journey'
import { trailHinterland } from './hinterland'
import { mountTrailHolidays } from './holidays'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { trailIds, trailLayout, type TrailId } from './layout'
import { mountLeafFall } from './leaf-fall'
import { trailMountainBackdrop } from './mountain-backdrop'
import { prepareNavigableTrails } from './navigation-workers'
import { navigableTrail, type NavigableTrail } from './obstacle-route'
import { mountParkPatrol } from './park-patrol'
import { trailPasture } from './pasture'
import { mountPastureWanderers } from './pasture-wander'
import { mountTrailPerspective } from './perspective'
import { petExpedition } from './pet-expedition'
import { trailRandom } from './random'
import { repoussoirSprites } from './repoussoir'
import { mountSceneCompositor } from './scene-compositor'
import { mountSceneLighting } from './scene-lighting'
import { mountSceneReveal } from './scene-reveal'
import { sceneSettings } from './scene-settings'
import { mountSceneSurface } from './scene-surface'
import {
  createTreePainter,
  planTrailTrees,
  type TreePlacement,
  trailFoothillWoodland,
  trailTreeSprites,
} from './scenery'
import { mountSeasonToggle } from './season-toggle'
import { mountSharedGrove, type Watering } from './shared-grove'
import { trailTerrain, trailTerraces } from './terrain'
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
// Use the viewport before the first layout; the homepage header is a narrow containing block.
if (typeof document.documentElement.moveBefore === 'function')
  document.documentElement.moveBefore(host, null)
const content = document.querySelector<HTMLElement>('#quartz-root > .page-body-grid')
if (content) host.style.height = `${trailSceneHeight(host, content)}px`
const scene = document.createElement('div')
scene.className = 'trail-scene'
scene.dataset.trailRendered = 'false'
scene.dataset.trailVisible = 'false'
scene.dataset.campPositioned = 'false'
const clip = `trail-clear-${crypto.randomUUID()}`
scene.innerHTML = `
    <svg class="trail-canvas" aria-hidden="true">
      <defs>
        <clipPath id="${clip}"><path data-trail-clear clip-rule="evenodd"/></clipPath>
        <clipPath id="${clip}-backdrop"><path data-trail-backdrop-clear clip-rule="evenodd"/></clipPath>
        <g data-trail-navigation></g>
      </defs>
      <g data-trail-model>
      ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
      <g data-pet-walk-paths></g>
      <path class="trail-path" data-pet-expedition/>
      </g>
      <rect class="trail-paper" data-trail-layer="paper" width="100%" height="100%" fill="var(--light)"/>
      <g class="trail-sky" aria-hidden="true"></g>
      <g clip-path="url(#${clip})">
      <g data-trail-backdrop-container></g>
      <g class="trail-far-gate-layer" data-trail-layer="far-gate" aria-hidden="true"></g>
      <g class="trail-ground-base" data-trail-layer="ground-base"></g>
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
    </svg>
    <div class="trail-visitor-labels"></div>
    <div class="trail-campsite" aria-hidden="true">
      <svg viewBox="0 0 128 96"><g class="trail-camp" transform="translate(64 64)">${trailCampsite(false)}</g></svg>
    </div>`
const styleReady = mountSceneSurface(host, scene, signal)
const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
const count = document.querySelector<HTMLElement>('[data-trail-connection]')
const retry = document.querySelector<HTMLButtonElement>('[data-trail-retry]')
const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
if (!svg || !count || !retry || !campsite) throw new Error('Shared trail markup is incomplete')
const opening = mountSceneReveal(scene, host, signal)

const { forest, animals, travelers, camps } = sceneSettings
let forestLayout: Landscape | undefined
let forestObstacles: Point[] = []
const paintTree = createTreePainter()
const paintWoodland = createTreePainter()
const forestNodes = createForestSprites(document)
const woodlandNodes = createForestSprites(document)
let treePlan: TreePlacement[] | null = null
// The forest is aged to one instant per layout, so re-planning for new water changes only the
// slots the water reached.
let forestNow = Date.now()
let groveWater = new Map<string, Watering[]>()
let plannedWater = groveWater
const forestClearings = (landscape: Landscape) =>
  campSites(landscape, sceneSeed, camps).flatMap(site => [
    { ...site, radius: 102 },
    ...site.tents.map(tent => ({ ...tent, radius: 38 })),
  ])
// The beeches keep clear of what a camp draws (hearth, people, tent), not the planner's room.
const beechClearings = (landscape: Landscape) =>
  campSites(landscape, sceneSeed, camps).flatMap(site => [
    { ...site, radius: 64 },
    ...site.tents.map(tent => ({ ...tent, radius: 52 })),
  ])
const forestSprites = (landscape: Landscape) => {
  const clearings = forestClearings(landscape)
  treePlan ??= planTrailTrees(
    landscape,
    sceneSeed,
    forestObstacles,
    clearings,
    forestNow,
    (plannedWater = groveWater),
  )
  return trailTreeSprites(
    landscape,
    sceneSeed,
    forest,
    forestObstacles,
    clearings,
    paintTree,
    true,
    treePlan,
  )
}
const drawWoodland = () => {
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
}
const refreshForest = () => {
  weather.refreshForest()
  crowns.refresh()
  holidays.refreshTrees()
  pastureJourneys.refreshForest()
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
const journeyObstacles = (landscape: Landscape) => {
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
  return [...solids.values()]
}
const journeyInputs = () =>
  geometries.map(({ id, length }) => ({
    id,
    length,
    readyAfter: 0,
    point: (distance: number) => point(Math.max(0, Math.min(1, distance / length)), id),
  }))
const journeyRoutes = (prepared?: { bodies: CollisionActor[]; routes: NavigableTrail[] }) => {
  if (!forestLayout) return []
  const landscape = forestLayout
  const bodies = prepared?.bodies ?? journeyObstacles(landscape)
  const signature = bodies.map(body => `${body.id}:${body.x}:${body.y}:${body.radius}`).join('|')
  if (navigationGeometry === geometries && navigationSignature === signature)
    return navigationRoutes
  navigationGeometry = geometries
  navigationSignature = signature
  navigationBodies = bodies
  navigationRoutes =
    prepared?.routes ??
    journeyInputs().flatMap(input => {
      const route = navigableTrail(input, landscape, bodies)
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
const sceneSeed = Math.floor(Math.random() * 4294967296)
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
mountSeasonToggle(scene, signal)
const crowns = mountCrownWind(scene, signal)
mountSceneCompositor(scene, host, signal, opening)
mountSceneLighting(scene, signal)
mountTrailBlazes(scene, signal)
mountLeafFall(scene, signal)
const groundDepth = mountGroundDepth(scene, host, signal)
const grove = mountSharedGrove(
  scene,
  host,
  signal,
  () => pastureJourneys.refreshForest(),
  water => {
    groveWater = water
    void regrowForest()
  },
)
// Water lengthens a tree's life and speeds its growth. Workers repaint only the slots whose tree
// changed; a newer watering supersedes a repaint still in flight.
let regrowing: AbortController | null = null
const regrowForest = async () => {
  const landmarks = scene.querySelector('.trail-landmarks')
  if (!forestLayout || !treePlan || !landmarks || scene.dataset.trailRendered !== 'true') return
  if (plannedWater === groveWater) return
  regrowing?.abort()
  const controller = new AbortController()
  regrowing = controller
  const layout = forestLayout
  const base = treePlan
  const previous = new Map(base.map(placement => [placement.id, placement]))
  const next = planTrailTrees(
    layout,
    sceneSeed,
    forestObstacles,
    forestClearings(layout),
    forestNow,
    (plannedWater = groveWater),
  )
  const changed = next.filter(placement => {
    const before = previous.get(placement.id)
    return (
      !before ||
      before.kind !== placement.kind ||
      before.treeSeed !== placement.treeSeed ||
      before.height !== placement.height ||
      before.log !== placement.log
    )
  })
  if (!changed.length) return
  const painting = prepareTreeSprites(layout, changed, forest, controller.signal)
  const markups = new Map<string, string | null>()
  for (const [id, markup] of painting) markups.set(id, await markup)
  // A new layout repaints the whole forest with this water already.
  if (controller.signal.aborted || signal.aborted || treePlan !== base) return
  treePlan = next
  for (const placement of changed)
    if (placement.log === null) forestNodes.remove(`deadfall-${placement.id}`)
  for (const sprite of forestSprites(layout)) {
    const markup = markups.get(sprite.id)
    forestNodes.update(markup ? { ...sprite, render: () => markup } : sprite, landmarks)
  }
  groundDepth.refresh()
  grove.refresh()
  refreshForest()
}
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
const generationScheduler =
  typeof TaskController === 'function' && typeof parent.scheduler?.postTask === 'function'
    ? parent.scheduler
    : null
let rendering: AbortController | null = null
const cancelRendering = () => {
  rendering?.abort()
  rendering = null
}
signal.addEventListener('abort', cancelRendering, { once: true })
const prioritizeRendering = (priority: TaskPriority) => {
  if (typeof TaskController === 'function' && rendering instanceof TaskController)
    rendering.setPriority(priority)
}
host.addEventListener('trail:open', () => prioritizeRendering('user-visible'), { signal })
host.addEventListener('trail:hide', () => prioritizeRendering('background'), { signal })
let generationSlice = 0
const yieldRender = (signal: AbortSignal, paint = false) => {
  if (!paint && performance.now() - generationSlice < 8) return Promise.resolve(!signal.aborted)
  if (!paint && generationScheduler)
    return generationScheduler
      .postTask(
        () => {
          generationSlice = performance.now()
          return !signal.aborted
        },
        { signal },
      )
      .catch(error => {
        if (signal.aborted) return false
        throw error
      })
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
  if (document.hidden || host.hidden || !scene.isConnected) return
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
  const cssWidth = `${width}px`
  const cssHeight = `${height}px`
  const viewBox = `0 0 ${width} ${height}`
  if (svg.style.width !== cssWidth) svg.style.width = cssWidth
  if (svg.style.height !== cssHeight) svg.style.height = cssHeight
  if (svg.getAttribute('viewBox') !== viewBox) svg.setAttribute('viewBox', viewBox)
  const setPath = (selector: string, d: string) => {
    const node = svg.querySelector(selector)
    if (node && node.getAttribute('d') !== d) node.setAttribute('d', d)
  }
  setPath('[data-trail-clear]', layout.clear)
  setPath('[data-trail-backdrop-clear]', layout.backdropClear)
  opening.update(layout)
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
    host.dispatchEvent(new Event('trail:ready'))
    return
  }
  cancelRendering()
  // Closed scenery yields behind input; opening promotes the pending generation task.
  rendering = generationScheduler
    ? new TaskController({ priority: host.hidden ? 'background' : 'user-visible' })
    : new AbortController()
  const renderSignal = rendering.signal
  scene.dataset.trailRendered = 'false'
  scene.dataset.trailRasterReady = 'false'
  // Geometry sampling uses isolated paths. Keep the growing model out of layout until its
  // complete forest needs measurement, instead of laying out every generation batch.
  if (scene.dataset.trailOpening !== 'complete') svg.style.display = 'none'
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
  scene.dataset.campPositioned = 'true'
  scene.dataset.trailVisible = String(!host.hidden)
  geometries = []
  for (const id of trailIds) {
    const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
    if (!path) continue
    path.setAttribute('d', layout.routes[id])
    const geometry = svgPathGeometry(path)
    const length = geometry.getTotalLength()
    const samples: TrailSample[] = []
    for (let index = 0; index <= Math.ceil(length / 4); index++) {
      const position = Math.min(1, (index * 4) / length)
      const p = geometry.getPointAtLength(position * length)
      samples.push({ position, x: p.x, y: p.y })
      if (index % 16 === 0 && !(await yieldRender(renderSignal))) return
    }
    geometries.push({ id, length, samples })
    if (!(await yieldRender(renderSignal))) return
  }
  const pasture = trailPasture(layout.landscape, geometries, sceneSeed, animals)
  const fauna = scene.querySelector('.trail-fauna')
  if (fauna) fauna.innerHTML = pasture.markup
  forestLayout = layout.landscape
  forestObstacles = [
    ...geometries.flatMap(route => route.samples.filter((_, index) => index % 4 === 0)),
    ...pasture.sites,
  ]
  forestNow = Date.now()
  treePlan = planTrailTrees(
    layout.landscape,
    sceneSeed,
    forestObstacles,
    forestClearings(layout.landscape),
    forestNow,
    (plannedWater = groveWater),
  )
  const sprites = [
    ...repoussoirSprites(layout.landscape, sceneSeed, beechClearings(layout.landscape)),
    ...forestSprites(layout.landscape),
  ]
  const visibleTrees = new Set(sprites.map(sprite => sprite.id))
  const painting = prepareTreeSprites(
    layout.landscape,
    treePlan.filter(placement => visibleTrees.has(placement.id)).sort((a, b) => b.y - a.y),
    forest,
    renderSignal,
  )
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
  const hinterland = scene.querySelector('.trail-hinterland')
  if (hinterland) hinterland.innerHTML = trailHinterland(width, layout.camp, sceneSeed)
  const ground = scene.querySelector('.trail-ground')
  if (ground)
    ground.innerHTML = trailTerrain(layout.landscape, sceneSeed, Object.values(layout.routes))
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
        if (marks.length % 16 === 0 && !(await yieldRender(renderSignal))) return
      }
      steps.insertAdjacentHTML('beforeend', `<g data-route="reading">${marks.join('')}</g>`)
      if (!(await yieldRender(renderSignal))) return
    }
  }
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
  community.update(layout.landscape, sceneSeed, camps, [])
  holidays.draw(layout.landscape, geometries, sceneSeed, camps)
  if (!(await yieldRender(renderSignal))) return
  forestNodes.clear()
  woodlandNodes.clear()
  const landmarks = scene.querySelector('.trail-landmarks')
  landmarks?.replaceChildren()
  const forestDraft = document.createElementNS('http://www.w3.org/2000/svg', 'g')
  let batch = performance.now()
  for (const sprite of sprites) {
    const markup = await painting.get(sprite.id)
    if (renderSignal.aborted) return
    forestNodes.update(markup ? { ...sprite, render: () => markup } : sprite, forestDraft)
    if (performance.now() - batch >= 8) {
      landmarks?.prepend(...forestDraft.childNodes)
      if (!(await yieldRender(renderSignal))) return
      batch = performance.now()
    }
  }
  // Publish complete batches from camp to ridge, with farther trees behind earlier ones.
  landmarks?.prepend(...forestDraft.childNodes)
  drawWoodland()
  groundDepth.refresh()
  svg.style.removeProperty('display')
  grove.refresh()
  const bodies = journeyObstacles(layout.landscape)
  const routing = prepareNavigableTrails(journeyInputs(), layout.landscape, bodies, renderSignal)
  // Plan from the complete, grown forest while climate geometry and static textures prepare.
  refreshForest()
  scene.dataset.trailRasterReady = 'true'
  host.dispatchEvent(new Event('trail:prepare'))
  const preparedRoutes = await routing
  if (renderSignal.aborted) return
  const routes = journeyRoutes({ bodies, routes: preparedRoutes })
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
  // Water that arrived while the forest was painting.
  void regrowForest()
  scene.dataset.trailVisible = String(!host.hidden)
  rendering = null
  retainedLayout = layout
  retainedSeed = sceneSeed
  host.dataset.trailWorld = mode
  host.dataset.trailWorldHeight = String(height)
  host.dispatchEvent(new Event('trail:ready'))
}
host.addEventListener(
  'trail:open',
  () => {
    void drawScene()
  },
  { signal },
)
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
  count.textContent = 'connection unavailable'
  retry.hidden = false
  scene.dataset.trailState = 'unavailable'
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
  parent.clearTimeout(timeout)
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
  connectCampPresence(
    scene,
    host,
    signal,
    Math.max(0, Math.min(camps.sites - 1, Math.floor(sceneSettings.destination))),
    visitorJourneys.setVisitors,
  )
  count.textContent = 'connected'
  retry.hidden = true
  scene.dataset.trailState = 'ready'
}
const startConnection = () => {
  if (connectionStarted || signal.aborted) return
  connectionStarted = true
  scene.dataset.trailState = 'connecting'
  count.textContent = 'connecting'
  retry.hidden = true
  timeout = parent.setTimeout(unavailable, 12000)
  void connect().catch(unavailable)
}
host.addEventListener(
  'trail:show',
  () => {
    scene.dataset.trailVisible = 'true'
    weather.resume()
    perspective.resume()
    groundDepth.resume()
    community.resume()
    alpineJourneys.resume()
    host.dispatchEvent(new Event('trail:ready'))
    startConnection()
  },
  { signal },
)
const pauseScene = () => {
  scene.dataset.trailVisible = 'false'
  journey.pause()
  readerJourneys.pause()
  pastureJourneys.pause()
  alpineJourneys.pause()
  perspective.pause()
  groundDepth.pause()
  weather.pause()
  community.pause()
  visitorJourneys.pause()
  parkPatrol.pause()
}
host.addEventListener('trail:close', pauseScene, { signal })
host.addEventListener('trail:hide', pauseScene, { signal })
startConnection()
