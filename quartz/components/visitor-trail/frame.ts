import { playhtml } from 'playhtml'
import { trailSceneSeed } from '../../util/trail-preference'
import { mountAlpineJourneys } from './alpine-journey'
import { mountTrailBlazes } from './blazes'
import { mountCampCommunity } from './camp-community'
import { campSites } from './camp-layout'
import { connectCampPresence } from './camp-presence'
import { mountCrownWind } from './crown-wind'
import { mountGroundDepth } from './depth-order'
import { createForestSprites } from './forest-sprites'
import { createFrameGeneration, type FrameModel } from './frame/generation'
import { createFrameSurface } from './frame/markup'
import { createFrameNavigation } from './frame/navigation'
import { mountFrog } from './frog'
import { mountHikerJourney } from './hiker-journey'
import { mountTrailHolidays } from './holidays'
import { mountJay } from './jay'
import { creekDistance, pondDistance, type Landscape, type Point } from './landscape'
import { mountLeafFall } from './leaf-fall'
import { mountMossCover } from './moss-cover'
import { createMotionTuning } from './motion-tuning'
import { createTrailNavigation } from './navigation-workers'
import { mountParkPatrol } from './park-patrol'
import { trailPasture } from './pasture'
import { mountPastureWanderers } from './pasture-wander'
import { mountTrailPerspective } from './perspective'
import { mountRiverBear } from './river-bear'
import { mountSceneCompositor } from './scene-compositor'
import { mountSceneLighting } from './scene-lighting'
import { sceneMotionReady } from './scene-reveal'
import { sceneSettings } from './scene-settings'
import { mountSceneTuning } from './scene-tuning-runtime'
import {
  createTreePainter,
  planTrailTrees,
  trailFoothillWoodland,
  trailTreeSprites,
} from './scenery'
import { mountSeasonToggle } from './season-toggle'
import { mountSharedGrove, type Watering } from './shared-grove'
import { defaultSun, trailSunlight, type SunCast } from './sunlight'
import { mountTrailAthletes } from './trail-athletes'
import { prepareTreeSprites } from './tree-workers'
import { mountVisitorJourneys } from './visitor-journey'
import { mountMapReaderJourneys } from './wayfarer-journey'
import { mountTrailWeather } from './weather'
import { mountWorldPatches } from './world-patches'

const document = parent.document
const host = window.frameElement?.parentElement
if (!host?.matches('.homepage-trail-scene') || parent === window)
  throw new Error('The trail needs its homepage host')
const controller = new AbortController()
const { signal } = controller
const surface = createFrameSurface(document, host, signal)
const { scene, count, retry, opening } = surface
const settings = structuredClone(sceneSettings)
const { forest, animals, camps } = settings
// The forest is aged to one instant per layout; watering changes only affected slots.
const initialWater = new Map<string, Watering[]>()
const model: FrameModel = {
  forestLayout: undefined,
  forestObstacles: [],
  treePlan: null,
  forestNow: Date.now(),
  groveWater: initialWater,
  plannedWater: initialWater,
}
// The sun the canopy shade is cast from; the weather's sun clock updates it through `trail:sun`.
let sunCast: SunCast = defaultSun
const paintSunlight = () => {
  const sunlight = scene.querySelector('.trail-sunlight:not(.trail-sunlight-canopy)')
  const canopyLight = scene.querySelector('.trail-sunlight-canopy')
  if (!model.forestLayout || !sunlight || !canopyLight) return
  const light = trailSunlight(model.forestLayout, sceneSeed, sunCast)
  sunlight.innerHTML = light.ground
  canopyLight.innerHTML = light.canopy
}
host.addEventListener(
  'trail:sun',
  event => {
    const next = (event as CustomEvent<SunCast>).detail
    const moved =
      Math.abs(next.dx - sunCast.dx) > 0.02 ||
      Math.abs(next.dy - sunCast.dy) > 0.02 ||
      Math.abs(next.low - sunCast.low) > 0.03
    sunCast = next
    if (moved) paintSunlight()
  },
  { signal },
)
const paintTree = createTreePainter()
const paintWoodland = createTreePainter()
const forestNodes = createForestSprites(document)
const woodlandNodes = createForestSprites(document)
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
  model.treePlan ??= planTrailTrees(
    landscape,
    sceneSeed,
    model.forestObstacles,
    clearings,
    model.forestNow,
    (model.plannedWater = model.groveWater),
  )
  return trailTreeSprites(
    landscape,
    sceneSeed,
    forest,
    model.forestObstacles,
    clearings,
    paintTree,
    true,
    model.treePlan,
  )
}
const drawWoodland = () => {
  const landmarks = scene.querySelector('.trail-landmarks')
  const woodland = scene.querySelector('.trail-woodland')
  if (woodland && model.forestLayout) {
    const wanted = new Set<string>()
    for (const sprite of trailFoothillWoodland(
      model.forestLayout,
      sceneSeed,
      forest,
      model.forestObstacles,
      paintWoodland,
      model.treePlan ?? [],
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
// The jay's larder keeps off the pond and the creek, out of the tents and inside the scene.
const larderClear = (point: Point) => {
  const landscape = model.forestLayout
  if (!landscape) return false
  if (point.x < 12 || point.x > landscape.width - 12 || point.y > landscape.height - 12)
    return false
  if (pondDistance(point, landscape.pond) < 1.35 || creekDistance(point, landscape) < 12)
    return false
  return campSites(landscape, sceneSeed, camps).every(
    site =>
      Math.hypot(site.x - point.x, site.y - point.y) > 48 &&
      site.tents.every(tent => Math.hypot(tent.x - point.x, tent.y - point.y) > 30),
  )
}
const forestRefreshers = [
  () => jay.refresh(model.treePlan ?? [], larderClear),
  () => model.forestLayout && frog.refresh(model.forestLayout),
  () => model.forestLayout && riverBear.refresh(model.forestLayout),
  () => crowns.refresh(),
  () => holidays.refreshTrees(),
  () => pastureJourneys.refreshForest(),
]
const refreshForest = () => {
  for (const refresh of forestRefreshers) refresh()
}
const holidays = mountTrailHolidays(scene, signal, () => {
  if (!model.forestLayout || scene.dataset.trailRendered !== 'true') return
  holidays.draw(model.forestLayout, routes.geometries, sceneSeed, camps)
  pastureJourneys.update(model.forestLayout, sceneSeed)
  groundDepth.refresh()
})
const sceneSeed = trailSceneSeed()
const navigation = createTrailNavigation(signal)
const routes = createFrameNavigation(scene, sceneSeed, camps, () => model.forestLayout, navigation)
const journey = mountHikerJourney(scene, host, signal, navigation)
const readerJourneys = mountMapReaderJourneys(scene, host, signal, navigation)
const pastureJourneys = mountPastureWanderers(scene, host, signal)
const alpineJourneys = mountAlpineJourneys(scene, host, signal)
const perspective = mountTrailPerspective(scene, host, signal)
const weather = mountTrailWeather(scene, host, signal)
const motionTuning = createMotionTuning(scene)
const crowns = mountCrownWind(scene, signal, motionTuning.settings)
mountSceneCompositor(scene, host, signal, opening)
mountSceneLighting(scene, signal)
mountTrailBlazes(scene, signal)
mountLeafFall(scene, signal, motionTuning.settings)
const jay = mountJay(scene, host, signal)
const frog = mountFrog(scene, host, signal)
const riverBear = mountRiverBear(scene, host, signal)
const groundDepth = mountGroundDepth(scene, host, signal)
const moss = mountMossCover(scene, host, signal)
const grove = mountSharedGrove(
  scene,
  host,
  signal,
  () => pastureJourneys.refreshForest(),
  water => {
    model.groveWater = water
    void regrowForest()
  },
)
// Water lengthens a tree's life and speeds its growth. Workers repaint only the slots whose tree
// changed; a newer watering supersedes a repaint still in flight.
let regrowing: AbortController | null = null
const regrowForest = async () => {
  const landmarks = scene.querySelector('.trail-landmarks')
  if (
    !model.forestLayout ||
    !model.treePlan ||
    !landmarks ||
    scene.dataset.trailRendered !== 'true' ||
    tuning.busy() ||
    !sceneMotionReady(scene)
  )
    return
  if (model.plannedWater === model.groveWater) return
  regrowing?.abort()
  const controller = new AbortController()
  regrowing = controller
  const layout = model.forestLayout
  const base = model.treePlan
  const water = model.groveWater
  const previous = new Map(base.map(placement => [placement.id, placement]))
  const next = planTrailTrees(
    layout,
    sceneSeed,
    model.forestObstacles,
    forestClearings(layout),
    model.forestNow,
    water,
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
  if (!changed.length) {
    model.plannedWater = water
    return
  }
  const painting = prepareTreeSprites(layout, changed, forest, controller.signal)
  const markups = new Map<string, string | null>()
  for (const [id, markup] of painting) markups.set(id, await markup)
  // A new layout repaints the whole forest with this water already.
  if (
    controller.signal.aborted ||
    signal.aborted ||
    model.treePlan !== base ||
    !sceneMotionReady(scene)
  )
    return
  model.plannedWater = water
  model.treePlan = next
  for (const placement of changed)
    if (placement.log === null) forestNodes.remove(`deadfall-${placement.id}`)
  for (const sprite of forestSprites(layout)) {
    const markup = markups.get(sprite.id)
    forestNodes.update(markup ? { ...sprite, render: () => markup } : sprite, landmarks)
  }
  groundDepth.refresh()
  grove.refresh()
  weather.refreshForest()
  refreshForest()
}
for (const event of ['trail:prepared', 'trail:reveal-end', 'trail:closed'])
  host.addEventListener(event, () => void regrowForest(), { signal })
const refreshStructures = () => {
  // Keep generation holders intact until every owner has replaced its previous sprites.
  if (scene.dataset.trailRendered === 'true' && model.forestLayout) {
    groundDepth.refresh()
  }
}
const community = mountCampCommunity(scene, host, signal, refreshStructures, navigation)
const parkPatrol = mountParkPatrol(scene, host, signal, navigation)
const athletes = mountTrailAthletes(scene, host, signal)
const visitorJourneys = mountVisitorJourneys(scene, host, signal, refreshStructures, navigation)
const worldPatches = mountWorldPatches(
  scene,
  host,
  signal,
  () => forest,
  () => model.forestObstacles,
  () => {
    weather.refreshForest()
    crowns.refresh()
    holidays.refreshTrees()
    grove.refresh()
    pastureJourneys.refreshForest()
    groundDepth.refresh()
  },
)
const refreshForestObstacles = (sites?: Point[]) => {
  if (!model.forestLayout) return
  model.forestObstacles = [
    ...routes.geometries.flatMap(route => route.samples.filter((_, index) => index % 4 === 0)),
    ...(sites ?? trailPasture(model.forestLayout, routes.geometries, sceneSeed, animals).sites),
  ]
}
const generation = createFrameGeneration({
  document,
  host,
  signal,
  surface,
  model,
  settings,
  sceneSeed,
  routes,
  forestNodes,
  woodlandNodes,
  forestClearings,
  beechClearings,
  forestSprites,
  drawWoodland,
  paintSunlight,
  refreshForest,
  regrowForest,
  refreshForestObstacles,
  beforeGenerate() {
    tuning.cancel()
    regrowing?.abort()
  },
  journey,
  readerJourneys,
  pastureJourneys,
  alpineJourneys,
  community,
  visitorJourneys,
  parkPatrol,
  athletes,
  groundDepth,
  moss,
  weather,
  worldPatches,
  holidays,
  grove,
  perspective,
})
const tuning = mountSceneTuning({
  scene,
  host,
  signal,
  settings,
  seed: sceneSeed,
  read(refreshObstacles) {
    if (refreshObstacles) refreshForestObstacles()
    return model.forestLayout && model.treePlan
      ? {
          landscape: model.forestLayout,
          plan: model.treePlan,
          obstacles: model.forestObstacles,
          clearings: forestClearings(model.forestLayout),
          foregroundClearings: beechClearings(model.forestLayout),
          geometries: routes.geometries,
        }
      : null
  },
  replan() {
    if (!model.forestLayout) return
    regrowing?.abort()
    refreshForestObstacles()
    model.treePlan = planTrailTrees(
      model.forestLayout,
      sceneSeed,
      model.forestObstacles,
      forestClearings(model.forestLayout),
      model.forestNow,
      (model.plannedWater = model.groveWater),
    )
  },
  refreshForest: [
    () => groundDepth.refresh(),
    () => grove.refresh(),
    () => weather.refreshForest(),
    ...forestRefreshers,
  ],
  async routes(request) {
    if (!model.forestLayout) return { routes: [], solids: [] }
    const prepared = await routes.prepare(model.forestLayout, request)
    if (request.aborted) return { routes: [], solids: [] }
    return { routes: routes.commit(prepared), solids: routes.bodies }
  },
  forestNodes,
  woodlandNodes,
  paintTree,
  paintWoodland,
  groundDepth,
  community,
  visitorJourneys,
  journey,
  readerJourneys,
  alpineJourneys,
  parkPatrol,
  pastureJourneys,
  holidays,
  worldPatches,
})
scene.addEventListener('trail:tuning-commit', () => void regrowForest(), { signal })
mountSeasonToggle(scene, signal, {
  settings,
  change(group) {
    regrowing?.abort()
    tuning.change(group)
  },
  motion: motionTuning.settings,
  motionChange: motionTuning.changed,
  destinationChange(destination) {
    settings.destination = destination
    host.dispatchEvent(new CustomEvent('trail:destination', { detail: destination }))
  },
  pack: () => community.packUp(settings.destination),
})
host.addEventListener(
  'trail:open',
  () => {
    void generation.draw()
  },
  { signal },
)
host.addEventListener(
  'trail:reflow',
  () => {
    void generation.draw()
  },
  { signal },
)
void generation.draw()
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
    Math.max(0, Math.min(camps.sites - 1, Math.floor(settings.destination))),
    visitorJourneys.setVisitors,
    () => camps.sites,
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
    moss.resume()
    host.dispatchEvent(new Event('trail:ready'))
    startConnection()
  },
  { signal },
)
const pauseScene = () => {
  generation.suspend()
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
  athletes.pause()
  moss.pause()
}
host.addEventListener('trail:close', pauseScene, { signal })
host.addEventListener('trail:hide', pauseScene, { signal })
startConnection()
