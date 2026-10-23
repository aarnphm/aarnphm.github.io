import type { mountAlpineJourneys } from '../alpine-journey'
import type { mountCampCommunity } from '../camp-community'
import type { mountGroundDepth } from '../depth-order'
import type { createForestSprites } from '../forest-sprites'
import type { mountHikerJourney } from '../hiker-journey'
import type { mountTrailHolidays } from '../holidays'
import type { Landscape, Point } from '../landscape'
import type { mountMossCover } from '../moss-cover'
import type { mountParkPatrol } from '../park-patrol'
import type { mountPastureWanderers } from '../pasture-wander'
import type { mountTrailPerspective } from '../perspective'
import type { SceneSettings } from '../scene-settings'
import type { TreePlacement } from '../scenery'
import type { Watering } from '../shared-grove'
import type { mountSharedGrove } from '../shared-grove'
import type { mountTrailAthletes } from '../trail-athletes'
import type { mountVisitorJourneys } from '../visitor-journey'
import type { mountMapReaderJourneys } from '../wayfarer-journey'
import type { mountTrailWeather } from '../weather'
import type { mountWorldPatches } from '../world-patches'
import type { createFrameSurface } from './markup'
import type { createFrameNavigation } from './navigation'
import { svgPathGeometry } from '../../../util/svg-geometry'
import { holdClockGeometry } from '../clock-clearance'
import { footprint, footprintTransform } from '../footprints'
import { trailIds, trailLayout } from '../layout'
import { trailPasture } from '../pasture'
import { petExpedition } from '../pet-expedition'
import { planTrailTrees, trailTreeSprites } from '../scenery'
import { prepareScenery } from '../scenery-workers'
import { prepareSceneTrees } from '../tree-workers'

export type FrameModel = {
  forestLayout: Landscape | undefined
  forestObstacles: Point[]
  treePlan: TreePlacement[] | null
  forestNow: number
  groveWater: Map<string, Watering[]>
  plannedWater: Map<string, Watering[]>
}
type Context = {
  document: Document
  host: HTMLElement
  signal: AbortSignal
  surface: ReturnType<typeof createFrameSurface>
  model: FrameModel
  settings: SceneSettings
  sceneSeed: number
  routes: ReturnType<typeof createFrameNavigation>
  forestNodes: ReturnType<typeof createForestSprites>
  woodlandNodes: ReturnType<typeof createForestSprites>
  forestClearings: (landscape: Landscape) => (Point & { radius: number })[]
  beechClearings: (landscape: Landscape) => (Point & { radius: number })[]
  forestSprites: (landscape: Landscape) => ReturnType<typeof trailTreeSprites>
  drawWoodland: () => void
  paintSunlight: () => void
  refreshForest: () => void
  regrowForest: () => Promise<void>
  refreshForestObstacles: (sites?: Point[]) => void
  beforeGenerate: () => void
  journey: ReturnType<typeof mountHikerJourney>
  readerJourneys: ReturnType<typeof mountMapReaderJourneys>
  pastureJourneys: ReturnType<typeof mountPastureWanderers>
  alpineJourneys: ReturnType<typeof mountAlpineJourneys>
  community: ReturnType<typeof mountCampCommunity>
  visitorJourneys: ReturnType<typeof mountVisitorJourneys>
  parkPatrol: ReturnType<typeof mountParkPatrol>
  athletes: ReturnType<typeof mountTrailAthletes>
  groundDepth: ReturnType<typeof mountGroundDepth>
  moss: ReturnType<typeof mountMossCover>
  weather: ReturnType<typeof mountTrailWeather>
  worldPatches: ReturnType<typeof mountWorldPatches>
  holidays: ReturnType<typeof mountTrailHolidays>
  grove: ReturnType<typeof mountSharedGrove>
  perspective: ReturnType<typeof mountTrailPerspective>
}

export function createFrameGeneration(context: Context) {
  const {
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
    beforeGenerate,
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
  } = context
  const { scene, clip, styleReady, svg, campsite, opening } = surface
  const { forest, animals, travelers, camps } = settings
  let layoutSignature = ''
  let clearSignature = ''
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
  const yieldRender = (signal: AbortSignal) => {
    if (performance.now() - generationSlice < (host.hidden ? 8 : 16))
      return Promise.resolve(!signal.aborted)
    if (generationScheduler) {
      const continuation =
        !host.hidden && typeof generationScheduler.yield === 'function'
          ? generationScheduler.yield()
          : generationScheduler.postTask(() => undefined, { signal })
      return continuation
        .then(() => {
          generationSlice = performance.now()
          return !signal.aborted
        })
        .catch(error => {
          if (signal.aborted) return false
          throw error
        })
    }
    return new Promise<boolean>(resolve => {
      if (signal.aborted) return resolve(false)
      const task = new MessageChannel()
      const finish = () => {
        task.port1.close()
        task.port2.close()
        signal.removeEventListener('abort', cancel)
        generationSlice = performance.now()
        resolve(!signal.aborted)
      }
      const cancel = finish
      signal.addEventListener('abort', cancel, { once: true })
      task.port1.onmessage = finish
      task.port2.postMessage(null)
    })
  }
  const drawScene = async () => {
    await styleReady
    if (signal.aborted || document.hidden || host.hidden || !scene.isConnected) return
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
    if (world && layout.landscape === world.landscape) {
      layoutSignature = signature
      const readingPaths = svg.querySelector('[data-pet-walk-paths]')
      if (
        readingPaths &&
        JSON.stringify(layout.readingRoutes) !== JSON.stringify(world.readingRoutes)
      ) {
        readingPaths.innerHTML = layout.readingRoutes
          .map(
            (route, index) =>
              `<path class="trail-path" data-pet-walk-path="${index}" d="${route}"/>`,
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
    beforeGenerate()
    cancelRendering()
    // Closed scenery yields behind input; opening promotes the pending generation task.
    rendering = generationScheduler
      ? new TaskController({ priority: host.hidden ? 'background' : 'user-visible' })
      : new AbortController()
    const renderSignal = rendering.signal
    const releaseClock = holdClockGeometry(svg, renderSignal)
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
    athletes.pause()
    groundDepth.restore()
    if (scene.dataset.trailOpening !== 'complete') {
      for (const holder of svg.querySelectorAll(
        '[data-trail-backdrop-container], .trail-woodland, .trail-ground-base, .trail-ground, .trail-hinterland, .trail-steps, .trail-landmarks, .trail-foothill-trees, .trail-foothill-objects, .trail-depth-objects, .trail-fauna',
      ))
        holder.replaceChildren()
    }
    moss.clear()
    retainedLayout = null
    delete host.dataset.trailWorld
    delete host.dataset.trailWorldHeight
    worldPatches.reset(layout.landscape, sceneSeed, `${clip}-backdrop`)
    layoutSignature = signature
    if (host.hidden && !(await yieldRender(renderSignal))) return
    weather.update(layout.landscape, sceneSeed)
    campsite.style.left = `${layout.camp.x}px`
    campsite.style.top = `${layout.camp.y}px`
    scene.dataset.campPositioned = 'true'
    scene.dataset.trailVisible = String(!host.hidden)
    const entrances = trailIds.flatMap(id => {
      const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
      if (!path) return []
      path.setAttribute('d', layout.routes[id])
      const { x, y } = svgPathGeometry(path).getPointAtLength(0)
      return [{ x, y }]
    })
    // Terraces, terrain and ranges paint in a worker while routes are sampled here.
    const scenery = prepareScenery(
      {
        landscape: layout.landscape,
        seed: sceneSeed,
        width,
        camp: layout.camp,
        routes: Object.values(layout.routes),
        backdropClip: `${clip}-backdrop`,
        entrances,
      },
      renderSignal,
    )
    if (!(await routes.sample(layout, renderSignal, yieldRender))) return
    const pasture = trailPasture(layout.landscape, routes.geometries, sceneSeed, animals)
    const fauna = scene.querySelector('.trail-fauna')
    if (fauna) fauna.innerHTML = pasture.markup
    model.forestLayout = layout.landscape
    paintSunlight()
    refreshForestObstacles(pasture.sites)
    model.forestNow = Date.now()
    model.treePlan = planTrailTrees(
      layout.landscape,
      sceneSeed,
      model.forestObstacles,
      forestClearings(layout.landscape),
      model.forestNow,
      (model.plannedWater = model.groveWater),
    )
    const regularSprites = [...forestSprites(layout.landscape)]
    const visibleTrees = new Set(regularSprites.map(sprite => sprite.id))
    const treePreparation = prepareSceneTrees(
      layout.landscape,
      model.treePlan.filter(placement => visibleTrees.has(placement.id)).sort((a, b) => b.y - a.y),
      forest,
      renderSignal,
      { seed: sceneSeed, clearings: beechClearings(layout.landscape) },
    )
    const painting = treePreparation.trees
    routes.paintApproach(layout)
    const terraces = await scenery.get('terraces')
    if (terraces === null || renderSignal.aborted) return
    const base = scene.querySelector('.trail-ground-base')
    if (base) base.innerHTML = terraces
    if (!(await yieldRender(renderSignal))) return
    const hinterlandMarkup = await scenery.get('hinterland')
    const terrain = await scenery.get('terrain')
    if (hinterlandMarkup === null || terrain === null || renderSignal.aborted) return
    const hinterland = scene.querySelector('.trail-hinterland')
    if (hinterland) hinterland.innerHTML = hinterlandMarkup
    const ground = scene.querySelector('.trail-ground')
    if (ground) ground.innerHTML = terrain
    if (!(await yieldRender(renderSignal))) return
    const backdropMarkup = await scenery.get('backdrop')
    if (backdropMarkup === null || renderSignal.aborted) return
    const backdrop = scene.querySelector('[data-trail-backdrop-container]')
    if (backdrop) backdrop.innerHTML = backdropMarkup
    if (!(await routes.paintPaths(layout, renderSignal, yieldRender))) return
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
    community.update(layout.landscape, sceneSeed, { ...camps }, [])
    if (!(await yieldRender(renderSignal))) return
    forestNodes.clear()
    woodlandNodes.clear()
    const landmarks = scene.querySelector('.trail-landmarks')
    landmarks?.replaceChildren()
    const forestDraft = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    let batch = performance.now()
    const foregroundSprites = await treePreparation.foreground
    if (renderSignal.aborted) return
    const sprites = [...foregroundSprites, ...regularSprites]
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
    if (!(await yieldRender(renderSignal))) return
    holidays.draw(layout.landscape, routes.geometries, sceneSeed, camps)
    groundDepth.refresh()
    moss.plant(
      layout.landscape,
      model.treePlan,
      routes.geometries.map(route => route.samples),
    )
    weather.refreshForest()
    grove.refresh()
    const routing = routes.prepare(layout.landscape, renderSignal)
    if (!(await yieldRender(renderSignal))) return
    scene.dataset.trailRasterReady = 'true'
    host.dispatchEvent(new Event('trail:prepare'))
    const preparedRoutes = await routing
    if (renderSignal.aborted) return
    const currentRoutes = routes.commit(preparedRoutes)
    community.update(layout.landscape, sceneSeed, { ...camps }, currentRoutes)
    if (!(await yieldRender(renderSignal))) return
    visitorJourneys.update(currentRoutes, layout.landscape, sceneSeed, camps)
    if (!(await yieldRender(renderSignal))) return
    parkPatrol.update(layout.landscape, sceneSeed, currentRoutes, travelers.rangers)
    athletes.update(layout.landscape, sceneSeed, currentRoutes)
    if (!(await yieldRender(renderSignal))) return
    journey.update(currentRoutes, layout.landscape, sceneSeed, travelers.walkers > 0)
    if (!(await yieldRender(renderSignal))) return
    await readerJourneys.update(
      currentRoutes,
      layout.landscape,
      sceneSeed,
      Math.max(0, travelers.walkers - 1),
      routes.bodies,
      renderSignal,
    )
    if (!(await yieldRender(renderSignal))) return
    pastureJourneys.update(layout.landscape, sceneSeed)
    if (!(await yieldRender(renderSignal))) return
    alpineJourneys.update(layout.landscape, sceneSeed, travelers.climbers, travelers.alpinists)
    if (!(await yieldRender(renderSignal))) return
    perspective.update()
    groundDepth.refresh()
    svg.style.removeProperty('display')
    releaseClock()
    // Forest measurements include the completed actor layout before the compositor reads it.
    refreshForest()
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
  const suspend = () => {
    if (rendering && scene.dataset.trailRendered !== 'true') {
      cancelRendering()
      layoutSignature = ''
    }
  }
  return { draw: drawScene, suspend }
}
