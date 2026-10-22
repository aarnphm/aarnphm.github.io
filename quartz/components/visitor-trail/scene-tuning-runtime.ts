import type { CollisionActor } from './actor-spacing'
import type { mountAlpineJourneys } from './alpine-journey'
import type { mountCampCommunity } from './camp-community'
import type { mountGroundDepth } from './depth-order'
import type { createForestSprites } from './forest-sprites'
import type { mountHikerJourney } from './hiker-journey'
import type { mountTrailHolidays } from './holidays'
import type { Landscape, Point } from './landscape'
import type { NavigableTrail } from './obstacle-route'
import type { mountParkPatrol } from './park-patrol'
import type { RouteSites } from './pasture'
import type { mountPastureWanderers } from './pasture-wander'
import type { SceneSettings } from './scene-settings'
import type { createTreePainter, TreePlacement } from './scenery'
import type { mountVisitorJourneys } from './visitor-journey'
import type { mountMapReaderJourneys } from './wayfarer-journey'
import type { mountWorldPatches } from './world-patches'
import { trailPasture } from './pasture'
import { trailFoothillWoodland, trailTreeSprites } from './scenery'
import { prepareSceneTrees } from './tree-workers'

export type SceneTuningGroup = 'forest' | 'animals' | 'travelers' | 'camps'
type ForestState = {
  landscape: Landscape
  plan: TreePlacement[]
  obstacles: Point[]
  clearings: (Point & { radius: number })[]
  foregroundClearings: (Point & { radius: number })[]
  geometries: RouteSites[]
}
type TuningContext = {
  scene: HTMLElement
  host: HTMLElement
  signal: AbortSignal
  settings: SceneSettings
  seed: number
  read: (refreshObstacles: boolean) => ForestState | null
  replan: () => void
  refreshForest: () => void
  routes: (signal: AbortSignal) => Promise<{ routes: NavigableTrail[]; solids: CollisionActor[] }>
  forestNodes: ReturnType<typeof createForestSprites>
  woodlandNodes: ReturnType<typeof createForestSprites>
  paintTree: ReturnType<typeof createTreePainter>
  paintWoodland: ReturnType<typeof createTreePainter>
  groundDepth: ReturnType<typeof mountGroundDepth>
  community: ReturnType<typeof mountCampCommunity>
  visitorJourneys: ReturnType<typeof mountVisitorJourneys>
  journey: ReturnType<typeof mountHikerJourney>
  readerJourneys: ReturnType<typeof mountMapReaderJourneys>
  alpineJourneys: ReturnType<typeof mountAlpineJourneys>
  parkPatrol: ReturnType<typeof mountParkPatrol>
  pastureJourneys: ReturnType<typeof mountPastureWanderers>
  holidays: ReturnType<typeof mountTrailHolidays>
  worldPatches: ReturnType<typeof mountWorldPatches>
}

// Structural edits commit after a gesture. Jobs touch only their owning populations and keep
// the retained terrain, sky, canvas, scene seed and unchanged actor nodes alive.
export function mountSceneTuning(context: TuningContext) {
  const { scene, host, signal, settings, seed } = context
  const view = scene.ownerDocument.defaultView
  if (!view) throw new Error('Scene tuning needs its window')
  const pending = new Set<SceneTuningGroup>()
  let timer = 0
  let working: AbortController | null = null
  let active = new Set<SceneTuningGroup>()
  let previousCamps = { ...settings.camps }
  let previousTravelers = { ...settings.travelers }
  const ready = () =>
    !signal.aborted &&
    !host.hidden &&
    scene.isConnected &&
    scene.dataset.trailRendered === 'true' &&
    scene.dataset.trailOpening === 'complete'
  const yieldWork = (signal: AbortSignal) =>
    new Promise<void>(resolve => {
      if (signal.aborted) return resolve()
      view.setTimeout(resolve, 0)
    })
  const forest = async (state: ForestState, signal: AbortSignal, replan: boolean) => {
    const { landscape, plan, obstacles, clearings, foregroundClearings } = state
    const current = { ...settings.forest }
    const sprites = [
      ...trailTreeSprites(
        landscape,
        seed,
        current,
        obstacles,
        clearings,
        context.paintTree,
        true,
        plan,
      ),
    ]
    const changed = new Set(
      sprites.filter(context.forestNodes.needsUpdate).map(sprite => sprite.id),
    )
    const prepared = prepareSceneTrees(
      landscape,
      plan.filter(tree => changed.has(tree.id)),
      current,
      signal,
      replan ? { seed, clearings: foregroundClearings } : undefined,
    )
    const foreground = await prepared.foreground
    if (signal.aborted) return
    const landmarks = scene.querySelector('.trail-landmarks')
    const woodland = scene.querySelector('.trail-woodland')
    if (!landmarks || !woodland) return
    const wanted = new Set<string>()
    if (!replan)
      for (const tree of scene.querySelectorAll('[data-tree-repoussoir]')) {
        const id = tree.getAttribute('data-tree-id')
        if (id) wanted.add(id)
      }
    let slice = view.performance.now()
    for (const sprite of [...foreground, ...sprites]) {
      wanted.add(sprite.id)
      if (!context.forestNodes.needsUpdate(sprite)) continue
      const markup = await prepared.trees.get(sprite.id)
      if (signal.aborted) return
      context.forestNodes.update(markup ? { ...sprite, render: () => markup } : sprite, landmarks)
      if (view.performance.now() - slice >= 4) {
        await yieldWork(signal)
        slice = view.performance.now()
      }
    }
    if (signal.aborted) return
    context.forestNodes.retain(wanted)
    const distant = new Set<string>()
    for (const sprite of trailFoothillWoodland(
      landscape,
      seed,
      current,
      obstacles,
      context.paintWoodland,
      plan,
    )) {
      if (signal.aborted) return
      distant.add(sprite.id)
      context.woodlandNodes.update(sprite, woodland)
      if (view.performance.now() - slice >= 4) {
        await yieldWork(signal)
        slice = view.performance.now()
      }
    }
    if (signal.aborted) return
    context.woodlandNodes.retain(distant)
    const foothillTrees = scene.querySelector('.trail-foothill-trees')
    foothillTrees?.append(...landmarks.querySelectorAll('[data-tree-depth="foothill"]'))
    context.worldPatches.refreshForest()
    context.refreshForest()
  }
  const animals = (state: ForestState) => {
    const layer = scene.querySelector('.trail-fauna')
    if (!layer) return
    const draft = scene.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'g')
    draft.innerHTML = trailPasture(state.landscape, state.geometries, seed, settings.animals).markup
    const key = (element: Element) => element.getAttribute('data-trail-wander') ?? 'river-bear'
    const selector = '[data-trail-wander], .trail-river-bear'
    const existing = new Map([...scene.querySelectorAll(selector)].map(node => [key(node), node]))
    const wanted = new Set<string>()
    for (const node of draft.querySelectorAll(selector)) {
      const id = key(node)
      wanted.add(id)
      if (!existing.has(id)) layer.append(node)
    }
    for (const [id, node] of existing) if (!wanted.has(id)) node.remove()
    context.pastureJourneys.update(state.landscape, seed)
    context.refreshForest()
  }
  const travelers = async (
    state: ForestState,
    routes: NavigableTrail[],
    solids: CollisionActor[],
    signal: AbortSignal,
    rerouted: boolean,
  ) => {
    const { landscape } = state
    const walkersChanged = previousTravelers.walkers !== settings.travelers.walkers
    if (rerouted || walkersChanged)
      context.journey.update(routes, landscape, seed, settings.travelers.walkers > 0)
    if (rerouted || walkersChanged)
      await context.readerJourneys.update(
        routes,
        landscape,
        seed,
        Math.max(0, settings.travelers.walkers - 1),
        solids,
        signal,
      )
    if (signal.aborted) return
    if (
      previousTravelers.climbers !== settings.travelers.climbers ||
      previousTravelers.alpinists !== settings.travelers.alpinists
    )
      context.alpineJourneys.update(
        landscape,
        seed,
        settings.travelers.climbers,
        settings.travelers.alpinists,
      )
    if (rerouted || previousTravelers.rangers !== settings.travelers.rangers)
      context.parkPatrol.update(landscape, seed, routes, settings.travelers.rangers)
  }
  const apply = async () => {
    timer = 0
    if (!pending.size || !ready()) return
    const groups = new Set(pending)
    active = groups
    pending.clear()
    const request = new AbortController()
    working = request
    const current = request.signal
    try {
      const campsChanged = groups.has('camps')
      const campGeometry =
        campsChanged &&
        (previousCamps.sites !== settings.camps.sites ||
          previousCamps.tents !== settings.camps.tents)
      if (campGeometry) context.replan()
      const state = context.read(groups.has('forest') && !campGeometry)
      if (!state) return
      if (groups.has('forest') || campGeometry) await forest(state, current, campGeometry)
      if (current.aborted || !ready()) return
      if (groups.has('animals')) animals(state)
      const routing =
        groups.has('forest') || campGeometry || groups.has('travelers') || campsChanged
      if (routing) {
        const { routes, solids } = await context.routes(current)
        if (current.aborted || !ready()) return
        if (groups.has('forest') || campsChanged) {
          if (
            campsChanged &&
            (previousCamps.sites !== settings.camps.sites ||
              previousCamps.tents !== settings.camps.tents ||
              previousCamps.builders !== settings.camps.builders)
          )
            context.groundDepth.restore('[data-camp-person], [data-camp-tent], .trail-extra-camp')
          context.community.update(state.landscape, seed, { ...settings.camps }, routes)
          context.visitorJourneys.update(routes, state.landscape, seed, settings.camps)
        }
        if (groups.has('forest') || campGeometry || groups.has('travelers'))
          await travelers(state, routes, solids, current, groups.has('forest') || campGeometry)
        if (current.aborted || !ready()) return
      }
      if (campsChanged) {
        previousCamps = { ...settings.camps }
        context.holidays.draw(state.landscape, state.geometries, seed, settings.camps)
        context.pastureJourneys.update(state.landscape, seed)
      }
      previousTravelers = { ...settings.travelers }
      context.groundDepth.refresh()
      working = null
      active = new Set()
      view.performance.mark('trail:tuning-commit', { detail: { groups: [...groups] } })
      scene.dispatchEvent(new view.Event('trail:tuning-commit'))
      if (pending.size) schedule()
    } catch (error) {
      if (current.aborted || signal.aborted || working !== request) return
      working = null
      active = new Set()
      console.error('Trail tuning failed', error)
      scene.dispatchEvent(
        new view.CustomEvent('trail:tuning-error', {
          detail: error instanceof Error ? error.message : 'Scene update failed',
        }),
      )
    } finally {
      if (working === request) {
        for (const group of active) pending.add(group)
        working = null
        active = new Set()
        if (pending.size && ready()) schedule()
      }
    }
  }
  const schedule = () => {
    if (timer || signal.aborted) return
    timer = view.setTimeout(() => void apply(), 0)
  }
  const cancel = () => {
    for (const group of active) pending.add(group)
    active = new Set()
    working?.abort()
    working = null
    view.clearTimeout(timer)
    timer = 0
  }
  host.addEventListener('trail:ready', schedule, { signal })
  host.addEventListener('trail:reveal-end', schedule, { signal })
  host.addEventListener('trail:hide', cancel, { signal })
  signal.addEventListener('abort', cancel, { once: true })
  return {
    change(group: SceneTuningGroup) {
      pending.add(group)
      cancel()
      schedule()
    },
    cancel,
    busy: () => working !== null,
  }
}
