import type { CollisionActor } from '../actor-spacing'
import type { Landscape } from '../landscape'
import type { trailLayout, TrailId } from '../layout'
import type { createTrailNavigation } from '../navigation-workers'
import type { NavigableTrail } from '../obstacle-route'
import type { CampSettings } from '../scene-settings'
import { svgPathGeometry } from '../../../util/svg-geometry'
import { invalidateSolids, readSolids } from '../actor-spacing'
import { campSites } from '../camp-layout'
import { footprint, footprintTransform, type FootprintKind } from '../footprints'
import { trailGroundScale } from '../landscape'
import { trailIds } from '../layout'
import { trailRandom } from '../random'

type TrailSample = { x: number; y: number; position: number }
type TrailGeometry = { id: TrailId; length: number; samples: TrailSample[] }
type Layout = NonNullable<ReturnType<typeof trailLayout>>
type YieldRender = (signal: AbortSignal) => Promise<boolean>
type PreparedRoutes = { bodies: CollisionActor[]; routes: NavigableTrail[] }

export function createFrameNavigation(
  scene: HTMLElement,
  seed: number,
  camps: CampSettings,
  landscape: () => Landscape | undefined,
  navigation: ReturnType<typeof createTrailNavigation>,
) {
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!svg) throw new Error('Scene navigation needs its SVG')
  let geometries: TrailGeometry[] = []
  let navigationGeometry: TrailGeometry[] | null = null
  let navigationSignature = ''
  let navigationRoutes: NavigableTrail[] = []
  let navigationBodies: CollisionActor[] = []
  const journeyObstacles = (landscape: Landscape) => {
    // A generation continuation can insert trees after an earlier read in the same frame.
    invalidateSolids(scene)
    const solids = new Map(
      readSolids(scene)
        .filter(solid => solid.id.startsWith('trunk-'))
        .map(solid => [solid.id, solid]),
    )
    for (const site of campSites(landscape, seed, camps)) {
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
  const journeyRoutes = (prepared: { bodies: CollisionActor[]; routes: NavigableTrail[] }) => {
    if (!landscape()) return []
    const bodies = prepared.bodies
    const signature = bodies.map(body => `${body.id}:${body.x}:${body.y}:${body.radius}`).join('|')
    if (navigationGeometry === geometries && navigationSignature === signature)
      return navigationRoutes
    navigationGeometry = geometries
    navigationSignature = signature
    navigationBodies = bodies
    navigationRoutes = prepared.routes
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
  const kinds: FootprintKind[] = ['boot', 'paw', 'hoof', 'blob']
  const offset = Math.floor(trailRandom(seed ^ 0x2c681f4b)() * kinds.length)

  return {
    get geometries() {
      return geometries
    },
    get bodies() {
      return navigationBodies
    },
    async prepare(current: Landscape, signal: AbortSignal): Promise<PreparedRoutes> {
      const bodies = journeyObstacles(current)
      const signature = bodies
        .map(body => `${body.id}:${body.x}:${body.y}:${body.radius}`)
        .join('|')
      if (navigationGeometry === geometries && navigationSignature === signature)
        return { routes: navigationRoutes, bodies: navigationBodies }
      return {
        bodies,
        routes: await navigation.prepare(
          journeyInputs(),
          current,
          bodies,
          signal,
          scene.dataset.trailOpening === 'complete' ? 'active' : 'opening',
        ),
      }
    },
    commit: journeyRoutes,
    async sample(layout: Layout, renderSignal: AbortSignal, yieldRender: YieldRender) {
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
          if (index % 16 === 0 && !(await yieldRender(renderSignal))) return false
        }
        geometries.push({ id, length, samples })
        if (!(await yieldRender(renderSignal))) return false
      }
      return true
    },
    paintApproach(layout: Layout) {
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
    },
    async paintPaths(layout: Layout, renderSignal: AbortSignal, yieldRender: YieldRender) {
      const readingPaths = svg.querySelector('[data-pet-walk-paths]')
      if (readingPaths)
        readingPaths.innerHTML = layout.readingRoutes
          .map(
            (route, index) =>
              `<path class="trail-path" data-pet-walk-path="${index}" d="${route}"/>`,
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
            if (marks.length % 16 === 0 && !(await yieldRender(renderSignal))) return false
          }
          steps.insertAdjacentHTML('beforeend', `<g data-route="reading">${marks.join('')}</g>`)
          if (!(await yieldRender(renderSignal))) return false
        }
      }
      return true
    },
  }
}
