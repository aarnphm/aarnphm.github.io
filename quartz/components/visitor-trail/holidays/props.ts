import type { RouteSites } from '../pasture'
import { circleClearance, readActors, readTrunks } from '../actor-spacing'
import { holidayArt, holidaySprite, type HolidaySpriteName } from '../holiday-sprites'
import { onDryLand, pondDistance, type Landscape, type Point } from '../landscape'

// A pool of candlelight on the ground, shown at night, in 2 px rows like the campfire's.
export function lanternGlow(width: number, depth: number) {
  const pool = (radius: number, rise: number) => {
    let d = ''
    for (let y = -rise; y < rise; y += 2) {
      const half = Math.round((radius * Math.sqrt(1 - ((y + 1) / rise) ** 2)) / 2) * 2
      if (half) d += `M${-half} ${y}h${half * 2}v2h${-half * 2}Z`
    }
    return d
  }
  return `<path class="trail-holiday-glow" d="${pool(width, depth)}"/><path class="trail-holiday-glow-core" d="${pool(width * 0.55, depth * 0.6)}"/>`
}

export type PropOptions = {
  lift?: number
  under?: string
  variant?: string
  radius?: number
  // Ground props draw at the scene's 2 px cell; a tree-sized piece takes 3 px.
  size?: 2 | 3
}

// Places ground props clear of the pond, trunks, actors, the road and one another. `prop` returns
// the ground point it used, so a cluster can grow from a placed piece.
export function createPropPlacer(scene: HTMLElement, landscape: Landscape, routes: RouteSites[]) {
  const occupied: (Point & { radius: number })[] = []
  const solids = readTrunks(scene)
  const bodies = readActors(scene)
  const road = routes.flatMap(route => route.samples)
  const lowerRoad = road.filter(point => point.y >= landscape.meadowTop + 80)
  const safe = (anchor: Point, radius: number) => {
    // The spiral widens with the footprint: a crypt needs more room to find than a pumpkin.
    for (let attempt = 0; attempt < 144; attempt++) {
      const angle = attempt * 2.399963
      const reach = attempt ? 6 + Math.floor(attempt / 12) * (8 + radius / 3) : 0
      const candidate = onDryLand(
        { x: anchor.x + Math.cos(angle) * reach, y: anchor.y + Math.sin(angle) * reach },
        landscape,
        1.8,
      )
      if (
        pondDistance(candidate, landscape.pond) >= 1.8 &&
        solids.every(solid => circleClearance(candidate, radius, solid) >= 0) &&
        bodies.every(body => circleClearance(candidate, radius, body) >= 0) &&
        occupied.every(
          point =>
            Math.hypot(candidate.x - point.x, candidate.y - point.y) > point.radius + radius + 8,
        ) &&
        road.every(point => Math.hypot(candidate.x - point.x, candidate.y - point.y) > radius + 12)
      ) {
        occupied.push({ ...candidate, radius })
        return candidate
      }
    }
    return null
  }
  const props: string[] = []
  const prop = (
    name: HolidaySpriteName,
    anchor: Point,
    { lift = 0, under = '', variant = '', radius = 12, size = 2 }: PropOptions = {},
  ): Point | null => {
    const point = safe(anchor, radius)
    if (!point) return null
    const rows = holidayArt[name]
    const x = Math.round(point.x)
    const y = Math.round(point.y)
    props.push(
      `<g class="trail-holiday-${name}${variant ? ` ${variant}` : ''}" data-holiday-decoration="${name}" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x} ${y})">${under}<g transform="translate(${-(rows[0].length * size) / 2} ${-rows.length * size - lift})"><g${name === 'ghost' ? ' class="trail-holiday-float"' : ''}>${holidaySprite(name, size)}</g></g></g>`,
    )
    return { x, y }
  }
  return { prop, props, road, lowerRoad }
}

export type PropPlacer = ReturnType<typeof createPropPlacer>
