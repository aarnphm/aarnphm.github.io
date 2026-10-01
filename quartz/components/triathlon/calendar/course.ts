import {
  decodeCalendarPolyline,
  type TriathlonCalendarCourse,
  type TriathlonCalendarLeg,
} from '../../../util/triathlon-calendar'

export const ROUTE_SIZE = 100
export const PROFILE_WIDTH = 100
export const PROFILE_HEIGHT = 20

const ROUTE_PAD = 7
// Swims sit on top: they are the shortest legs and usually hide inside the bike loop.
const DRAW_ORDER: readonly TriathlonCalendarLeg[] = ['bike', 'run', 'swim']
const RACE_ORDER: readonly TriathlonCalendarLeg[] = ['swim', 'bike', 'run']

export interface CourseFigure {
  legs: { leg: TriathlonCalendarLeg; d: string }[]
  start: [number, number]
  finish: [number, number]
}

const round = (value: number): number => Math.round(value * 10) / 10

/** Projects every leg into one square frame so their relative positions stay true. */
export const courseFigure = (courses: readonly TriathlonCalendarCourse[]): CourseFigure | null => {
  const legs = courses
    .map(course => ({ leg: course.leg, points: decodeCalendarPolyline(course.polyline) }))
    .filter(entry => entry.points.length >= 2)
  if (legs.length === 0) return null
  const points = legs.flatMap(entry => entry.points)
  const meanLat = points.reduce((sum, [lat]) => sum + lat, 0) / points.length
  const xScale = Math.cos((meanLat * Math.PI) / 180)
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const [lat, lng] of points) {
    minX = Math.min(minX, lng * xScale)
    maxX = Math.max(maxX, lng * xScale)
    minY = Math.min(minY, lat)
    maxY = Math.max(maxY, lat)
  }
  const spanX = Math.max(maxX - minX, 1e-6)
  const spanY = Math.max(maxY - minY, 1e-6)
  const scale = Math.min((ROUTE_SIZE - 2 * ROUTE_PAD) / spanX, (ROUTE_SIZE - 2 * ROUTE_PAD) / spanY)
  const offsetX = (ROUTE_SIZE - spanX * scale) / 2
  const offsetY = (ROUTE_SIZE - spanY * scale) / 2
  const project = ([lat, lng]: [number, number]): [number, number] => [
    round(offsetX + (lng * xScale - minX) * scale),
    round(offsetY + (maxY - lat) * scale),
  ]
  const ordered = [...legs].sort(
    (left, right) => RACE_ORDER.indexOf(left.leg) - RACE_ORDER.indexOf(right.leg),
  )
  const first = ordered[0].points
  const last = ordered[ordered.length - 1].points
  return {
    legs: [...legs]
      .sort((left, right) => DRAW_ORDER.indexOf(left.leg) - DRAW_ORDER.indexOf(right.leg))
      .map(entry => ({
        leg: entry.leg,
        d: entry.points
          .map((point, index) => {
            const [x, y] = project(point)
            return `${index === 0 ? 'M' : 'L'}${x} ${y}`
          })
          .join(''),
      })),
    start: project(first[0]),
    finish: project(last[last.length - 1]),
  }
}

/**
 * One elevation scale for every leg of an event, so a flat run reads flat beside a mountain bike
 * course. A floor on the span keeps GPS noise on flat courses from drawing hills.
 */
export const profileDomain = (courses: readonly TriathlonCalendarCourse[]): [number, number] => {
  const values = courses.flatMap(course => course.profile)
  if (values.length === 0) return [0, 1]
  const min = Math.min(...values)
  const max = Math.max(...values)
  return max - min >= 80 ? [min, max] : [min, min + 80]
}

export const profilePaths = (
  profile: readonly number[],
  [min, max]: [number, number],
): { line: string; area: string } | null => {
  if (profile.length < 2) return null
  const span = max - min || 1
  const line = profile
    .map((value, index) => {
      const x = round((index / (profile.length - 1)) * PROFILE_WIDTH)
      const y = round(PROFILE_HEIGHT - 1 - ((value - min) / span) * (PROFILE_HEIGHT - 3))
      return `${index === 0 ? 'M' : 'L'}${x} ${y}`
    })
    .join('')
  return { line, area: `${line}L${PROFILE_WIDTH} ${PROFILE_HEIGHT}L0 ${PROFILE_HEIGHT}Z` }
}

export const formatDistance = (meters: number): string =>
  meters < 1000 ? `${Math.round(meters)} m` : `${(meters / 1000).toFixed(1)} km`
