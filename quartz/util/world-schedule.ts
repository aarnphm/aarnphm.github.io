import type { Almanac, Sky } from '../components/scripts/404-almanac'
import type { Season } from './visitor-world'
import { wetFall } from './visitor-precipitation'

// What the world does at a given sun, season and sky. The 404 and the homepage trail both read it.

export type HerdState = 'grazing' | 'lying' | 'away'

// Night begins at civil dusk, the 404's own rule.
export const night = (al: Pick<Almanac, 'sun'>) => al.sun.alt < -6

// Campers stay by the fire for about two hours after civil dusk (the sun near -20°), then sleep
// until civil dawn. At high summer latitudes the sun may never get that low, so 22:00 also counts.
export const campSleeps = (al: Pick<Almanac, 'sun' | 'hour'>) =>
  night(al) && (al.hour < 12 || al.sun.alt < -20 || al.hour >= 22)

// Cows leave the pasture for the barn when snow lies or a storm comes at night. They lie down at
// night and through heavy rain or a storm, and graze otherwise.
export function herd(al: Pick<Almanac, 'sun'>, s: Pick<Season, 'ground'>, sky: Sky): HerdState {
  if (s.ground >= 0.5 || (sky.storm && night(al))) return 'away'
  if (night(al) || sky.storm || (wetFall(sky.fall) && sky.heavy >= 0.6)) return 'lying'
  return 'grazing'
}

// Rain or falling snow puts the camp fire out.
export const fireOut = (sky: Pick<Sky, 'fall'>) => sky.fall !== 'none'

// Leaves per tree per second: they fall with the autumn colour and the wind, and stop once the
// crowns are bare. At the height of autumn in still air this is the old peak rate of 0.08.
export function leafRate(s: Pick<Season, 'autumn' | 'bare'>, sky: Pick<Sky, 'wind'>) {
  if (s.bare > 0.8 || s.autumn <= 0) return 0
  return s.autumn * (1 - s.bare) * (0.02 + 0.06 * s.autumn) * (1 + sky.wind)
}
