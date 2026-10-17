import type { Almanac, Sky } from '../components/scripts/404-almanac'
import type { Season } from './visitor-world'
import { smooth } from '../components/scripts/404-pixel'
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

// How the moss is faring. Moss holds no water of its own: rain, drizzle, fog and wet ground green
// it within minutes, and sun, wind and heat dry it pale and dormant within hours; night dew brings
// some of it back. It grows in cool weather above freezing, slows in heat, and rests while frozen
// or under lying snow, so a boot print in it heals only while it grows. Rime takes the frost.
export type MossClimate = {
  // -1 dry and pale .. 1 soaked and dark.
  moisture: number
  // Healing speed, 0 dormant .. 1 a wet, mild day.
  growth: number
  // Share of the clump under snow, and share white with rime.
  buried: number
  rime: number
}

export function mossClimate(
  al: Pick<Almanac, 'sun'>,
  s: Pick<Season, 'ground' | 'frost' | 'thaw'>,
  sky: Sky,
  damp: boolean,
): MossClimate {
  const wet =
    sky.fall === 'rain'
      ? 1
      : sky.fall === 'drizzle'
        ? 0.8
        : sky.fall === 'snow' || damp
          ? 0.6
          : sky.fog
            ? 0.5
            : 0
  const sun = al.sun.alt > 0 ? (1 - sky.cover) * Math.min(1, al.sun.alt / 30) : 0
  const heat = sky.temp === null ? 0 : smooth(18, 30, sky.temp)
  const drying = (0.4 * sun + 0.35 * sky.wind + 0.45 * heat) * (1 - wet)
  const dew = al.sun.alt < 0 ? (sky.wind < 0.15 ? 0.2 : 0.1) : 0
  // Cool air holds moisture in the clump; heat takes that away too.
  const moisture = Math.max(-1, Math.min(1, wet + dew + 0.15 * (1 - heat) - drying))
  const thawed = sky.temp === null ? 1 : smooth(-1, 4, sky.temp)
  const buried = Math.min(1, s.ground)
  const growth = buried >= 0.5 ? 0 : thawed * (1 - 0.6 * heat) * (0.1 + 0.45 * (moisture + 1))
  // Rime holds where the sun has not yet taken the frost back, as on the 404.
  return { moisture, growth: Math.min(1, growth), buried, rime: s.frost * (1 - s.thaw) }
}
