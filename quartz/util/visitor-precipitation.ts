// One reading of the WMO 4677 present-weather code for every scene that paints the visitor's sky:
// the homepage trail, its page overlay, and the 404 landscape.
export type VisitorFall = 'none' | 'drizzle' | 'rain' | 'snow'
export type VisitorPrecipitation = {
  fall: VisitorFall
  // How hard it is coming down, 0..1.
  heavy: number
  fog: boolean
  storm: boolean
}

// Open-Meteo sends only these codes. A code inside a band but missing here takes the band's
// lightest strength.
const drizzle = new Map([
  [51, 0.25],
  [53, 0.4],
  [55, 0.55],
  [56, 0.3],
  [57, 0.5],
])
const rain = new Map([
  [61, 0.35],
  [63, 0.6],
  [65, 0.9],
  [66, 0.4],
  [67, 0.7],
  [80, 0.45],
  [81, 0.7],
  [82, 1],
  [95, 0.8],
  [96, 0.9],
  [97, 0.95],
  [99, 1],
])
const snow = new Map([
  [71, 0.35],
  [73, 0.6],
  [75, 0.9],
  [77, 0.2],
  [85, 0.5],
  [86, 0.9],
])

export function visitorPrecipitation(code: number | null | undefined): VisitorPrecipitation {
  const c = typeof code === 'number' && Number.isFinite(code) ? code : -1
  const fog = c === 45 || c === 48
  const storm = c >= 95
  if (c >= 51 && c <= 57) return { fall: 'drizzle', heavy: drizzle.get(c) ?? 0.25, fog, storm }
  if ((c >= 61 && c <= 67) || (c >= 80 && c <= 82) || storm)
    return { fall: 'rain', heavy: rain.get(c) ?? 0.35, fog, storm }
  if ((c >= 71 && c <= 77) || c === 85 || c === 86)
    return { fall: 'snow', heavy: snow.get(c) ?? 0.35, fog, storm }
  return { fall: 'none', heavy: 0, fog, storm }
}

// Rain and drizzle both wet the ground and put out a fire.
export const wetFall = (fall: VisitorFall) => fall === 'rain' || fall === 'drizzle'

// A storm flashes on one wall clock, so the page's flash and the trail's bolt land together
// whichever bundle sets them up.
export const STORM_PERIOD = 11
export const stormDelay = (now = Date.now()) => `${(-((now / 1000) % STORM_PERIOD)).toFixed(2)}s`
