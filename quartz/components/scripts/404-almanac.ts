// Where the sun and moon stand for whoever is looking, the season where they are, and the weather
// they are standing in. The worker places the visitor from Cloudflare's reading of their address,
// rounded to a half degree, and passes on Open-Meteo's current conditions for that cell. Without it
// the page guesses 45° of latitude on whichever side of the equator the time zone lies, and a
// longitude from the clock's offset from UTC.

// Open-Meteo's current conditions, as the worker passes them on.
export type Weather = {
  // WMO 4677 present-weather code.
  code: number
  // Cloud cover, 0..1.
  cloud: number
  // Wind at 10 m, km/h.
  wind: number
  // Air temperature at 2 m, °C.
  temp: number
}
export type Site = { lat: number; lon: number; place: string | null; weather: Weather | null }

export type Fall = 'none' | 'drizzle' | 'rain' | 'snow'
export type Sky = {
  cover: number
  fall: Fall
  // How hard it is coming down, 0..1.
  heavy: number
  fog: boolean
  storm: boolean
  // 0 calm .. 1 a gale.
  wind: number
  temp: number | null
}

// A body in the sky: degrees above the horizon, and how far across the view it stands, from -1 at
// the left edge to 1 at the right (the view faces the equator, so the sun rises on the left).
export type Body = { alt: number; across: number }
export type Almanac = {
  sun: Body
  // Phase runs 0 new, 0.25 first quarter, 0.5 full, 0.75 last quarter.
  moon: Body & { phase: number }
  // Daylight, 0 night to 1 full day, and the glow of twilight, 0..1, peaking a degree past sunset.
  day: number
  dusk: number
  // Days since the solstice nearest midwinter, so every season reads the same north and south.
  winterDay: number
  // Local solar time in hours from midnight, leaving out the equation of time (a quarter hour at most).
  hour: number
  south: boolean
}

const RAD = Math.PI / 180
// The view looks this many degrees east of the equator, so the noon sun stands right of the gate
// and the sunset keeps to the right edge.
const VIEW = 35
const SYNODIC = 29.530588853
// A new moon, 2000-01-06 18:14 UTC, as days since the Unix epoch.
const NEW_MOON = 10962.7597

const smooth = (a: number, b: number, v: number) => {
  const t = Math.max(0, Math.min(1, (v - a) / (b - a)))
  return t * t * (3 - 2 * t)
}
const wrap = (deg: number) => ((((deg + 180) % 360) + 360) % 360) - 180

// Low-precision ephemeris (the Astronomical Almanac's solar formula, good to about a hundredth of a
// degree). The moon is placed on the ecliptic at its elongation from the sun, which ignores its
// five-degree tilt: close enough to say where it hangs, and exact about its phase.
function place(days: number, lat: number, lon: number, elongation: number): Body {
  const n = days - 10957.5
  const g = (357.528 + 0.9856003 * n) * RAD
  const lambda = (280.46 + 0.9856474 * n + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * RAD
  const l = lambda + elongation * RAD
  const eps = (23.439 - 4e-7 * n) * RAD
  const ra = Math.atan2(Math.cos(eps) * Math.sin(l), Math.cos(l))
  const dec = Math.asin(Math.sin(eps) * Math.sin(l))
  const ha = (280.46061837 + 360.98564736629 * n + lon) * RAD - ra
  const phi = lat * RAD
  const alt = Math.asin(
    Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(ha),
  )
  // Azimuth from the south, positive toward the west.
  const az =
    Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(phi) - Math.tan(dec) * Math.cos(phi)) / RAD
  const across = ((lat >= 0 ? az : wrap(az + 180)) + VIEW) / 90
  return { alt: alt / RAD, across: Math.max(-1.25, Math.min(1.25, across)) }
}

export function almanac(now: Date, site: Site): Almanac {
  const days = now.getTime() / 864e5
  const phase = ((((days - NEW_MOON) / SYNODIC) % 1) + 1) % 1
  const sun = place(days, site.lat, site.lon, 0)
  const moon = { ...place(days, site.lat, site.lon, phase * 360), phase }
  const south = site.lat < 0
  // Midwinter falls on the 21st of December in the north and of June in the south.
  const solstice = Date.UTC(now.getUTCFullYear() - 1, south ? 5 : 11, 21) / 864e5
  const winterDay = (days - solstice) % 365.2422
  return {
    sun,
    moon,
    day: smooth(-9, 3, sun.alt),
    dusk: Math.max(0, 1 - Math.abs(sun.alt + 1) / 10),
    winterDay,
    hour: ((((days % 1) * 24 + site.lon / 15) % 24) + 24) % 24,
    south,
  }
}

// A guess at where the visitor is when the worker cannot say: the clock's UTC offset puts them
// within a time zone's width of longitude, and the zone's name tells north from south.
export function guessSite(): Site {
  let zone = ''
  try {
    zone = Intl.DateTimeFormat().resolvedOptions().timeZone ?? ''
  } catch {}
  const south =
    /^(Australia|Antarctica|Pacific\/(Auckland|Chatham|Fiji|Tongatapu|Apia|Noumea|Efate))|^America\/(Argentina|Santiago|Sao_Paulo|Montevideo|Asuncion|La_Paz|Lima)|^Africa\/(Johannesburg|Maputo|Harare|Windhoek|Gaborone|Lusaka|Maseru|Mbabane)|^Indian\/(Mauritius|Reunion|Antananarivo)/.test(
      zone,
    )
  return {
    lat: south ? -35 : 45,
    lon: -new Date().getTimezoneOffset() / 4,
    place: null,
    weather: null,
  }
}

// The WMO code, read for what the scene has to paint.
export function readSky(w: Weather | null): Sky {
  if (!w)
    return { cover: 0.3, fall: 'none', heavy: 0, fog: false, storm: false, wind: 0.2, temp: null }
  const c = w.code
  const steps = (codes: number[], at: number[]) => at[Math.max(0, codes.indexOf(c))]
  let fall: Fall = 'none'
  let heavy = 0
  if (c >= 51 && c <= 57) {
    fall = 'drizzle'
    heavy = steps([51, 53, 55, 56, 57], [0.25, 0.4, 0.55, 0.3, 0.5])
  } else if ((c >= 61 && c <= 67) || (c >= 80 && c <= 82) || c >= 95) {
    fall = 'rain'
    heavy = steps(
      [61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99],
      [0.35, 0.6, 0.9, 0.4, 0.7, 0.45, 0.7, 1, 0.8, 0.9, 1],
    )
  } else if ((c >= 71 && c <= 77) || c === 85 || c === 86) {
    fall = 'snow'
    heavy = steps([71, 73, 75, 77, 85, 86], [0.35, 0.6, 0.9, 0.2, 0.5, 0.9])
  }
  const fog = c === 45 || c === 48
  // Showers and storms need cloud even when the cover reading lags behind them.
  const cover = Math.max(w.cloud, fall === 'none' ? 0 : 0.75, fog ? 0.9 : 0)
  return { cover, fall, heavy, fog, storm: c >= 95, wind: Math.min(1, w.wind / 45), temp: w.temp }
}

// Refresh the visitor's place on each visit; the worker caches the forecast for fifteen minutes.
// A night bright enough that a lamp is not doing all the work: a moon within four days of full, well
// up, through a mostly clear sky. Full-moon rides and skin-ups bring people out after midnight.
export const moonlit = (al: Almanac, sky: Sky) =>
  al.sun.alt < -6 && al.moon.alt > 10 && Math.abs(al.moon.phase - 0.5) < 0.14 && sky.cover < 0.6

export async function fetchSite(signal: AbortSignal): Promise<Site | null> {
  try {
    const res = await fetch('/api/weather', { signal, cache: 'no-store' })
    if (!res.ok) return null
    const body = (await res.json()) as Partial<Site>
    if (
      typeof body.lat !== 'number' ||
      !Number.isFinite(body.lat) ||
      typeof body.lon !== 'number' ||
      !Number.isFinite(body.lon)
    )
      return null
    const place = typeof body.place === 'string' && body.place.trim() ? body.place.trim() : null
    const w = body.weather
    const weather =
      w && [w.code, w.cloud, w.wind, w.temp].every(v => typeof v === 'number' && Number.isFinite(v))
        ? w
        : null
    return { lat: body.lat, lon: body.lon, place, weather }
  } catch {
    return null
  }
}
