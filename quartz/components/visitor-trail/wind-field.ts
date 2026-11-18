// One wind for the whole scene: when a gust passes a point, how hard it blows there, and how a
// crown answers it. Crowns and falling leaves read the same tables, so they move together.

export type WindShape = 'broadleaf' | 'conifer'
export type WindTables = {
  gust: Float32Array
  breeze: Float32Array
  lean: Record<WindShape, Float32Array>
}

const samples = 256
// A gust takes about half a second to pass a point; this is the deviation of its bell, in seconds.
const gustSpan = 0.22
// Where the first gust peaks in the period, with the streaks and the grass sweeps.
const gustPeak = 0.34
// A second, weaker gust follows this many seconds behind the first.
const gustGap = 1.3
const gustEcho = 0.7
// The breeze around a gust is the same field spread wider. Outside it the air is still.
const breezeSpread = 2.2
// A crown is a damped spring: its own period in seconds and its damping ratio. A conifer is
// slower and rings longer, so it rocks back past rest more than once.
const springs: Record<WindShape, { period: number; damping: number }> = {
  broadleaf: { period: 2.2, damping: 0.35 },
  conifer: { period: 2.8, damping: 0.22 },
}

const bell = (u: number, centre: number, width: number) => {
  let distance = u - centre
  distance -= Math.round(distance)
  return Math.exp(-0.5 * (distance / width) ** 2)
}
const field = (u: number, period: number, spread: number) => {
  const width = (gustSpan / period) * spread
  return bell(u, gustPeak, width) + gustEcho * bell(u, gustPeak + gustGap / period, width)
}

// The lean a crown settles into over one period: the spring driven by the gust, run until it
// repeats, with its largest swing scaled to 1. The swing past rest after the gust comes out of
// the spring and is negative.
function leanTable(gust: Float32Array, period: number, shape: WindShape) {
  const { period: own, damping } = springs[shape]
  const k = (Math.PI * 2) / own
  const substeps = 4
  const dt = period / samples / substeps
  const table = new Float32Array(samples)
  let lean = 0
  let speed = 0
  let peak = 0
  for (let lap = 0; lap < 3; lap++)
    for (let index = 0; index < samples; index++) {
      for (let step = 0; step < substeps; step++) {
        speed += (k * k * (gust[index] - lean) - 2 * damping * k * speed) * dt
        lean += speed * dt
      }
      if (lap === 2) {
        table[index] = lean
        peak = Math.max(peak, Math.abs(lean))
      }
    }
  if (peak > 0) for (let index = 0; index < samples; index++) table[index] /= peak
  return table
}

const cache = new Map<number, WindTables>()

// Gust, breeze and lean over one wind period, sampled by the share of the period that has passed.
export function windTables(period: number): WindTables {
  const key = Math.round(period * 100)
  let tables = cache.get(key)
  if (tables) return tables
  const gust = new Float32Array(samples)
  const breeze = new Float32Array(samples)
  for (let index = 0; index < samples; index++) {
    const u = (index + 0.5) / samples
    gust[index] = Math.min(1, field(u, period, 1))
    const air = Math.min(1, 1.6 * field(u, period, breezeSpread))
    // The tail of the bell never reaches zero; still air has to be exactly zero.
    breeze[index] = air < 0.04 ? 0 : air
  }
  tables = {
    gust,
    breeze,
    lean: {
      broadleaf: leanTable(gust, period, 'broadleaf'),
      conifer: leanTable(gust, period, 'conifer'),
    },
  }
  if (cache.size > 16) cache.clear()
  cache.set(key, tables)
  return tables
}

// Reads a table at u, the share of the period that has passed, in any range.
export function sampleWind(table: Float32Array, u: number) {
  return table[Math.floor((u - Math.floor(u)) * samples) % samples]
}

// Each front blows hardest along one band across the wind and weaker to either side, and no two
// fronts take the same band. `across` is the point's place across the wind, from 0 to 1.
export function frontStrength(front: number, across: number) {
  let hash = Math.imul(front | 0, 0x9e3779b1)
  hash = Math.imul(hash ^ (hash >>> 15), 0x85ebca6b)
  hash = (hash ^ (hash >>> 13)) >>> 0
  const centre = (hash & 1023) / 1023
  const peak = 0.9 + (((hash >>> 10) & 255) / 255) * 0.35
  const distance = (across - centre) / 0.3
  return 0.65 + (peak - 0.65) * Math.exp(-0.5 * distance * distance)
}

// A leaf clump walks a loop: rest, up, up and downwind, downwind, back. These are the two halves
// of the loop at an angle of the travelling wave, in cells for a clump at the crown's edge in the
// strongest wind.
export function loopAlong(angle: number, breeze: number, gust: number) {
  return (0.9 * breeze + 1.3 * gust) * (0.5 - 0.5 * Math.cos(angle))
}
export function loopUp(angle: number, breeze: number, gust: number) {
  return (0.75 * breeze + 0.5 * gust) * Math.max(Math.sin(angle), 0)
}
