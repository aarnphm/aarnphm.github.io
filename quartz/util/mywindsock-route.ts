import type { MyWindsockArchive } from './mywindsock-archive'

type Capture = NonNullable<MyWindsockArchive['browserCapture']>
type Mapping = MyWindsockArchive['mappings'][number]

export interface MyWindsockRouteSample {
  elapsedS: number
  distanceKm: number
  // Rider-height wind resolved on the course: + headwind, + crosswind from the right.
  providerHeadwindKph: number | null
  providerCrosswindKph: number | null
  // Extra power over the 200 W reference to hold the no-weather speed.
  weatherCostW: number | null
  // Air distance minus ground distance, summed over moving intervals only.
  movingAirPenaltyKm: number | null
}

export interface MyWindsockRoute {
  source: 'provider-native'
  transport: 'browser-runtime'
  provider: 'mywindsock'
  activityId: number
  capturedAt: string
  referenceWatts: number | null
  samples: MyWindsockRouteSample[]
  // Moving time by relative wind angle: eight 45° sectors from 0° (headwind), clockwise.
  relativeWindPct: number[]
}

const series = (capture: Capture, key: string): number[] | null => {
  const values = capture.runtime.series[key]
  if (!values?.length) return null
  const numbers: number[] = []
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null
    numbers.push(value)
  }
  return numbers
}

const radians = (degrees: number): number => (degrees * Math.PI) / 180

const MAXIMUM_INTERVAL_S = 30
const MINIMUM_GROUND_SPEED_KPH = 3.6
const MAXIMUM_SAMPLES = 320
const SECTOR_COUNT = 8

const PATH = {
  time: '/runtime/series/time',
  pointTime: '/runtime/series/point_time',
  distance: '/runtime/series/dist',
  cumulativeDistance: '/runtime/series/dist_acc',
  groundSpeed: '/runtime/series/groundspd',
  airDistance: '/runtime/series/airdist',
  riderWind: '/runtime/series/adj_wind',
  relativeWind: '/runtime/series/direction360',
  withWeatherWatts: '/runtime/series/with_weather_watts',
  noWeatherWatts: '/runtime/series/no_weather_watts',
} as const

const maxError = (count: number, error: (index: number) => number): number => {
  let maximum = 0
  for (let index = 0; index < count; index += 1) maximum = Math.max(maximum, error(index))
  return maximum
}

const mapping = (
  path: string,
  meaning: string,
  unit: string,
  multiplier: number,
  holds: boolean,
  evidence: string,
): Mapping => ({
  path,
  meaning,
  unit,
  multiplier,
  status: holds ? 'verified' : 'unresolved',
  evidence,
})

// Recomputes the provider identities that the route graphs depend on. Each record states the
// largest error over every sample, so a later capture with different semantics stays unresolved.
export function verifyMyWindsockRouteMappings(record: MyWindsockArchive): Mapping[] {
  const capture = record.browserCapture
  if (!capture) return []
  const time = series(capture, 'time')
  const pointTime = series(capture, 'point_time')
  const distance = series(capture, 'dist')
  const cumulativeDistance = series(capture, 'dist_acc')
  const groundSpeed = series(capture, 'groundspd')
  const airDistance = series(capture, 'airdist')
  const effective = series(capture, 'effective')
  const sidewind = series(capture, 'sidewind')
  const riderWind = series(capture, 'adj_wind')
  const windSpeed = series(capture, 'windspeed')
  const windBearing = series(capture, 'windbearing')
  const bearing = series(capture, 'bearing')
  const relativeWind = series(capture, 'direction360')
  const withWeather = series(capture, 'with_weather_watts')
  const noWeather = series(capture, 'no_weather_watts')
  const noWeatherTime = series(capture, 'no_weather_point_time')
  const cda = series(capture, 'cda')
  const count = time?.length ?? 0
  const aligned = [
    pointTime,
    distance,
    cumulativeDistance,
    groundSpeed,
    airDistance,
    effective,
    sidewind,
    riderWind,
    windSpeed,
    windBearing,
    bearing,
    relativeWind,
    withWeather,
    noWeather,
    noWeatherTime,
    cda,
  ].every(values => values?.length === count)
  if (
    !aligned ||
    count < 2 ||
    !time ||
    !pointTime ||
    !distance ||
    !cumulativeDistance ||
    !groundSpeed ||
    !airDistance ||
    !effective ||
    !sidewind ||
    !riderWind ||
    !windSpeed ||
    !windBearing ||
    !bearing ||
    !relativeWind ||
    !withWeather ||
    !noWeather ||
    !noWeatherTime ||
    !cda
  )
    return []
  const n = `${count} samples`
  const fixed = (value: number, digits = 2) => value.toExponential(digits)

  const clockError = maxError(count, i =>
    Math.abs(time[i] - (i === 0 ? 0 : time[i - 1]) - pointTime[i]),
  )
  const elapsedGapS = Math.abs(
    time[count - 1] * 3_600 - record.activity.stravaBaseline.elapsedTimeS,
  )
  let summed = 0
  const distanceError = maxError(count, i => {
    summed += distance[i]
    return Math.abs(summed - cumulativeDistance[i])
  })
  // groundspd is a recorded speed channel, so it differs from dist / point_time where GPS
  // distance jitters. Over short intervals its integral still reproduces the route distance.
  let integratedKm = 0
  let intervalKm = 0
  for (let i = 1; i < count; i += 1) {
    if (pointTime[i] * 3_600 > MAXIMUM_INTERVAL_S) continue
    integratedKm += groundSpeed[i] * pointTime[i]
    intervalKm += distance[i]
  }
  const speedRatio = intervalKm > 0 ? integratedKm / intervalKm : Number.NaN
  const airError = maxError(count, i => Math.abs(effective[i] * pointTime[i] - airDistance[i]))
  let penalty = 0
  let stationaryPenalty = 0
  for (let i = 1; i < count; i += 1) {
    const excess = airDistance[i] - distance[i]
    penalty += excess
    if (pointTime[i] * 3_600 > MAXIMUM_INTERVAL_S || groundSpeed[i] < MINIMUM_GROUND_SPEED_KPH)
      stationaryPenalty += excess
  }
  const ratios = riderWind.flatMap((value, i) => (windSpeed[i] > 0 ? [value / windSpeed[i]] : []))
  const ratioSpread = Math.max(...ratios) - Math.min(...ratios)
  const component = (i: number) => {
    const speedKph = riderWind[i] * 3.6
    const angle = radians(relativeWind[i])
    return { head: speedKph * Math.cos(angle), cross: speedKph * Math.sin(angle) }
  }
  const sideError = maxError(count, i => Math.abs(Math.abs(component(i).cross) - sidewind[i]))
  const angleError = maxError(count, i => {
    const delta = Math.abs(((((windBearing[i] - bearing[i]) % 360) + 360) % 360) - relativeWind[i])
    return Math.min(delta, 360 - delta)
  })
  const airSpeedError = maxError(count, i => {
    const { head, cross } = component(i)
    return Math.abs(Math.hypot(groundSpeed[i] + head, cross) - effective[i])
  })
  const reference = noWeather[0]
  const referenceSpread = maxError(count, i => Math.abs(noWeather[i] - reference))
  let weight = 0
  let weighted = 0
  for (let i = 0; i < count; i += 1) {
    weight += noWeatherTime[i]
    weighted += noWeatherTime[i] * (withWeather[i] - noWeather[i])
  }
  const reported = capture.runtime.averages.wwatts
  const impactError =
    reported == null || weight <= 0 ? null : Math.abs(weighted / weight - reported)
  const sortedCda = cda.toSorted((a, b) => a - b)
  const medianCda = sortedCda[Math.floor(sortedCda.length / 2)]
  // A rider's CdA is near 0.2-0.6 m²; the run model returns 8-16 m², so its watts describe no body.
  const humanCda = medianCda < 2

  return [
    mapping(
      PATH.time,
      'Cumulative analysis time',
      'h',
      3_600,
      clockError < 1e-9 && elapsedGapS <= 60,
      `Cumulative sum of point_time within ${fixed(clockError)} h on ${n}; ends ${elapsedGapS.toFixed(0)} s from the Strava elapsed time.`,
    ),
    mapping(
      PATH.pointTime,
      'Analysis interval duration, including recording pauses',
      'h',
      3_600,
      clockError < 1e-9,
      `Equals successive differences of time within ${fixed(clockError)} h on ${n}.`,
    ),
    mapping(
      PATH.distance,
      'Analysis interval ground distance',
      'km',
      1_000,
      distanceError < 1e-6,
      `Cumulative sum equals dist_acc within ${fixed(distanceError)} km on ${n}.`,
    ),
    mapping(
      PATH.cumulativeDistance,
      'Cumulative analysis distance',
      'km',
      1_000,
      distanceError < 1e-6,
      `Cumulative sum of dist within ${fixed(distanceError)} km; ends at ${cumulativeDistance[count - 1].toFixed(3)} km against Strava ${(record.activity.stravaBaseline.distanceM / 1_000).toFixed(3)} km.`,
    ),
    mapping(
      PATH.groundSpeed,
      'Ground speed',
      'km/h',
      1 / 3.6,
      Math.abs(speedRatio - 1) <= 0.05 && airSpeedError < 1e-5,
      `Integrated over intervals up to ${MAXIMUM_INTERVAL_S} s it gives ${(speedRatio * 100).toFixed(1)}% of dist; it is the ground term of effective (within ${fixed(airSpeedError)} km/h) on ${n}.`,
    ),
    mapping(
      PATH.airDistance,
      'Analysis interval air distance, |apparent air| x interval time',
      'km',
      1_000,
      airError < 1e-6,
      `Equals effective x point_time within ${fixed(airError)} km on ${n}. Of the ${penalty.toFixed(3)} km air minus ground distance, ${stationaryPenalty.toFixed(3)} km (${penalty > 0 ? ((stationaryPenalty / penalty) * 100).toFixed(0) : '0'}%) accrues in intervals over ${MAXIMUM_INTERVAL_S} s or below ${MINIMUM_GROUND_SPEED_KPH} km/h, where wind passes a stopped rider.`,
    ),
    mapping(
      PATH.riderWind,
      'Rider-height wind speed, the 10 m windspeed after myWindsock shear reduction',
      'm/s',
      1,
      ratioSpread < 1e-6 && sideError < 1e-5 && airSpeedError < 1e-5,
      `adj_wind / windspeed = ${ratios[0]?.toFixed(5)} (spread ${fixed(ratioSpread)}); |adj_wind x 3.6 x sin(direction360)| equals sidewind within ${fixed(sideError)} km/h on ${n}.`,
    ),
    mapping(
      PATH.relativeWind,
      'Wind-from angle relative to the course: 0° headwind, 90° from the right, clockwise',
      'deg',
      1,
      angleError < 1e-6 && airSpeedError < 1e-5,
      `(windbearing - bearing) mod 360 within ${fixed(angleError)}°; hypot(groundspd + adj_wind x 3.6 x cos, adj_wind x 3.6 x sin) equals effective within ${fixed(airSpeedError)} km/h on ${n}.`,
    ),
    mapping(
      PATH.noWeatherWatts,
      'Reference power for the no-weather speed',
      'W',
      1,
      referenceSpread === 0,
      `Constant ${reference} W on ${n}.`,
    ),
    mapping(
      PATH.withWeatherWatts,
      'Power that holds the no-weather speed under the analysed weather',
      'W',
      1,
      humanCda && impactError != null && impactError <= 1,
      `Mean of with_weather_watts - no_weather_watts weighted by no_weather_point_time is ${impactError == null ? 'unavailable' : `${(weighted / weight).toFixed(3)} W against the reported ${reported?.toFixed(3)} W`}. Model CdA median ${medianCda.toFixed(2)} m²${humanCda ? '' : ', outside a human range, so these watts describe no body'}.`,
    ),
  ]
}

const verified = (record: MyWindsockArchive, path: string): boolean =>
  record.mappings.some(item => item.path === path && item.status === 'verified')

// Projects verified provider fields onto at most 320 elapsed-time bins. Only moving intervals
// count: long intervals are recording pauses and slow ones have no defined course bearing.
export function projectMyWindsockRoute(record: MyWindsockArchive): MyWindsockRoute | null {
  const capture = record.browserCapture
  if (
    !capture ||
    ![
      PATH.time,
      PATH.pointTime,
      PATH.cumulativeDistance,
      PATH.groundSpeed,
      PATH.riderWind,
      PATH.relativeWind,
    ].every(path => verified(record, path))
  )
    return null
  const time = series(capture, 'time')
  const pointTime = series(capture, 'point_time')
  const cumulativeDistance = series(capture, 'dist_acc')
  const groundSpeed = series(capture, 'groundspd')
  const riderWind = series(capture, 'adj_wind')
  const relativeWind = series(capture, 'direction360')
  if (!time || !pointTime || !cumulativeDistance || !groundSpeed || !riderWind || !relativeWind)
    return null
  const count = time.length
  if (
    count < 2 ||
    [pointTime, cumulativeDistance, groundSpeed, riderWind, relativeWind].some(
      values => values.length !== count,
    )
  )
    return null
  const bike = record.activity.sport === 'bike'
  const cost =
    bike && verified(record, PATH.withWeatherWatts) && verified(record, PATH.noWeatherWatts)
      ? {
          withWeather: series(capture, 'with_weather_watts'),
          noWeather: series(capture, 'no_weather_watts'),
        }
      : null
  // Air Penalty on runs stays out of the graphs until its meaning for runners is settled.
  const penalty =
    bike && verified(record, PATH.airDistance) && verified(record, PATH.distance)
      ? { air: series(capture, 'airdist'), ground: series(capture, 'dist') }
      : null

  const endS = time[count - 1] * 3_600
  if (!(endS > 0)) return null
  const binCount = Math.min(MAXIMUM_SAMPLES, count)
  const binS = endS / binCount
  const bins = Array.from({ length: binCount }, () => ({
    weightS: 0,
    head: 0,
    cross: 0,
    costWeightS: 0,
    cost: 0,
    distanceKm: 0,
  }))
  const penaltyAtBin: number[] = Array.from({ length: binCount }, () => 0)
  const sectors = Array.from({ length: SECTOR_COUNT }, () => 0)
  let movingS = 0
  let movingPenaltyKm = 0
  for (let i = 0; i < count; i += 1) {
    const durationS = pointTime[i] * 3_600
    const elapsedS = time[i] * 3_600 - durationS / 2
    const bin = bins[Math.min(binCount - 1, Math.max(0, Math.floor(elapsedS / binS)))]
    bin.distanceKm = cumulativeDistance[i]
    if (
      i === 0 ||
      durationS <= 0 ||
      durationS > MAXIMUM_INTERVAL_S ||
      groundSpeed[i] < MINIMUM_GROUND_SPEED_KPH
    )
      continue
    const speedKph = riderWind[i] * 3.6
    const angle = radians(relativeWind[i])
    bin.weightS += durationS
    bin.head += durationS * speedKph * Math.cos(angle)
    bin.cross += durationS * speedKph * Math.sin(angle)
    const withWeather = cost?.withWeather?.[i]
    const noWeather = cost?.noWeather?.[i]
    if (withWeather != null && noWeather != null) {
      bin.costWeightS += durationS
      bin.cost += durationS * (withWeather - noWeather)
    }
    const air = penalty?.air?.[i]
    const ground = penalty?.ground?.[i]
    if (air != null && ground != null) movingPenaltyKm += air - ground
    penaltyAtBin[Math.min(binCount - 1, Math.floor(elapsedS / binS))] = movingPenaltyKm
    sectors[Math.round((((relativeWind[i] % 360) + 360) % 360) / 45) % SECTOR_COUNT] += durationS
    movingS += durationS
  }
  if (movingS <= 0) return null
  const round = (value: number, digits: number) => Number(value.toFixed(digits))
  let lastPenalty = 0
  let lastDistance = 0
  const samples = bins.map((bin, index): MyWindsockRouteSample => {
    const moving = bin.weightS > 0
    if (moving) lastPenalty = penaltyAtBin[index]
    lastDistance = Math.max(lastDistance, bin.distanceKm)
    return {
      elapsedS: round((index + 0.5) * binS, 1),
      distanceKm: round(lastDistance, 3),
      providerHeadwindKph: moving ? round(bin.head / bin.weightS, 1) : null,
      providerCrosswindKph: moving ? round(bin.cross / bin.weightS, 1) : null,
      weatherCostW: bin.costWeightS > 0 ? round(bin.cost / bin.costWeightS, 1) : null,
      // Cumulative: a stop holds the running total flat instead of opening a gap.
      movingAirPenaltyKm: penalty ? round(lastPenalty, 3) : null,
    }
  })
  return {
    source: 'provider-native',
    transport: 'browser-runtime',
    provider: 'mywindsock',
    activityId: Number(record.activity.stravaId),
    capturedAt: capture.capturedAt,
    referenceWatts: cost?.noWeather?.[0] ?? null,
    samples,
    relativeWindPct: sectors.map(seconds => round((seconds / movingS) * 100, 1)),
  }
}
