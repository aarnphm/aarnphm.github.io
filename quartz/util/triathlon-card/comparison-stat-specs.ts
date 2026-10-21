import type { ActivityKind } from '../../plugins/stores/strava'
import type { ActivityComparisonStat } from '../activity-comparison-metrics'
import type { TriathlonPresentation } from '../triathlon-presentation'
import {
  comparisonFixed,
  comparisonSigned,
  comparisonSignedClock,
  durationStat,
  numericStat,
  pairedSideStat,
  peakStat,
  type ComparisonStatSpec,
} from './comparison-stat-format'
import {
  activityCadenceUnit,
  clock,
  dur,
  elevationValue,
  formatMl,
  isImperial,
  KM_TO_MI,
  LB_TO_KG,
  M_TO_FT,
  temperatureUnit,
  temperatureValue,
} from './metric-text'

export const activityComparisonStatSpecs = (
  presentation: TriathlonPresentation,
): readonly ComparisonStatSpec[] => {
  const imperial = isImperial(presentation)
  const speedDisplay = (kph: number): number => (imperial ? kph * KM_TO_MI : kph)
  const speedUnit = imperial ? ' mph' : ' km/h'
  const paceSeconds = (kph: number, sport: ActivityKind): number =>
    sport === 'swim' ? 360 / kph : 3_600 / speedDisplay(kph)
  const paceUnit = (sport: ActivityKind): string =>
    sport === 'swim' ? ' /100m' : imperial ? ' /mi' : ' /km'
  const distanceText = (km: number, sport: ActivityKind): string =>
    sport === 'swim'
      ? `${comparisonFixed(km * 1_000, 0)} m`
      : `${comparisonFixed(imperial ? km * KM_TO_MI : km, 2)}${imperial ? ' mi' : ' km'}`
  const signedDistance = (km: number, sport: ActivityKind): string =>
    sport === 'swim'
      ? `${comparisonSigned(km * 1_000, 0)} m`
      : `${comparisonSigned(imperial ? km * KM_TO_MI : km, 2)}${imperial ? ' mi' : ' km'}`
  const temperature = (celsius: number): number => temperatureValue(presentation, celsius)
  const millilitres = (stat: ActivityComparisonStat): string => formatMl(stat.value)
  const millilitreDelta = (stat: ActivityComparisonStat, base: ActivityComparisonStat): string =>
    `${comparisonSigned(stat.value - base.value, 0)} ml`
  return [
    numericStat('tss', 'load', 'TSS', {
      gloss:
        '$\\mathrm{TSS} = h \\cdot \\mathrm{IF}^2 \\cdot 100$. Native from Wahoo or Garmin; a selection recomputes it from the 30 s power trace and FTP.',
    }),
    numericStat('intensity-factor', 'load', 'IF', {
      digits: 2,
      relative: false,
      gloss: '$\\mathrm{IF} = \\mathrm{NP} / \\mathrm{FTP}$',
    }),
    numericStat('aerobic-effect', 'load', 'aerobic TE', {
      digits: 1,
      relative: false,
      sameSourceDelta: true,
      gloss:
        'Garmin training effect on a 0–5 scale, or the garden estimate when Garmin did not report. Deltas compare same-source columns only.',
    }),
    numericStat('anaerobic-effect', 'load', 'anaerobic TE', {
      digits: 1,
      relative: false,
      sameSourceDelta: true,
      gloss:
        'Garmin anaerobic training effect on a 0–5 scale, or the garden estimate when Garmin did not report. Deltas compare same-source columns only.',
    }),
    numericStat('exercise-load', 'load', 'exercise load', {
      sameSourceDelta: true,
      gloss:
        'Garmin load from heart-rate EPOC, or the garden estimate from $\\mathrm{IF}$ and moving time when Garmin did not report. The two diverge on long easy rides, so deltas compare same-source columns only.',
    }),
    numericStat('relative-effort', 'load', 'relative effort'),
    durationStat('moving-time', 'session', 'moving time'),
    durationStat('elapsed-time', 'session', 'elapsed time', selection =>
      selection ? 'down' : undefined,
    ),
    {
      key: 'distance',
      group: 'session',
      label: () => 'distance',
      format: (stat, sport) => distanceText(stat.value, sport),
      delta: (stat, base, sport) => signedDistance(stat.value - base.value, sport),
      scalar: stat => stat.value,
      relative: true,
    },
    {
      key: 'speed',
      group: 'session',
      label: sport => (sport === 'bike' ? 'speed' : 'pace'),
      format: (stat, sport) =>
        sport === 'bike'
          ? `${comparisonFixed(speedDisplay(stat.value), 1)}${speedUnit}`
          : `${clock(paceSeconds(stat.value, sport))}${paceUnit(sport)}`,
      delta: (stat, base, sport) =>
        sport === 'bike'
          ? `${comparisonSigned(speedDisplay(stat.value) - speedDisplay(base.value), 1)}${speedUnit}`
          : comparisonSignedClock(paceSeconds(stat.value, sport) - paceSeconds(base.value, sport)),
      scalar: stat => stat.value,
      relativeScalar: (stat, sport) =>
        sport === 'bike' ? stat.value : paceSeconds(stat.value, sport),
      better: 'up',
      relative: true,
    },
    numericStat('max-speed', 'session', 'max speed', {
      digits: 1,
      unit: speedUnit,
      display: speedDisplay,
      relative: false,
    }),
    numericStat('elevation-gain', 'session', 'elevation gain', {
      unit: imperial ? ' ft' : ' m',
      display: meters => elevationValue(presentation, meters),
    }),
    numericStat('grade', 'session', 'net grade', {
      digits: 1,
      unit: '%',
      deltaUnit: ' pp',
      relative: false,

      gloss: 'Altitude change from the start to the end of the selection, over its distance.',
    }),
    numericStat('work', 'session', 'work', {
      unit: ' kJ',
      gloss: 'Mechanical work, $W = \\sum P \\, \\Delta t$.',
    }),
    numericStat('calories', 'session', 'calories', { unit: ' kcal' }),
    numericStat('average-power', 'power', 'avg power', { unit: ' W', better: 'up' }),
    numericStat('normalized-power', 'power', 'NP', {
      unit: ' W',
      better: 'up',
      gloss:
        '$\\mathrm{NP} = \\left(\\overline{P_{30}^{\\,4}}\\right)^{1/4}$, the fourth-power mean of 30 s rolling power. Strava-only rides carry Strava weighted average power.',
    }),
    numericStat('watts-per-kg', 'power', 'NP / kg', {
      digits: 2,
      unit: ' W/kg',
      better: 'up',
      gloss: 'NP over the Garmin weigh-in nearest the ride.',
    }),
    numericStat('variability-index', 'power', 'VI', {
      digits: 2,
      better: 'down',
      relative: false,

      gloss: '$\\mathrm{VI} = \\mathrm{NP} / \\bar{P}$. Values near 1 mean steady pacing.',
    }),
    numericStat('max-power', 'power', 'max power', { unit: ' W', better: 'up' }),
    numericStat('critical-power', 'power', 'CP', {
      unit: ' W',
      better: 'up',
      gloss: 'Critical power from a two-parameter fit to this activity’s power curve.',
    }),
    numericStat('w-prime', 'power', 'W′', {
      digits: 1,
      unit: ' kJ',
      better: 'up',
      gloss: 'Work capacity above CP from the same fit.',
    }),
    {
      key: 'power-balance',
      group: 'power',
      label: () => 'L / R',
      gloss: 'Power-weighted left and right split from the pedal balance samples.',
      format: stat => `${stat.value.toFixed(1)} / ${(100 - stat.value).toFixed(1)}`,
      delta: (stat, base) => `${comparisonSigned(stat.value - base.value, 1)} pp L`,
      scalar: stat => stat.value,
      better: 'balance',
      relative: false,
    },
    peakStat('peak-5s', 'peak 5 s'),
    peakStat('peak-30s', 'peak 30 s'),
    peakStat('peak-1m', 'peak 1 min'),
    peakStat('peak-5m', 'peak 5 min'),
    peakStat('peak-20m', 'peak 20 min'),
    peakStat('peak-60m', 'peak 60 min'),
    numericStat('average-hr', 'heart rate', 'avg hr', { unit: ' bpm', relative: false }),
    numericStat('max-hr', 'heart rate', 'max hr', { unit: ' bpm', relative: false }),
    numericStat('efficiency-factor', 'heart rate', 'EF', {
      digits: 2,
      better: 'up',
      gloss:
        'Output per heartbeat: $\\mathrm{NP} / \\overline{\\mathrm{HR}}$ on the bike, $v_{\\mathrm{m/min}} / \\overline{\\mathrm{HR}}$ on foot.',
    }),
    numericStat('decoupling', 'heart rate', sport => (sport === 'bike' ? 'Pw:HR' : 'Pa:HR'), {
      digits: 1,
      unit: '%',
      deltaUnit: ' pp',
      better: 'down',
      relative: false,

      gloss:
        '$(\\mathrm{EF}_1 - \\mathrm{EF}_2) / \\mathrm{EF}_1$ across two halves of at least 20 min. Under 5% reads as aerobically durable.',
    }),
    numericStat('cadence', 'technique', 'cadence', {
      unit: sport => ` ${activityCadenceUnit(sport)}`,
      relative: false,
    }),
    numericStat('stride-length', 'technique', 'stride length', {
      digits: 2,
      unit: imperial ? ' ft' : ' m',
      display: meters => (imperial ? meters * M_TO_FT : meters),
    }),
    numericStat('ground-contact-time', 'technique', 'ground contact time', {
      unit: ' ms',
      better: 'down',
      relative: false,
    }),
    numericStat('vertical-oscillation', 'technique', 'vertical oscillation', {
      digits: 1,
      unit: imperial ? ' in' : ' cm',
      display: centimeters => (imperial ? centimeters / 2.54 : centimeters),
      better: 'down',
      relative: false,
    }),
    numericStat('vertical-ratio', 'technique', 'vertical ratio', {
      digits: 1,
      unit: '%',
      deltaUnit: ' pp',
      better: 'down',
      relative: false,
    }),
    numericStat('stroke-rate', 'technique', 'stroke rate', {
      digits: 1,
      unit: ' str/min',
      relative: false,
    }),
    numericStat('strokes-per-length', 'technique', 'strokes / length', {
      digits: 1,
      better: 'down',
      relative: false,
    }),
    pairedSideStat(
      'torque-effectiveness',
      'torque effectiveness',
      'Left / right share of each pedal stroke that pushes forward.',
    ),
    pairedSideStat(
      'pedal-smoothness',
      'pedal smoothness',
      'Left / right mean over peak power within each pedal stroke.',
    ),
    numericStat('average-torque', 'technique', 'avg torque', { digits: 1, unit: ' N·m' }),
    numericStat('standing-time', 'technique', 'standing', {
      unit: '%',
      deltaUnit: ' pp',
      relative: false,
    }),
    numericStat('temperature', 'conditions', 'temperature', {
      unit: temperatureUnit(presentation),
      display: temperature,
      relative: false,
    }),
    numericStat('humidity', 'conditions', 'humidity', {
      unit: '%',
      deltaUnit: ' pp',
      relative: false,
    }),
    numericStat('wind', 'conditions', 'wind', {
      unit: speedUnit,
      display: speedDisplay,
      relative: false,
    }),
    numericStat('water-temperature', 'conditions', 'water temp', {
      digits: 1,
      unit: temperatureUnit(presentation),
      display: temperature,
      relative: false,
    }),
    numericStat('body-weight', 'athlete', 'weight', {
      digits: 1,
      unit: imperial ? ' lb' : ' kg',
      display: kg => (imperial ? kg / LB_TO_KG : kg),
      relative: false,
    }),
    numericStat('readiness', 'athlete', 'readiness', { better: 'up', relative: false }),
    numericStat('hrv', 'athlete', 'hrv', { unit: ' ms', better: 'up' }),
    numericStat('resting-hr', 'athlete', 'resting hr', {
      unit: ' bpm',
      better: 'down',
      relative: false,
    }),
    numericStat('sleep-score', 'athlete', 'sleep score', { better: 'up', relative: false }),
    {
      key: 'sleep-duration',
      group: 'athlete',
      label: () => 'slept',
      format: stat => dur(stat.value),
      delta: (stat, base) => {
        const minutes = Math.round((stat.value - base.value) / 60)
        return minutes === 0 ? '±0′' : `${minutes > 0 ? '+' : '−'}${dur(Math.abs(minutes) * 60)}`
      },
      scalar: stat => stat.value,
      better: 'up',
      relative: false,
    },
    numericStat('carbs', 'fueling', 'carbs', { unit: ' g' }),
    numericStat('carbs-per-hour', 'fueling', 'carbs / h', { unit: ' g/h', better: 'up' }),
    {
      key: 'fluid',
      group: 'fueling',
      label: () => 'fluid',
      format: millilitres,
      delta: millilitreDelta,
      scalar: stat => stat.value,
      relative: true,
    },
    {
      key: 'sweat-loss',
      group: 'fueling',
      label: () => 'sweat',
      format: millilitres,
      delta: millilitreDelta,
      scalar: stat => stat.value,
      relative: true,
    },
  ]
}
