import fs from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
export type {
  ExportAppleMultisportTcxOptions,
  ExportAppleMultisportTcxResult,
  ExportedTcxFile,
  MultisportDiscipline,
} from './export-apple-multisport-tcx/conversion-model'
import type {
  ExportAppleMultisportTcxOptions,
  ExportAppleMultisportTcxResult,
  ExportedTcxFile,
  MultisportDiscipline,
} from './export-apple-multisport-tcx/conversion-model'
import {
  prepareActivity,
  prepareCourseActivity,
  retimedActivity,
  validatePrepared,
} from './export-apple-multisport-tcx/activity-preparation'
import { parseAppleMultisportTcxArgs } from './export-apple-multisport-tcx/cli-arguments'

export { parseAppleMultisportTcxArgs }
import { parseCourseRoute, parseGpx } from './export-apple-multisport-tcx/route-input'
import { MAX_FILE_BYTES, tcxXml } from './export-apple-multisport-tcx/tcx-output'
import { parseWorkout, workoutRecord } from './export-apple-multisport-tcx/workout-input'

function normalizedSports(
  sports: readonly MultisportDiscipline[] | undefined,
): MultisportDiscipline[] {
  const selected = sports ?? ['swim', 'bike', 'run']
  const unique: MultisportDiscipline[] = []
  for (const sport of selected) {
    if (sport !== 'swim' && sport !== 'bike' && sport !== 'run')
      throw new Error(`invalid sport ${String(sport)}`)
    if (!unique.includes(sport)) unique.push(sport)
  }
  if (unique.length === 0) throw new Error('at least one sport must be selected')
  return unique
}

function normalizedSwimDistance(value: number | undefined): number | null {
  if (value == null) return null
  if (!Number.isFinite(value) || value <= 0)
    throw new Error('swim distance override must be a positive finite number')
  return value
}

function normalizedDuration(value: number | undefined, label: string): number | null {
  if (value == null) return null
  if (!Number.isFinite(value) || value <= 0)
    throw new Error(`${label} must be a positive finite number`)
  return value
}

export async function exportAppleMultisportTcx(
  options: ExportAppleMultisportTcxOptions,
): Promise<ExportAppleMultisportTcxResult> {
  const inputPath = resolve(options.inputPath)
  const outputDir = resolve(options.outputDir)
  const sports = normalizedSports(options.sports)
  const swimDistanceM = normalizedSwimDistance(options.swimDistanceM)
  const swimDurationS = normalizedDuration(options.swimDurationS, 'swim duration override')
  const swimElapsedTimeS = normalizedDuration(
    options.swimElapsedTimeS,
    'swim elapsed time override',
  )
  if (swimDurationS != null && swimElapsedTimeS != null && swimElapsedTimeS < swimDurationS)
    throw new Error('swim elapsed time override must be at least the swim duration override')
  const transition1DurationS = normalizedDuration(
    options.transition1DurationS,
    'transition 1 duration override',
  )
  const includeTransitions = options.includeTransitions ?? false
  const document: unknown = JSON.parse(await fs.readFile(inputPath, 'utf8'))
  const workout = parseWorkout(workoutRecord(document, options.workoutId))
  const gpxPath = resolve(dirname(inputPath), workout.gpxFile)
  const route = parseGpx(await fs.readFile(gpxPath, 'utf8'))
  const [swimCourse, transition1Course] = await Promise.all([
    options.swimRoutePath ? parseCourseRoute(options.swimRoutePath) : null,
    options.transition1RoutePath ? parseCourseRoute(options.transition1RoutePath) : null,
  ])
  const sourceSwim = workout.legs[0]
  const outputSwim =
    swimDurationS == null && swimElapsedTimeS == null
      ? sourceSwim
      : retimedActivity(
          sourceSwim,
          sourceSwim.start.timeMs,
          swimDurationS ?? sourceSwim.durationS,
          workout.heartRate,
          swimElapsedTimeS ?? swimDurationS ?? sourceSwim.elapsedTimeS,
        )
  const preparedLegs = workout.legs.map(activity =>
    activity.discipline === 'swim' && swimCourse
      ? prepareCourseActivity(activity, outputSwim, swimCourse, workout.heartRate, swimDistanceM)
      : prepareActivity(
          activity.discipline === 'swim' ? outputSwim : activity,
          route,
          workout.heartRate,
          activity.discipline === 'swim' ? swimDistanceM : null,
        ),
  )
  if (!swimCourse && (swimDurationS != null || swimElapsedTimeS != null)) {
    const preparedSwim = preparedLegs[0]
    if (!preparedSwim) throw new Error('prepared swim is missing')
    preparedSwim.sourceActivity = sourceSwim
  }
  for (const activity of preparedLegs) validatePrepared(activity)
  const selected = preparedLegs
    .filter(
      activity =>
        activity.activity.kind === 'sport' && sports.includes(activity.activity.discipline),
    )
    .map(activity => ({ activity, filename: `${activity.activity.discipline}.tcx` }))
  if (includeTransitions) {
    for (const [index, transition] of workout.transitions.entries()) {
      const outputTransition =
        index === 0 && transition1DurationS != null
          ? retimedActivity(
              transition,
              outputSwim.end.timeMs,
              transition1DurationS,
              workout.heartRate,
            )
          : transition
      const activity =
        index === 0 && transition1Course
          ? prepareCourseActivity(
              transition,
              outputTransition,
              transition1Course,
              workout.heartRate,
            )
          : prepareActivity(outputTransition, route, workout.heartRate)
      activity.sourceActivity = transition
      validatePrepared(activity)
      selected.push({ activity, filename: `transition-${index + 1}.tcx` })
    }
  }
  const staged = selected.map(({ activity, filename }) => {
    return {
      activity,
      filename,
      path: join(outputDir, filename),
      xml: tcxXml(activity, workout.id),
    }
  })
  const files: ExportedTcxFile[] = staged.map(({ activity, filename, path }) => ({
    discipline: activity.activity.discipline,
    sport: activity.sport,
    filename,
    path,
    sourceActivityId: activity.sourceActivity.id,
    trackPointCount: activity.trackPoints.length,
    routePointCount: activity.routePointCount,
    usedHeartRateFallback: activity.usedHeartRateFallback,
    sourceDistanceM: activity.sourceActivity.distanceM,
    outputDistanceM: activity.distanceM,
    distanceOverrideM: activity.distanceOverrideM,
    sourceStart: activity.sourceActivity.start.iso,
    sourceEnd: activity.sourceActivity.end.iso,
    outputStart: activity.activity.start.iso,
    outputEnd: activity.activity.end.iso,
    outputDurationS: activity.activity.durationS,
    outputElapsedTimeS: activity.activity.elapsedTimeS,
    routeOverridePath: activity.routeOverridePath,
  }))
  const swim = sourceSwim
  const manifest = {
    version: 1,
    source: {
      input: inputPath,
      workoutId: workout.id,
      activity: workout.activity,
      start: workout.start.iso,
      end: workout.end.iso,
      gpxFile: workout.gpxFile,
    },
    selectedSports: sports,
    includeTransitions,
    heartRateTimestampRepairs: workout.heartRateTimestampRepairs,
    timingOverrides: [
      ...(swimDurationS == null && swimElapsedTimeS == null
        ? []
        : [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              sourceStart: swim.start.iso,
              sourceEnd: swim.end.iso,
              sourceDurationS: swim.durationS,
              outputStart: outputSwim.start.iso,
              outputEnd: outputSwim.end.iso,
              outputDurationS: outputSwim.durationS,
              outputElapsedTimeS: outputSwim.elapsedTimeS,
              provenance: 'cli:--swim-duration-s/--swim-elapsed-s',
            },
          ]),
      ...(transition1DurationS == null || !workout.transitions[0]
        ? []
        : [
            {
              discipline: 'transition',
              ordinal: 1,
              sourceActivityId: workout.transitions[0].id,
              sourceStart: workout.transitions[0].start.iso,
              sourceEnd: workout.transitions[0].end.iso,
              sourceDurationS: workout.transitions[0].durationS,
              outputStart: outputSwim.end.iso,
              outputEnd: new Date(
                outputSwim.end.timeMs + transition1DurationS * 1000,
              ).toISOString(),
              outputDurationS: transition1DurationS,
              provenance: 'cli:--transition-1-duration-s',
            },
          ]),
    ],
    routeOverrides: [
      ...(swimCourse
        ? [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              path: swimCourse.path,
              sourcePointCount: swimCourse.points.length,
              sourceDistanceM: swimCourse.distanceM,
              provenance: 'cli:--swim-route',
            },
          ]
        : []),
      ...(transition1Course && workout.transitions[0]
        ? [
            {
              discipline: 'transition',
              ordinal: 1,
              sourceActivityId: workout.transitions[0].id,
              path: transition1Course.path,
              sourcePointCount: transition1Course.points.length,
              sourceDistanceM: transition1Course.distanceM,
              provenance: 'cli:--transition-1-route',
            },
          ]
        : []),
    ],
    distanceOverrides:
      swimDistanceM == null
        ? []
        : [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              sourceDistanceM: swim.distanceM,
              outputDistanceM: swimDistanceM,
              provenance: 'cli:--swim-distance-m',
            },
          ],
    activities: workout.activities.map(activity => ({
      id: activity.id,
      kind: activity.kind,
      discipline: activity.discipline,
      activity: activity.activity,
      start: activity.start.iso,
      end: activity.end.iso,
      durationS: activity.durationS,
      elapsedTimeS: activity.elapsedTimeS,
      distanceM: activity.distanceM,
      activeEnergyKcal: activity.activeEnergyKcal,
      averageHeartRateBpm: activity.averageHeartRateBpm,
      averagePowerW: activity.averagePowerW,
      averageCadencePerMinute: activity.averageCadencePerMinute,
      lapCount: activity.lapCount,
    })),
    transitions: workout.transitions.map(transition => ({
      id: transition.id,
      activity: transition.activity,
      start: transition.start.iso,
      end: transition.end.iso,
      durationS: transition.durationS,
      elapsedTimeS: transition.elapsedTimeS,
      distanceM: transition.distanceM,
      activeEnergyKcal: transition.activeEnergyKcal,
      averageHeartRateBpm: transition.averageHeartRateBpm,
      averagePowerW: transition.averagePowerW,
      averageCadencePerMinute: transition.averageCadencePerMinute,
      lapCount: transition.lapCount,
    })),
    files: files.map(file => ({
      discipline: file.discipline,
      sport: file.sport,
      filename: file.filename,
      sourceActivityId: file.sourceActivityId,
      trackPointCount: file.trackPointCount,
      routePointCount: file.routePointCount,
      usedHeartRateFallback: file.usedHeartRateFallback,
      sourceDistanceM: file.sourceDistanceM,
      outputDistanceM: file.outputDistanceM,
      distanceOverrideM: file.distanceOverrideM,
      sourceStart: file.sourceStart,
      sourceEnd: file.sourceEnd,
      outputStart: file.outputStart,
      outputEnd: file.outputEnd,
      outputDurationS: file.outputDurationS,
      outputElapsedTimeS: file.outputElapsedTimeS,
      routeOverridePath: file.routeOverridePath,
    })),
  }
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`
  if (Buffer.byteLength(manifestText) >= MAX_FILE_BYTES)
    throw new Error('manifest is 25MB or larger')
  const manifestPath = join(outputDir, 'manifest.json')
  await fs.mkdir(outputDir, { recursive: true })
  await Promise.all([
    ...staged.map(file => fs.writeFile(file.path, file.xml, 'utf8')),
    fs.writeFile(manifestPath, manifestText, 'utf8'),
  ])
  return { workoutId: workout.id, outputDir, manifestPath, files }
}

async function main(): Promise<void> {
  const args = parseAppleMultisportTcxArgs(process.argv.slice(2))
  const result = await exportAppleMultisportTcx(args)
  for (const file of result.files)
    console.log(
      `[apple-tcx] ${file.discipline} ${file.trackPointCount} trackpoints -> ${file.path}`,
    )
  console.log(`[apple-tcx] manifest -> ${result.manifestPath}`)
}

const entry = process.argv[1]
if (entry && import.meta.url === pathToFileURL(resolve(entry)).href) {
  main().catch(error => {
    console.error(`[apple-tcx] failed: ${error instanceof Error ? error.message : error}`)
    process.exitCode = 1
  })
}
