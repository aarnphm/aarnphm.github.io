import type { ActivityStrength } from '../plugins/stores/strava/activity-types'
import type { StrengthExercise } from '../plugins/stores/tracking'
import {
  STRENGTH_EXERCISE_ALIASES,
  STRENGTH_ROUTINES,
  type StrengthRoutine,
} from './strength-routines'

interface Circuit {
  warmup: StrengthExercise[]
  exercises: string[]
  cooldown: StrengthExercise
  rounds: number
  workS: number
  restS: number
  optionalRoundRestS?: number
}

export const normalizeStrengthRoutineTitle = (value: string): string =>
  value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/^\s*\([a-z]\)\s*/, '')
    .replace(/^focus:\s*/, '')
    .replace(/&/g, ' and ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()

function exerciseKey(value: string): string {
  const key = normalizeStrengthRoutineTitle(
    value.replace(/\([^)]*\)/g, '').replace(/,\s*soft landing$/i, ''),
  )
    .replace(/\band\b/g, '')
    .replace(/\b(bridge|raise|lunge|hinge|squat|dog)s\b/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
  return key === 'plank' ? 'forearm plank' : key
}

const exerciseNames = new Map(
  STRENGTH_ROUTINES.flatMap(routine =>
    [...routine.warmup, ...routine.exercises, routine.cooldown].map(
      name => [exerciseKey(name), name] satisfies [string, string],
    ),
  ),
)
for (const [alias, name] of Object.entries(STRENGTH_EXERCISE_ALIASES))
  exerciseNames.set(exerciseKey(alias), name)

function timedExercise(instructions: string, seconds: number, count = 1): StrengthExercise {
  const shortName = instructions.split(/\s+[—–]\s+/)[0].trim()
  const name = exerciseNames.get(exerciseKey(shortName)) ?? shortName
  return {
    name,
    ...(instructions !== name ? { instructions } : {}),
    setCount: count,
    repetitions: null,
    durationS: seconds * count,
    sets: Array.from({ length: count }, () => ({
      repetitions: null,
      durationS: seconds,
      weightKg: null,
    })),
  }
}

function duration(value: string): number {
  const match = /^(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m)\b/i.exec(value.trim())
  return match ? Number(match[1]) * (/^m/i.test(match[2]) ? 60 : 1) : 0
}

function parseDescription(description: string): Circuit | null {
  const lines = description
    .split(/\r?\n/)
    .map(line => line.trim().replace(/^[•*#-]+\s*/, ''))
    .filter(Boolean)
  const warmups = lines.flatMap((line, index) => (/^warm[-\s]?up\b/i.test(line) ? [index] : []))
  const mains = lines.flatMap((line, index) => (/^main\s+circuit\b/i.test(line) ? [index] : []))
  const cooldowns = lines.flatMap((line, index) => (/^cool[-\s]?down\b/i.test(line) ? [index] : []))
  if (warmups.length !== 1 || mains.length !== 1 || cooldowns.length !== 1) return null
  const [warmupIndex] = warmups
  const [mainIndex] = mains
  const [cooldownIndex] = cooldowns
  if (!(warmupIndex < mainIndex && mainIndex < cooldownIndex)) return null
  const warmupHeader = /^warm[-\s]?up\s*\(([^)]+)\)\s*$/i.exec(lines[warmupIndex])
  const main = /\b(\d+)\s+rounds?\s*,\s*(\d+)\s*s\s+work\s*\/\s*(\d+)\s*s\s+rest\b/i.exec(
    lines[mainIndex],
  )
  const cooldownHeader = /^cool[-\s]?down\s*\(([^)]+)\)\s*[-–—:]?\s*(.*)$/i.exec(
    lines[cooldownIndex],
  )
  if (!warmupHeader || !main || !cooldownHeader) return null
  const rounds = Number(main[1])
  const workS = Number(main[2])
  const restS = Number(main[3])
  const warmupS = duration(warmupHeader[1])
  const cooldownS = duration(cooldownHeader[1])
  if (
    rounds < 1 ||
    rounds > 20 ||
    workS < 1 ||
    workS > 600 ||
    restS > 600 ||
    !warmupS ||
    !cooldownS
  )
    return null
  const warmup: StrengthExercise[] = []
  for (const line of lines.slice(warmupIndex + 1, mainIndex)) {
    const seconds = duration(line)
    const name = line.replace(/^\d+\s*(?:seconds?|secs?|s|minutes?|mins?|m)\b\s*/i, '')
    if (!seconds || !name || seconds > 600) return null
    warmup.push(timedExercise(name, seconds))
  }
  if (warmup.reduce((sum, exercise) => sum + (exercise.durationS ?? 0), 0) !== warmupS) return null
  const exercises = lines.slice(mainIndex + 1, cooldownIndex)
  if (
    !exercises.length ||
    exercises.length > 20 ||
    exercises.some(line => /^(?:\d+\s*(?:s|reps?|x)\b|rest\b|round\b)/i.test(line))
  )
    return null
  const cooldownName = [cooldownHeader[2], ...lines.slice(cooldownIndex + 1)]
    .filter(Boolean)
    .join(', ')
  if (!cooldownName) return null
  const optionalRest = /additional\s+(\d+)\s*(?:sec|s)\s+between rounds if needed/i.exec(
    lines[mainIndex],
  )
  return {
    warmup,
    exercises,
    cooldown: timedExercise(cooldownName, cooldownS),
    rounds,
    workS,
    restS,
    ...(optionalRest ? { optionalRoundRestS: Number(optionalRest[1]) } : {}),
  }
}

function routineCircuit(routine: StrengthRoutine): Circuit {
  return {
    ...routine,
    warmup: routine.warmup.map(name => timedExercise(name, 30)),
    exercises: [...routine.exercises],
    cooldown: timedExercise(routine.cooldown, 120),
  }
}

export function inferStrengthPrescription(
  title: string,
  description?: string | null,
): ActivityStrength | null {
  if ((description?.length ?? 0) > 32_000) return null
  let circuit = description ? parseDescription(description) : null
  let source: ActivityStrength['source'] = 'strava-description'
  if (!circuit) {
    // A changed or incomplete circuit must never silently inherit an older title's defaults.
    if (description && /warm[-\s]?up|main\s+circuit|cool[-\s]?down/i.test(description)) return null
    const matches = STRENGTH_ROUTINES.filter(
      routine =>
        normalizeStrengthRoutineTitle(routine.title) === normalizeStrengthRoutineTitle(title) ||
        (description &&
          [routine.title, routine.focus].some(
            value =>
              normalizeStrengthRoutineTitle(value) === normalizeStrengthRoutineTitle(description),
          )),
    )
    if (matches.length !== 1) return null
    circuit = routineCircuit(matches[0])
    source =
      normalizeStrengthRoutineTitle(matches[0].title) === normalizeStrengthRoutineTitle(title)
        ? 'routine-title'
        : 'routine-description'
  }
  const exercises = [
    ...circuit.warmup,
    ...circuit.exercises.map(name => timedExercise(name, circuit.workS, circuit.rounds)),
    circuit.cooldown,
  ]
  const durationS =
    circuit.warmup.reduce((sum, exercise) => sum + (exercise.durationS ?? 0), 0) +
    circuit.exercises.length * circuit.rounds * (circuit.workS + circuit.restS) +
    (circuit.cooldown.durationS ?? 0)
  return {
    source,
    exercises,
    volumeKg: null,
    totalSets: null,
    totalReps: null,
    prescription: {
      rounds: circuit.rounds,
      workS: circuit.workS,
      restS: circuit.restS,
      durationS,
      ...(circuit.optionalRoundRestS ? { optionalRoundRestS: circuit.optionalRoundRestS } : {}),
    },
  }
}
