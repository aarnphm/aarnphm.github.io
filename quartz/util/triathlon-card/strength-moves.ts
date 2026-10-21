import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { dlabel, isImperial, LB_TO_KG } from './metric-text'

export type ActivityStrength = NonNullable<StravaActivityDetail['strength']>
export type ActivityStrengthExercise = ActivityStrength['exercises'][number]
export type ActivityStrengthSet = ActivityStrengthExercise['sets'][number]
export type ActivityMoves = NonNullable<StravaActivityDetail['moves']>
export type ActivityMove = ActivityMoves['entries'][number]

export const strengthMass = (presentation: TriathlonPresentation, kilograms: number): string => {
  const imperial = isImperial(presentation)
  const value = imperial ? kilograms / LB_TO_KG : kilograms
  const rounded = Math.round(value * 10) / 10
  return `${rounded.toLocaleString('en-US', { maximumFractionDigits: 1 })} ${imperial ? 'lb' : 'kg'}`
}

const strengthEffort = (presentation: TriathlonPresentation, set: ActivityStrengthSet): string => {
  const effort =
    set.repetitions != null
      ? `${set.repetitions} ${set.repetitions === 1 ? 'rep' : 'reps'}`
      : set.durationS != null
        ? dlabel(set.durationS)
        : 'set'
  return set.weightKg == null ? effort : `${effort} @ ${strengthMass(presentation, set.weightKg)}`
}

export const strengthExerciseSummary = (
  presentation: TriathlonPresentation,
  exercise: ActivityStrengthExercise,
): string => {
  const sets = `${exercise.setCount} ${exercise.setCount === 1 ? 'set' : 'sets'}`
  if (exercise.sets.length === 0) {
    if (exercise.repetitions != null)
      return `${sets} · ${exercise.repetitions} ${exercise.repetitions === 1 ? 'rep' : 'reps'}`
    if (exercise.durationS != null) return `${sets} · ${dlabel(exercise.durationS)}`
    return sets
  }
  const efforts = exercise.sets.map(set => strengthEffort(presentation, set))
  if (efforts.every(effort => effort === efforts[0])) return `${sets} · ${efforts[0]} each`
  return `${sets} · ${efforts.join(', ')}`
}

const activityMoveEffort = (move: ActivityMove, index: number): string => {
  const set = move.sets[index]
  return `${set.repetitions} ${set.repetitions === 1 ? 'rep' : 'reps'}${set.perSide ? ' per side' : ''}`
}

export const activityMoveSummary = (move: ActivityMove): string => {
  const sets = `${move.sets.length} ${move.sets.length === 1 ? 'set' : 'sets'}`
  const efforts = move.sets.map((_, index) => activityMoveEffort(move, index))
  if (efforts.every(effort => effort === efforts[0]))
    return `${sets} · ${efforts[0]}${efforts.length === 1 ? '' : ' each'}`
  return `${sets} · ${efforts.join(', ')}`
}

export const buildStrengthExercises = <N>(
  f: TriNodeFactory<N>,
  strength: ActivityStrength,
): N | null => {
  if (strength.exercises.length === 0) return null
  const wrap = f.el('section', 'tri-act-strength')
  f.add(wrap, f.el('h3', 'tri-act-strength-h', 'exercises'))
  const list = f.el('ol', 'tri-strength-exercises')
  for (const exercise of strength.exercises) {
    const item = f.el('li', 'tri-strength-exercise')
    f.add(
      item,
      f.el('span', 'tri-strength-exercise-name', exercise.name),
      f.el(
        'span',
        'tri-strength-exercise-summary',
        strengthExerciseSummary(f.presentation, exercise),
      ),
    )
    f.add(list, item)
  }
  f.add(wrap, list)
  return wrap
}

export const buildActivityMoves = <N>(f: TriNodeFactory<N>, moves: ActivityMoves): N | null => {
  if (moves.entries.length === 0) return null
  const wrap = f.el('section', 'tri-act-strength tri-act-moves')
  f.add(wrap, f.el('h3', 'tri-act-strength-h tri-act-moves-h', 'moves'))
  const list = f.el('ol', 'tri-strength-exercises tri-move-list')
  for (const move of moves.entries) {
    const item = f.el('li', 'tri-strength-exercise tri-move')
    f.add(
      item,
      f.el('span', 'tri-strength-exercise-name tri-move-name', move.name),
      f.el('span', 'tri-strength-exercise-summary tri-move-summary', activityMoveSummary(move)),
    )
    f.add(list, item)
  }
  f.add(wrap, list)
  return wrap
}
