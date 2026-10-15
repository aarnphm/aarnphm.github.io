import type { HolidaySelection, TrailHoliday } from './visitor-holiday'

// A visitor's holiday choice: one holiday, 'none' for no decorations, or null for the calendar's.
export type TrailHolidayChoice = Exclude<HolidaySelection, 'auto'>

export const trailHolidayChoices: readonly TrailHolidayChoice[] = [
  'thanksgiving',
  'columbus-day',
  'halloween',
  'christmas',
  'easter',
  'none',
]

const key = 'trail-holiday'

type HolidayChoiceEvents = { 'trail:holiday-choice': CustomEvent<TrailHolidayChoice | null> }

declare global {
  interface CustomEventMap extends HolidayChoiceEvents {}
  interface DocumentEventMap extends HolidayChoiceEvents {}
}

export function isTrailHolidayChoice(value: unknown): value is TrailHolidayChoice {
  return trailHolidayChoices.includes(value as TrailHolidayChoice)
}

export function isTrailHoliday(value: unknown): value is TrailHoliday {
  return isTrailHolidayChoice(value) && value !== 'none'
}

export function readTrailHolidayChoice(view: Window | null | undefined): TrailHolidayChoice | null {
  try {
    const value = view?.localStorage.getItem(key)
    return isTrailHolidayChoice(value) ? value : null
  } catch {
    return null
  }
}

export function setTrailHolidayChoice(document: Document, choice: TrailHolidayChoice | null) {
  try {
    const storage = document.defaultView?.localStorage
    if (choice) storage?.setItem(key, choice)
    else storage?.removeItem(key)
  } catch {
    // Private windows and blocked storage keep the choice for the page only.
  }
  document.dispatchEvent(new CustomEvent('trail:holiday-choice', { detail: choice }))
}

export function subscribeTrailHolidayChoice(
  document: Document,
  listener: (choice: TrailHolidayChoice | null) => void,
  signal: AbortSignal,
) {
  document.addEventListener('trail:holiday-choice', event => listener(event.detail), { signal })
}
