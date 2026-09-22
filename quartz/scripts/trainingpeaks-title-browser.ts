import {
  trainingPeaksDuration,
  trainingPeaksSaunaCandidates,
  trainingPeaksSaunaDescription,
  trainingPeaksStartMatches,
  type TrainingPeaksSaunaSource,
  type TrainingPeaksWorkoutSummary,
} from '../util/trainingpeaks-title-sync'

export interface TrainingPeaksTitleResult {
  stravaId: number
  date: string
  workoutId?: string
  title: string
  status: 'updated' | 'unchanged' | 'planned' | 'unmatched' | 'ambiguous' | 'skipped'
  reason?: string
}

export interface TrainingPeaksTitleRun {
  running: boolean
  results: TrainingPeaksTitleResult[]
  error?: string
}

declare global {
  interface Window {
    gardenTrainingPeaksTitles?: TrainingPeaksTitleRun
  }
}

const CARD = '.MuiCard-root.activity.workout[data-workoutid]'
const TITLE = 'input[placeholder="Untitled Workout"]'
const DESCRIPTION = '#descriptionInput'

function element<T extends Element>(selector: string, type: { new(): T }): T {
  const found = document.querySelector(selector)
  if (!(found instanceof type)) throw new Error(`TrainingPeaks control missing: ${selector}`)
  return found
}

async function until(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 15_000
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('TrainingPeaks did not finish opening or saving the workout')
    await new Promise(resolve => setTimeout(resolve, 100))
  }
}

function workouts(): TrainingPeaksWorkoutSummary[] {
  return Array.from(document.querySelectorAll(CARD)).flatMap(card => {
    const durationS = trainingPeaksDuration(card.querySelector('.duration')?.textContent ?? '')
    const id = card.getAttribute('data-workoutid')
    const date = card.closest('.day')?.getAttribute('data-date')
    if (!id || !date || durationS == null) return []
    const distance = card.querySelector('.distance .value')?.textContent?.trim()
    return [{
      id, date, durationS,
      title: card.querySelector('h6')?.textContent?.trim() ?? '',
      sport: card.classList.contains('Other') ? 'Other' : '',
      distance: distance == null ? null : Number(distance),
      planned: /\bP:\s/.test(card.textContent ?? ''),
    }]
  })
}

function button(name: string): HTMLButtonElement {
  const buttons = Array.from(document.querySelectorAll('button'))
    .filter(button => button.textContent?.trim() === name && button.getClientRects().length > 0)
  if (buttons.length !== 1 || buttons[0].disabled) throw new Error(`TrainingPeaks button unavailable: ${name}`)
  return buttons[0]
}

async function open(workout: TrainingPeaksWorkoutSummary): Promise<void> {
  element(`.workoutDiv[data-workoutid="${workout.id}"]`, HTMLElement).click()
  await until(() => document.querySelector(TITLE) != null)
}

async function close(name: string): Promise<void> {
  button(name).click()
  await until(() => document.querySelector(TITLE) == null)
}

function readDescription(): string {
  return element(DESCRIPTION, HTMLElement).innerText.replace(/\r\n?/g, '\n').trim()
}

function edit(source: TrainingPeaksSaunaSource, description: string): void {
  const title = element(TITLE, HTMLInputElement)
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
  if (!setter) throw new Error('Browser input setter unavailable')
  setter.call(title, source.title)
  title.dispatchEvent(new Event('input', { bubbles: true }))
  title.dispatchEvent(new Event('change', { bubbles: true }))
  const field = element(DESCRIPTION, HTMLElement)
  field.innerText = description
  field.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText' }))
  // TrainingPeaks commits its contenteditable description on keyup.
  field.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ' ' }))
  field.dispatchEvent(new Event('change', { bubbles: true }))
  field.blur()
}

async function sync(sources: readonly TrainingPeaksSaunaSource[], write: boolean, state: TrainingPeaksTitleRun): Promise<void> {
  if (location.origin !== 'https://app.trainingpeaks.com' || !location.hash.startsWith('#calendar'))
    throw new Error('Open the signed-in TrainingPeaks calendar first')
  if (document.querySelector(TITLE)) throw new Error('Close the open workout editor before syncing')
  const inventory = workouts()
  if (!inventory.length) throw new Error('No completed workouts loaded in the TrainingPeaks calendar')
  const claimed = new Set<string>()
  for (const source of sources) {
    const result: TrainingPeaksTitleResult = { stravaId: source.stravaId, date: source.date, title: source.title, status: 'unmatched' }
    const candidates = trainingPeaksSaunaCandidates(source, inventory)
    if (candidates.length !== 1) {
      result.status = candidates.length ? 'ambiguous' : 'unmatched'
      result.reason = candidates.length ? 'More than one workout matches the date and duration' : 'No matching cardio workout loaded in the calendar'
      state.results.push(result)
      continue
    }
    const workout = candidates[0]
    result.workoutId = workout.id
    const competing = sources.filter(other => trainingPeaksSaunaCandidates(other, [workout]).length > 0)
    if (competing.length !== 1 || claimed.has(workout.id)) {
      result.status = 'ambiguous'
      result.reason = 'Multiple Strava activities match this workout'
      state.results.push(result)
      continue
    }
    claimed.add(workout.id)
    await open(workout)
    const time = element('input[placeholder="Enter Time"]', HTMLInputElement).value
    const existing = readDescription()
    const otherSource = [...existing.matchAll(/strava\.com\/activities\/(\d+)/g)].some(match => Number(match[1]) !== source.stravaId)
    if (!trainingPeaksStartMatches(source, time) || otherSource) {
      result.status = 'skipped'
      result.reason = otherSource ? 'Description links to another Strava activity' : `Start time differs: TrainingPeaks ${time}, Strava ${source.startTime}`
      await close('Cancel')
    } else {
      const description = trainingPeaksSaunaDescription(source, existing)
      if (element(TITLE, HTMLInputElement).value === source.title && existing === description) {
        result.status = 'unchanged'
        await close('Cancel')
      } else if (!write) {
        result.status = 'planned'
        await close('Cancel')
      } else {
        edit(source, description)
        await close('Save & Close')
        await open(workout)
        if (element(TITLE, HTMLInputElement).value !== source.title || readDescription() !== description)
          throw new Error(`TrainingPeaks readback failed for workout ${workout.id}`)
        await close('Cancel')
        result.status = 'updated'
      }
    }
    state.results.push(result)
  }
}

export function startTrainingPeaksTitleSync(sources: readonly TrainingPeaksSaunaSource[], write: boolean): void {
  if (window.gardenTrainingPeaksTitles?.running) throw new Error('A TrainingPeaks title sync is already running')
  const state: TrainingPeaksTitleRun = { running: true, results: [] }
  window.gardenTrainingPeaksTitles = state
  void sync(sources, write, state).catch((error: unknown) => {
    state.error = error instanceof Error ? error.message : String(error)
  }).finally(() => { state.running = false })
}
