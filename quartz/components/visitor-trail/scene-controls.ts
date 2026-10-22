import type { MotionTuning } from './motion-tuning'
import { debounce } from '../../util/debounce'
import { buildSelect, type SelectOption } from '../controls/select'
import {
  forestLimits,
  sceneSettings,
  type AnimalSettings,
  type CampSettings,
  type ForestSettings,
  type SceneSettings,
  type TravelerSettings,
} from './scene-settings'

export type SceneTuningGroup = 'forest' | 'animals' | 'travelers' | 'camps'
export type SceneTuningControls = {
  settings: SceneSettings
  change: (group: SceneTuningGroup) => void | Promise<void>
  motion: MotionTuning
  motionChange: () => void
  destinationChange?: (destination: number) => void
  pack?: () => void
}

type Field<K> = { key: K; label: string; name: string; suffix?: string; step?: number }
const forestFields: Field<keyof ForestSettings>[] = [
  { key: 'density', label: 'trees', name: 'Tree count' },
  { key: 'size', label: 'size', name: 'Tree size', suffix: '%', step: 5 },
  { key: 'depth', label: 'branches', name: 'Branch depth' },
  { key: 'angle', label: 'angle', name: 'Branch angle', suffix: '°' },
  { key: 'curl', label: 'curl', name: 'Branch curl', suffix: '°' },
  { key: 'branching', label: 'branch chance', name: 'Branching probability', suffix: '%', step: 5 },
  { key: 'variation', label: 'variation', name: 'Tree variation', suffix: '%', step: 5 },
  { key: 'dapple', label: 'dapple', name: 'Canopy dapple', suffix: '%', step: 5 },
]
const animalFields: Field<keyof AnimalSettings>[] = [
  { key: 'cow', label: 'cows', name: 'Cow count' },
  { key: 'dog', label: 'dogs', name: 'Dog count' },
  { key: 'cat', label: 'cats', name: 'Cat count' },
  { key: 'goose', label: 'geese', name: 'Goose count' },
  { key: 'bear', label: 'bears', name: 'Bear count' },
]
const animalMax: Record<keyof AnimalSettings, number> = {
  cow: 12,
  dog: 6,
  cat: 6,
  goose: 8,
  bear: 4,
}
const travelerFields: Field<keyof TravelerSettings>[] = [
  { key: 'walkers', label: 'walkers', name: 'Walker count' },
  { key: 'climbers', label: 'climbers', name: 'Climber count' },
  { key: 'alpinists', label: 'alpinists', name: 'Alpinist count' },
  { key: 'rangers', label: 'park patrol', name: 'Park ranger count' },
]

// HTML controls stay outside the retained SVG. Only committed geometry edits reach its owners.
export function mountSceneControls(
  scene: HTMLElement,
  panel: HTMLElement,
  signal: AbortSignal,
  controls: SceneTuningControls,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return
  const { settings } = controls
  const pending = new Map<SceneTuningGroup, Map<HTMLElement, () => void>>()
  const pointers = new Set<number>()
  const cleanups = new Set<() => void>()
  const resets: (() => void)[] = []
  const dirty = new Set<HTMLElement>()
  const reset = document.createElement('button')
  reset.type = 'button'
  reset.className = 'trail-tuning-action trail-tuning-reset'
  reset.textContent = 'reset tuning'
  reset.disabled = true
  const mark = (input: HTMLElement, changed: boolean) => {
    if (changed) dirty.add(input)
    else dirty.delete(input)
    reset.disabled = dirty.size === 0
  }
  const status = document.createElement('span')
  status.className = 'trail-tuning-status'
  status.setAttribute('role', 'status')
  panel.append(status)
  const failed = (event?: Event) => {
    if (signal.aborted) return
    status.classList.add('trail-tuning-error')
    status.textContent =
      event && 'detail' in event && typeof event.detail === 'string'
        ? event.detail
        : 'scene update failed; adjust the control to retry'
  }
  scene.addEventListener('trail:tuning-error', failed, { signal })
  let syncDestination = () => {}
  const applyPending = debounce(() => {
    if (signal.aborted || pointers.size || scene.dataset.trailOpening !== 'complete') return
    for (const [group, assignments] of pending) {
      for (const apply of assignments.values()) apply()
      pending.delete(group)
      if (group === 'camps') syncDestination()
      status.textContent = ''
      status.classList.remove('trail-tuning-error')
      void Promise.resolve()
        .then(() => controls.change(group))
        .catch(() => failed())
    }
    panel.dataset.tuningPending = 'false'
  }, 180)
  const defer = (group: SceneTuningGroup, input: HTMLElement, apply: () => void) => {
    const assignments = pending.get(group) ?? new Map<HTMLElement, () => void>()
    assignments.set(input, apply)
    pending.set(group, assignments)
    panel.dataset.tuningPending = 'true'
    applyPending()
  }
  const section = (name: string, open = false) => {
    const details = document.createElement('details')
    details.className = 'trail-tuning-section'
    details.open = open
    const summary = document.createElement('summary')
    summary.textContent = name
    details.append(summary)
    panel.append(details)
    return details
  }
  const range = (
    into: HTMLElement,
    field: Field<string>,
    value: number,
    defaultValue: number,
    limit: { min: number; max: number },
    changed: (value: number, input: HTMLInputElement) => void,
  ) => {
    const row = document.createElement('div')
    row.className = 'trail-tuning-row'
    const label = document.createElement('label')
    const input = document.createElement('input')
    const output = document.createElement('output')
    input.type = 'range'
    input.id = `trail-tuning-${field.key}-${crypto.randomUUID()}`
    input.min = String(limit.min)
    input.max = String(limit.max)
    input.step = String(field.step ?? 1)
    input.dataset.tuningField = field.key
    input.setAttribute('aria-label', field.name)
    label.htmlFor = input.id
    label.textContent = field.label
    output.htmlFor = input.id
    label.append(output)
    const readout = (next: number) => {
      input.value = String(next)
      mark(input, next !== defaultValue)
      output.value = `${next}${field.suffix ?? ''}`
      input.setAttribute('aria-valuetext', output.value)
      input.style.setProperty(
        '--trail-control-progress',
        `${((next - limit.min) / (limit.max - limit.min)) * 100}%`,
      )
    }
    readout(value)
    input.addEventListener(
      'input',
      () => {
        const next = Math.max(limit.min, Math.min(limit.max, input.valueAsNumber))
        if (!Number.isFinite(next)) return
        readout(next)
        changed(next, input)
      },
      { signal },
    )
    resets.push(() => {
      if (input.valueAsNumber === defaultValue) return
      readout(defaultValue)
      changed(defaultValue, input)
    })
    row.append(label, input)
    into.append(row)
  }
  const animation = section('animation', true)
  for (const field of [
    { key: 'wind', label: 'tree sway', name: 'Tree sway', suffix: '%', step: 5 },
    { key: 'leaves', label: 'falling leaves', name: 'Falling leaves', suffix: '%', step: 5 },
  ] satisfies Field<keyof MotionTuning>[]) {
    range(animation, field, controls.motion[field.key], 100, { min: 0, max: 200 }, value => {
      controls.motion[field.key] = value
      controls.motionChange()
    })
  }
  const forest = section('forest / l-system')
  const note = document.createElement('p')
  note.className = 'trail-tuning-note'
  note.textContent = 'shape changes apply on release'
  forest.append(note)
  for (const field of forestFields)
    range(
      forest,
      field,
      settings.forest[field.key],
      sceneSettings.forest[field.key],
      forestLimits[field.key],
      (value, input) =>
        defer('forest', input, () => {
          settings.forest[field.key] = value
        }),
    )
  const animals = section('animals')
  for (const field of animalFields)
    range(
      animals,
      field,
      settings.animals[field.key],
      sceneSettings.animals[field.key],
      { min: 0, max: animalMax[field.key] },
      (value, input) =>
        defer('animals', input, () => {
          settings.animals[field.key] = value
        }),
    )
  const camps = section('campsites')
  for (const field of [
    { key: 'sites', label: 'camps', name: 'Campsite count', min: 1, max: 3 },
    { key: 'tents', label: 'tents / camp', name: 'Tents per campsite', min: 0, max: 3 },
    { key: 'builders', label: 'builders / camp', name: 'Builders per campsite', min: 0, max: 2 },
  ] satisfies (Field<'sites' | 'tents' | 'builders'> & { min: number; max: number })[])
    range(
      camps,
      field,
      settings.camps[field.key],
      sceneSettings.camps[field.key],
      field,
      (value, input) =>
        defer('camps', input, () => {
          settings.camps[field.key] = value
        }),
    )
  const select = <T>(
    into: HTMLElement,
    label: string,
    name: string,
    options: SelectOption<T>[],
    value: T,
    changed: (value: T) => void,
  ) => {
    const row = document.createElement('div')
    row.className = 'trail-tuning-select'
    const text = document.createElement('span')
    text.textContent = label
    const choice = buildSelect({
      id: `trail-tuning-${crypto.randomUUID()}`,
      label: name,
      groups: [{ options }],
      selected: value,
      onSelect: changed,
      floating: true,
    })
    row.append(text, choice.element)
    into.append(row)
    const unmount = choice.mount()
    cleanups.add(unmount)
    return {
      ...choice,
      remove() {
        unmount()
        cleanups.delete(unmount)
        row.remove()
      },
    }
  }
  let lightValue = settings.camps.light
  const light = select<CampSettings['light']>(
    camps,
    'light',
    'Camp light',
    [
      { value: 'weather', label: 'local weather' },
      { value: 'day', label: 'day preview' },
      { value: 'night', label: 'night preview' },
    ],
    settings.camps.light,
    value => {
      lightValue = value
      mark(camps, value !== sceneSettings.camps.light)
      defer('camps', camps, () => {
        settings.camps.light = value
      })
    },
  )
  mark(camps, lightValue !== sceneSettings.camps.light)
  resets.push(() => {
    if (lightValue === sceneSettings.camps.light) return
    lightValue = sceneSettings.camps.light
    mark(camps, false)
    light.setSelected(sceneSettings.camps.light)
    defer('camps', camps, () => {
      settings.camps.light = sceneSettings.camps.light
    })
  })
  const destinationChange = controls.destinationChange
  if (destinationChange) {
    let destinationSites = 0
    let removeDestination = () => {}
    syncDestination = () => {
      if (destinationSites === settings.camps.sites) return
      destinationSites = settings.camps.sites
      removeDestination()
      const destination = Math.min(settings.destination, destinationSites - 1)
      const choice = select(
        camps,
        'visit',
        'Visit campsite',
        Array.from({ length: destinationSites }, (_, value) => ({
          value,
          label: `camp ${value + 1}`,
        })),
        destination,
        value => {
          settings.destination = value
          destinationChange(value)
        },
      )
      removeDestination = choice.remove
      if (destination !== settings.destination) {
        settings.destination = destination
        destinationChange(destination)
      }
    }
    syncDestination()
  }
  if (controls.pack) {
    const pack = document.createElement('button')
    pack.type = 'button'
    pack.className = 'trail-tuning-action'
    pack.textContent = 'pack up the visited camp'
    pack.addEventListener('click', controls.pack, { signal })
    camps.append(pack)
  }
  const travelers = section('people')
  for (const field of travelerFields)
    range(
      travelers,
      field,
      settings.travelers[field.key],
      sceneSettings.travelers[field.key],
      { min: 0, max: field.key === 'rangers' ? 3 : 6 },
      (value, input) =>
        defer('travelers', input, () => {
          settings.travelers[field.key] = value
        }),
    )
  reset.addEventListener(
    'click',
    () => {
      for (const restore of resets) restore()
      status.classList.remove('trail-tuning-error')
      status.textContent = 'tuning reset'
    },
    { signal },
  )
  panel.append(reset)
  panel.addEventListener(
    'pointerdown',
    event => {
      if (event.target instanceof view.HTMLInputElement) pointers.add(event.pointerId)
    },
    { signal },
  )
  const release = (event: PointerEvent) => {
    if (pointers.delete(event.pointerId) && pending.size) applyPending()
  }
  document.addEventListener('pointerup', release, { signal })
  document.addEventListener('pointercancel', release, { signal })
  view.addEventListener(
    'blur',
    () => {
      pointers.clear()
      if (pending.size) applyPending()
    },
    { signal },
  )
  panel.addEventListener(
    'keydown',
    event => {
      if (
        [
          'ArrowLeft',
          'ArrowRight',
          'ArrowUp',
          'ArrowDown',
          'Home',
          'End',
          'PageUp',
          'PageDown',
        ].includes(event.key)
      )
        event.stopPropagation()
    },
    { signal },
  )
  const opening = new MutationObserver(() => {
    if (pending.size) applyPending()
  })
  opening.observe(scene, { attributes: true, attributeFilter: ['data-trail-opening'] })
  signal.addEventListener(
    'abort',
    () => {
      applyPending.cancel()
      opening.disconnect()
      pending.clear()
      pointers.clear()
      for (const cleanup of cleanups) cleanup()
    },
    { once: true },
  )
}
