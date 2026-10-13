import { holidayLabels, type HolidaySelection } from '../../util/visitor-holiday'
import { buildSelect, type Select } from '../controls/select'
import { defaultCampSettings, campLimits, type CampSettings } from './camp-settings'
import { animalLimits, defaultAnimalSettings, type AnimalSettings } from './pasture'
import { defaultForestSettings, forestLimits, type ForestSettings } from './scenery'
import { trailSettingsRocks } from './settings-rocks'
import { defaultTravelerSettings, travelerLimits, type TravelerSettings } from './traveler-settings'

const fields: { key: keyof ForestSettings; label: string; name: string; suffix: string }[] = [
  { key: 'density', label: 'trees', name: 'Tree count', suffix: '' },
  { key: 'size', label: 'size', name: 'Tree size', suffix: '%' },
  { key: 'depth', label: 'branches', name: 'Branch depth', suffix: '' },
  { key: 'angle', label: 'angle', name: 'Branch angle', suffix: '°' },
  { key: 'curl', label: 'curl', name: 'Branch curl', suffix: '°' },
  { key: 'branching', label: 'branch chance', name: 'Branching probability', suffix: '%' },
  { key: 'variation', label: 'variation', name: 'Tree variation', suffix: '%' },
  { key: 'dapple', label: 'dapple', name: 'Canopy dapple', suffix: '%' },
]

const animalFields: { key: keyof AnimalSettings; label: string; name: string }[] = [
  { key: 'cow', label: 'cows', name: 'Cow count' },
  { key: 'dog', label: 'dogs', name: 'Dog count' },
  { key: 'cat', label: 'cats', name: 'Cat count' },
  { key: 'goose', label: 'geese', name: 'Goose count' },
  { key: 'bear', label: 'bears', name: 'Bear count' },
]
const travelerFields: { key: keyof TravelerSettings; label: string; name: string }[] = [
  { key: 'walkers', label: 'walkers', name: 'Walker count' },
  { key: 'climbers', label: 'climbers', name: 'Climber count' },
  { key: 'alpinists', label: 'alpinists', name: 'Alpinist count' },
  { key: 'rangers', label: 'park patrol', name: 'Park ranger count' },
]
const campFields: { key: keyof typeof campLimits; label: string; name: string; suffix?: string }[] =
  [
    { key: 'sites', label: 'camps', name: 'Campsite count' },
    { key: 'tents', label: 'tents / camp', name: 'Tents per campsite' },
    { key: 'builders', label: 'builders / camp', name: 'Builders per campsite' },
  ]

type RangeLimit = { min: number; max: number; step: number }
const rangeProgress = ({ min, max }: RangeLimit, value: number) => {
  return `${((value - min) / (max - min)) * 100}%`
}

function rangeMarkup(
  id: string,
  group: 'forest' | 'animal' | 'traveler' | 'camp',
  field: { key: string; label: string; name: string; suffix?: string },
  value: number,
  limit: RangeLimit,
) {
  const { key, label, name, suffix = '' } = field
  const inputId = `${id}-${key}`
  return `<div class="trail-tree-control">
    <label for="${inputId}">${label}<output for="${inputId}" data-${group}-output="${key}">${value}${suffix}</output></label>
    <input type="range" id="${inputId}" aria-label="${name}" aria-valuetext="${value}${suffix}" min="${limit.min}" max="${limit.max}" step="${limit.step}" value="${value}" data-${group}-field="${key}" style="--trail-control-progress: ${rangeProgress(limit, value)}"/>
  </div>`
}

export function forestControlsMarkup(id: string) {
  return `
    <button class="trail-tree-toggle" type="button" aria-label="Trail settings" aria-expanded="false" aria-controls="${id}-forest">
      ${trailSettingsRocks()}
      <span class="trail-tree-hint" aria-hidden="true">settings</span>
    </button>
    <div class="trail-tree-panel" id="${id}-forest" role="group" aria-label="Trail settings" hidden>
      <div class="trail-tree-panel-head"><span>settings</span><span>grove</span><button class="trail-tree-close g-icon-button" type="button" aria-label="Close trail settings"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8M12 4l-8 8"/></svg></button></div>
      <div class="trail-tree-controls">
        <fieldset class="trail-control-section"><legend>forest / l-system</legend>
        ${fields
          .map(field =>
            rangeMarkup(
              id,
              'forest',
              field,
              defaultForestSettings[field.key],
              forestLimits[field.key],
            ),
          )
          .join('')}
        </fieldset>
        <fieldset class="trail-control-section"><legend>holidays</legend>
          <div class="trail-tree-control trail-camp-select" data-holiday-select="holiday">scene</div>
        </fieldset>
        <fieldset class="trail-control-section"><legend>animals</legend>
          ${animalFields
            .map(field =>
              rangeMarkup(
                id,
                'animal',
                field,
                defaultAnimalSettings[field.key],
                animalLimits[field.key],
              ),
            )
            .join('')}
        </fieldset>
        <fieldset class="trail-control-section"><legend>campsites</legend>
          ${campFields.map(field => rangeMarkup(id, 'camp', field, defaultCampSettings[field.key], campLimits[field.key])).join('')}
          <div class="trail-tree-control trail-camp-select" data-camp-select="light">light</div>
          <div class="trail-tree-control trail-camp-select" data-camp-select="visit">visit</div>
          <div class="trail-tree-control trail-camp-select">pack up<button type="button" class="trail-camp-join" data-camp-pack aria-label="Pack up the visited camp">pack</button></div>
          <span class="trail-status" data-camp-connection role="status">connecting</span>
        </fieldset>
        <fieldset class="trail-control-section"><legend>people</legend>
          ${travelerFields
            .map(field =>
              rangeMarkup(
                id,
                'traveler',
                field,
                defaultTravelerSettings[field.key],
                travelerLimits[field.key],
              ),
            )
            .join('')}
        </fieldset>
      </div>
    </div>`
}

export function mountForestControls(
  root: HTMLElement,
  signal: AbortSignal,
  onChange: (settings: ForestSettings) => void,
  onAnimalsChange: (settings: AnimalSettings) => void,
  onTravelersChange: (settings: TravelerSettings) => void,
  onCampsChange: (settings: CampSettings) => void,
  onHolidayChange: (selection: HolidaySelection) => void,
  onOpen: () => void,
) {
  const toggle = root.querySelector<HTMLButtonElement>('.trail-tree-toggle')
  const panel = root.querySelector<HTMLElement>('.trail-tree-panel')
  const dismiss = root.querySelector<HTMLButtonElement>('.trail-tree-close')
  if (!toggle || !panel || !dismiss) throw new Error('Trail settings are incomplete')
  const settings = { ...defaultForestSettings }
  const animals = { ...defaultAnimalSettings }
  const travelers = { ...defaultTravelerSettings }
  const camps = { ...defaultCampSettings }
  const close = () => {
    panel.hidden = true
    toggle.setAttribute('aria-expanded', 'false')
  }
  const dismissPanel = () => {
    close()
    toggle.focus({ preventScroll: true })
  }
  dismiss.addEventListener('click', dismissPanel, { signal })
  panel.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      dismissPanel()
    },
    { signal },
  )
  // Sliders and selects keep their navigation keys from reaching the page.
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
      ) {
        event.stopPropagation()
      }
    },
    { signal },
  )
  toggle.addEventListener(
    'click',
    () => {
      if (!panel.hidden) {
        close()
        return
      }
      panel.hidden = false
      toggle.setAttribute('aria-expanded', 'true')
      onOpen()
      dismiss.focus({ preventScroll: true })
    },
    { signal },
  )
  const mountRange = (
    group: 'forest' | 'animal' | 'traveler' | 'camp',
    key: string,
    limit: RangeLimit,
    suffix: string,
    changed: (value: number) => void,
  ) => {
    const input = panel.querySelector<HTMLInputElement>(`[data-${group}-field="${key}"]`)
    const output = panel.querySelector<HTMLOutputElement>(`[data-${group}-output="${key}"]`)
    if (!input || !output) throw new Error('Grove slider is incomplete')
    input.addEventListener(
      'input',
      () => {
        const value = Math.max(limit.min, Math.min(limit.max, Math.round(input.valueAsNumber)))
        output.value = `${value}${suffix}`
        input.setAttribute('aria-valuetext', output.value)
        input.style.setProperty('--trail-control-progress', rangeProgress(limit, value))
        changed(value)
      },
      { signal },
    )
  }
  const mountSelect = <T>(
    name: string,
    label: string,
    options: { value: T; label: string }[],
    selected: T,
    onSelect: (value: T) => void,
    group = 'camp-select',
  ) => {
    const host = panel.querySelector<HTMLElement>(`[data-${group}="${name}"]`)
    if (!host) throw new Error('Grove select is incomplete')
    const select: Select<T> = buildSelect({
      id: `${panel.id}-${name}`,
      label,
      groups: [{ options }],
      selected,
      onSelect,
    })
    host.querySelector('.g-select')?.remove()
    host.append(select.element)
    return select.mount()
  }
  const selections: HolidaySelection[] = [
    'auto',
    'none',
    'thanksgiving',
    'columbus-day',
    'halloween',
    'christmas',
    'easter',
  ]
  const unmountHoliday = mountSelect<HolidaySelection>(
    'holiday',
    'Holiday scene',
    selections.map(value => ({ value, label: holidayLabels[value] })),
    'auto',
    onHolidayChange,
    'holiday-select',
  )
  const unmountLight = mountSelect<CampSettings['light']>(
    'light',
    'Camp light',
    [
      { value: 'weather', label: 'local weather' },
      { value: 'day', label: 'day preview' },
      { value: 'night', label: 'night preview' },
    ],
    camps.light,
    value => {
      camps.light = value
      onCampsChange({ ...camps })
    },
  )
  let destination = 0
  const destinationListeners = new Set<() => void>()
  const chooseDestination = (value: number) => {
    if (value === destination) return
    destination = value
    for (const listener of destinationListeners) listener()
  }
  // The visit list follows the campsite count, so the campsite slider rebuilds it.
  let unmountDestination = () => {}
  const mountDestination = (sites: number) => {
    unmountDestination()
    const selected = Math.min(destination, sites - 1)
    unmountDestination = mountSelect(
      'visit',
      'Visit campsite',
      Array.from({ length: sites }, (_, index) => ({ value: index, label: `camp ${index + 1}` })),
      selected,
      chooseDestination,
    )
    chooseDestination(selected)
  }
  mountDestination(camps.sites)
  signal.addEventListener(
    'abort',
    () => {
      unmountLight()
      unmountDestination()
      unmountHoliday()
    },
    { once: true },
  )
  for (const { key, suffix } of fields)
    mountRange('forest', key, forestLimits[key], suffix, value => {
      settings[key] = value
      onChange({ ...settings })
    })
  for (const { key } of animalFields)
    mountRange('animal', key, animalLimits[key], '', value => {
      animals[key] = value
      onAnimalsChange({ ...animals })
    })
  for (const { key } of travelerFields)
    mountRange('traveler', key, travelerLimits[key], '', value => {
      travelers[key] = value
      onTravelersChange({ ...travelers })
    })
  for (const { key, suffix = '' } of campFields)
    mountRange('camp', key, campLimits[key], suffix, value => {
      camps[key] = value
      if (key === 'sites') mountDestination(value)
      onCampsChange({ ...camps })
    })
  return {
    toggle,
    close,
    isOpen: () => !panel.hidden,
    destination: {
      value: () => destination,
      onChange: (listener: () => void) => destinationListeners.add(listener),
    },
  }
}
