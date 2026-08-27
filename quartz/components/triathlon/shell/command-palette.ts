import type { TriathlonContext } from '../runtime/context'
import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressurePsi,
  formatTirePressureWeight,
  isTirePressureBalanceId,
  isTirePressureBikeId,
  isTirePressureBikeMassLb,
  isTirePressureInnerWidthMm,
  isTirePressureMeasuredWidthMm,
  isTirePressureRiderKg,
  isTirePressureSpeed,
  isTirePressureSurfaceId,
  isTirePressureTireId,
  isTirePressureWheelId,
  isTirePressureWeightUnit,
  latestMorningBodyWeight,
  selectedTirePressureWheel,
  tirePressureBalance,
  tirePressureBike,
  tirePressureSurface,
  tirePressureTire,
  tirePressureWeightToKg,
  TIRE_PRESSURE_BIKES,
  TIRE_PRESSURE_BALANCES,
  TRI_TIRE_PRESSURE_CHANGE_EVENT,
  TRI_TIRE_PRESSURE_OPEN_EVENT,
  TIRE_PRESSURE_SPEEDS,
  TIRE_PRESSURE_SURFACES,
  TIRE_PRESSURE_TIRES,
  TIRE_PRESSURE_WHEELS,
  TIRE_PRESSURE_WEIGHT_UNITS,
  type TirePressureChange,
  type TirePressureSelection,
} from '../../../util/triathlon-tire-pressure'
import { el } from '../runtime/dom'
import { nextTriMapStyle } from '../runtime/preferences'
import { toggleTriMapStyle } from '../runtime/preferences'
import { toggleTriPanelsFullscreen } from '../runtime/preferences'
import { toggleTriPowerFilter } from '../runtime/preferences'
import { toggleTriUnit } from '../runtime/preferences'
import { readTirePressureSelection, storeTirePressureSelection } from '../tools/tire-pressure'

export const TRI_PAGES: { path: string; label: string; hint: string }[] = [
  { path: '/triathlon', label: 'triathlon', hint: 'overview' },
  { path: '/triathlon/tools', label: 'tools', hint: 'gears' },
  { path: '/triathlon/calc', label: 'calculator', hint: 'race · gears · PSI' },
  { path: '/triathlon/analytics', label: 'analytics', hint: 'charts' },
  { path: '/triathlon/maps', label: 'maps', hint: 'routes' },
  { path: '/triathlon/training', label: 'training', hint: 'plans' },
  { path: '/triathlon/feed', label: 'feed', hint: 'all activities' },
  { path: '/triathlon/on', label: 'on', hint: 'by date' },
]

export type SearchShortcut = { view: string; openClass?: string; search: string }

export const TRI_SEARCH_SHORTCUTS: SearchShortcut[] = [
  { view: 'analytics', openClass: 'tri-analytics-open', search: '.tri-analytics .tri-ana-search' },
  { view: 'maps', openClass: 'tri-map-open', search: '.tri-map .tri-map-search' },
  {
    view: 'training',
    openClass: 'tri-training-open',
    search: '.tri-training .tri-training-search',
  },
  { view: 'feed', search: '.tri-feed .tri-feed-search' },
]

export const isEditable = (el: HTMLElement): boolean => {
  const tag = el.tagName.toLowerCase()
  return (
    tag === 'input' ||
    tag === 'textarea' ||
    tag === 'select' ||
    el.isContentEditable ||
    el.closest('.search-container') !== null
  )
}

export const currentSearchShortcut = (root: HTMLElement): SearchShortcut | undefined => {
  const subView = root.dataset.triView
  if (subView) return TRI_SEARCH_SHORTCUTS.find(shortcut => shortcut.view === subView)
  return TRI_SEARCH_SHORTCUTS.find(
    shortcut => shortcut.openClass && root.classList.contains(shortcut.openClass),
  )
}

export const toggleSearchFocus = (root: HTMLElement, target: HTMLElement | null): boolean => {
  const shortcut = currentSearchShortcut(root)
  if (!shortcut) return false
  const search = root.querySelector<HTMLInputElement>(shortcut.search)
  if (!search) return false
  if (target && isEditable(target) && target !== search) return false
  if (document.activeElement === search) search.blur()
  else {
    search.focus()
    search.select()
  }
  return true
}

export const blurFocusedPanelSearch = (root: HTMLElement): boolean => {
  const shortcut = currentSearchShortcut(root)
  if (!shortcut?.openClass) return false
  const search = root.querySelector<HTMLInputElement>(shortcut.search)
  if (!search || document.activeElement !== search) return false
  search.blur()
  return true
}

export const nextMapMetricShortcutIndex = (
  shortcuts: readonly (string | undefined)[],
  activeIndex: number,
  key: string,
): number => {
  if (key.length !== 1 || shortcuts.length === 0) return -1
  const shortcut = key.toLowerCase()
  const start = activeIndex >= 0 && activeIndex < shortcuts.length ? activeIndex : -1
  for (let offset = 1; offset <= shortcuts.length; offset++) {
    const index = (start + offset) % shortcuts.length
    if (shortcuts[index]?.toLowerCase() === shortcut) return index
  }
  return -1
}

export const mapDetailMetricTabForKey = (
  root: HTMLElement,
  key: string,
): HTMLButtonElement | null => {
  if (key.length !== 1) return null
  const tablist = root.querySelector<HTMLElement>('.tri-map--detail .tri-map-tablist')
  if (!tablist) return null
  const tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab'))
  const active = tabs.findIndex(tab => tab.getAttribute('aria-selected') === 'true')
  return (
    tabs[
      nextMapMetricShortcutIndex(
        tabs.map(tab => tab.dataset.shortcut),
        active,
        key,
      )
    ] ?? null
  )
}

export type TirePressurePaletteStep =
  | 'weightUnit'
  | 'riderMass'
  | 'bike'
  | 'bikeMass'
  | 'balance'
  | 'wheel'
  | 'customWheelFront'
  | 'customWheelRear'
  | 'measuredTireFront'
  | 'measuredTireRear'
  | 'tire'
  | 'surface'
  | 'speed'
  | 'result'

export const nextTirePressurePaletteStep = (
  step: Exclude<TirePressurePaletteStep, 'result'>,
  selection: TirePressureSelection = DEFAULT_TIRE_PRESSURE_SELECTION,
): TirePressurePaletteStep => {
  if (step === 'weightUnit') return 'riderMass'
  if (step === 'riderMass') return 'bike'
  if (step === 'bike') return 'bikeMass'
  if (step === 'bikeMass') return 'balance'
  if (step === 'balance') return 'wheel'
  if (step === 'wheel')
    return selection.wheel === 'custom' ? 'customWheelFront' : 'measuredTireFront'
  if (step === 'customWheelFront') return 'customWheelRear'
  if (step === 'customWheelRear') return 'measuredTireFront'
  if (step === 'measuredTireFront') return 'measuredTireRear'
  if (step === 'measuredTireRear') return 'tire'
  if (step === 'tire') return 'surface'
  if (step === 'surface') return 'speed'
  return 'result'
}

export const previousTirePressurePaletteStep = (
  step: TirePressurePaletteStep,
  selection: TirePressureSelection = DEFAULT_TIRE_PRESSURE_SELECTION,
): TirePressurePaletteStep | 'commands' => {
  if (step === 'result') return 'speed'
  if (step === 'speed') return 'surface'
  if (step === 'surface') return 'tire'
  if (step === 'tire') return 'measuredTireRear'
  if (step === 'measuredTireRear') return 'measuredTireFront'
  if (step === 'measuredTireFront')
    return selection.wheel === 'custom' ? 'customWheelRear' : 'wheel'
  if (step === 'customWheelRear') return 'customWheelFront'
  if (step === 'customWheelFront') return 'wheel'
  if (step === 'wheel') return 'balance'
  if (step === 'balance') return 'bikeMass'
  if (step === 'bikeMass') return 'bike'
  if (step === 'bike') return 'riderMass'
  if (step === 'riderMass') return 'weightUnit'
  return 'commands'
}

export const tirePressurePaletteSelectionIndex = (
  step: TirePressurePaletteStep,
  selection: TirePressureSelection,
): number => {
  if (step === 'weightUnit')
    return Math.max(0, TIRE_PRESSURE_WEIGHT_UNITS.indexOf(selection.weightUnit))
  if (step === 'bike')
    return Math.max(
      0,
      TIRE_PRESSURE_BIKES.findIndex(bike => bike.id === selection.bike),
    )
  if (step === 'balance')
    return Math.max(
      0,
      TIRE_PRESSURE_BALANCES.findIndex(balance => balance.id === selection.balance),
    )
  if (step === 'wheel')
    return Math.max(
      0,
      TIRE_PRESSURE_WHEELS.findIndex(wheel => wheel.id === selection.wheel),
    )
  if (step === 'tire')
    return Math.max(
      0,
      TIRE_PRESSURE_TIRES.findIndex(tire => tire.id === selection.tire),
    )
  if (step === 'surface')
    return Math.max(
      0,
      TIRE_PRESSURE_SURFACES.findIndex(surface => surface.id === selection.surface),
    )
  if (step === 'speed')
    return Math.max(
      0,
      TIRE_PRESSURE_SPEEDS.findIndex(speed => speed.mph === selection.speedMph),
    )
  return 0
}

const tirePressureSelectionFromRoot = (root: HTMLElement): TirePressureSelection => {
  const calculator = root.querySelector<HTMLElement>('.tri-pressure')
  const rootRiderKg = Number(calculator?.dataset.riderKg)
  const stored = readTirePressureSelection(
    isTirePressureRiderKg(rootRiderKg) ? rootRiderKg : null,
    calculator?.dataset.weightDate,
  )
  const weightUnit = calculator?.dataset.weightUnit
  const bike = calculator?.dataset.bike
  const bikeMassLb = Number(calculator?.dataset.bikeMassLb)
  const balance = calculator?.dataset.balance
  const wheel = calculator?.dataset.wheel
  const customWheelFrontMm = Number(calculator?.dataset.customWheelFrontMm)
  const customWheelRearMm = Number(calculator?.dataset.customWheelRearMm)
  const measuredTireFrontMm = Number(calculator?.dataset.measuredTireFrontMm)
  const measuredTireRearMm = Number(calculator?.dataset.measuredTireRearMm)
  const tire = calculator?.dataset.tire
  const surface = calculator?.dataset.surface
  const speedMph = Number(calculator?.dataset.speedMph)
  const selectedBike = bike && isTirePressureBikeId(bike) ? bike : stored.bike
  return {
    riderKg: isTirePressureRiderKg(rootRiderKg) ? rootRiderKg : stored.riderKg,
    weightUnit: weightUnit && isTirePressureWeightUnit(weightUnit) ? weightUnit : stored.weightUnit,
    bike: selectedBike,
    bikeMassesLb: isTirePressureBikeMassLb(bikeMassLb)
      ? { ...stored.bikeMassesLb, [selectedBike]: bikeMassLb }
      : stored.bikeMassesLb,
    balance: balance && isTirePressureBalanceId(balance) ? balance : stored.balance,
    wheel: wheel && isTirePressureWheelId(wheel) ? wheel : stored.wheel,
    customWheel: {
      frontInnerWidthMm: isTirePressureInnerWidthMm(customWheelFrontMm)
        ? customWheelFrontMm
        : stored.customWheel.frontInnerWidthMm,
      rearInnerWidthMm: isTirePressureInnerWidthMm(customWheelRearMm)
        ? customWheelRearMm
        : stored.customWheel.rearInnerWidthMm,
    },
    measuredTire: {
      frontWidthMm: isTirePressureMeasuredWidthMm(measuredTireFrontMm)
        ? measuredTireFrontMm
        : stored.measuredTire.frontWidthMm,
      rearWidthMm: isTirePressureMeasuredWidthMm(measuredTireRearMm)
        ? measuredTireRearMm
        : stored.measuredTire.rearWidthMm,
    },
    tire: tire && isTirePressureTireId(tire) ? tire : stored.tire,
    surface: surface && isTirePressureSurfaceId(surface) ? surface : stored.surface,
    speedMph: isTirePressureSpeed(speedMph) ? speedMph : stored.speedMph,
  }
}

export const setupCommandPalette = (root: HTMLElement, context: TriathlonContext): (() => void) => {
  const trigger = root.querySelector<HTMLButtonElement>('.tri-cmdk-trigger')
  const overlay = el('div', 'tri-cmdk', undefined, {
    id: 'tri-command-palette',
    'aria-hidden': 'true',
  })
  const box = el('div', 'tri-cmdk-box', undefined, {
    role: 'dialog',
    'aria-label': 'command palette',
  })
  const input = el('input', 'tri-cmdk-input', undefined, {
    type: 'text',
    placeholder: 'go to page · toggle units...',
    'aria-label': 'command',
    autocomplete: 'off',
    spellcheck: 'false',
  }) as HTMLInputElement
  const list = el('div', 'tri-cmdk-list', undefined, { role: 'listbox' })
  box.append(input, list)
  overlay.appendChild(box)
  root.appendChild(overlay)

  interface Cmd {
    label: () => string
    hint: string | (() => string)
    keys: string
    run: () => void
  }
  type PaletteMode = 'commands' | TirePressurePaletteStep
  let mode: PaletteMode = 'commands'
  let pressureSelection = tirePressureSelectionFromRoot(root)
  const pressureWeightDate = root.querySelector<HTMLElement>('.tri-pressure')?.dataset.weightDate

  const commandHint = (command: Cmd): string =>
    typeof command.hint === 'function' ? command.hint() : command.hint

  const navTo = (path: string) => (): void => {
    close()
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }

  const updatePressureSelection = (change: TirePressureChange): void => {
    if (change.field === 'riderMass')
      pressureSelection = { ...pressureSelection, riderKg: change.valueKg }
    else if (change.field === 'weightUnit')
      pressureSelection = { ...pressureSelection, weightUnit: change.value }
    else if (change.field === 'bike')
      pressureSelection = { ...pressureSelection, bike: change.value }
    else if (change.field === 'bikeMass')
      pressureSelection = {
        ...pressureSelection,
        bike: change.bike,
        bikeMassesLb: { ...pressureSelection.bikeMassesLb, [change.bike]: change.value },
      }
    else if (change.field === 'balance')
      pressureSelection = { ...pressureSelection, balance: change.value }
    else if (change.field === 'wheel')
      pressureSelection = { ...pressureSelection, wheel: change.value }
    else if (change.field === 'customWheelWidth')
      pressureSelection = {
        ...pressureSelection,
        wheel: 'custom',
        customWheel: {
          ...pressureSelection.customWheel,
          [change.axle === 'front' ? 'frontInnerWidthMm' : 'rearInnerWidthMm']: change.value,
        },
      }
    else if (change.field === 'measuredTireWidth')
      pressureSelection = {
        ...pressureSelection,
        measuredTire: {
          ...pressureSelection.measuredTire,
          [change.axle === 'front' ? 'frontWidthMm' : 'rearWidthMm']: change.value,
        },
      }
    else if (change.field === 'tire')
      pressureSelection = { ...pressureSelection, tire: change.value }
    else if (change.field === 'surface')
      pressureSelection = { ...pressureSelection, surface: change.value }
    else pressureSelection = { ...pressureSelection, speedMph: change.value }
    storeTirePressureSelection(pressureSelection, pressureWeightDate)
    root.dispatchEvent(new CustomEvent(TRI_TIRE_PRESSURE_CHANGE_EVENT, { detail: change }))
  }

  const pressureRecommendation = () => calculateTirePressure(pressureSelection)

  const tirePressureHint = (): string => {
    const recommendation = pressureRecommendation()
    return recommendation
      ? `${formatTirePressurePsi(recommendation.frontPsi)}/${formatTirePressurePsi(recommendation.rearPsi)} PSI`
      : 'daily PSI'
  }

  const cmds: Cmd[] = [
    ...TRI_PAGES.map(p => ({
      label: () => `${p.label}`,
      hint: p.hint,
      keys: `go ${p.label} ${p.path}`,
      run: navTo(p.path),
    })),
    {
      label: () => 'gear ratios',
      hint: 'chainring × cassette',
      keys: 'gear ratio gearing chainring cassette drivetrain teeth',
      run: () => {
        const tab = root.querySelector<HTMLButtonElement>('[data-calc-tab="gear-ratios"]')
        if (tab) {
          close()
          tab.click()
        } else navTo('/triathlon/calc#gear-ratios')()
      },
    },
    {
      label: () => 'tire pressure',
      hint: tirePressureHint,
      keys: 'tire tyre pressure psi front rear pirelli silca cervelo speedmax hunt aerodynamicist reserve custom wheel bike rider weight kg lb tubeless tpu tube',
      run: () => setPressureMode('weightUnit'),
    },
    {
      label: () =>
        context.presentation.distance === 'imperial' ? 'imperial → metric' : 'metric → imperial',
      hint: 'units',
      keys: 'toggle units km mi miles kg lb imperial metric pace distance speed weight',
      run: () => {
        toggleTriUnit(context.preferences)
        render()
      },
    },
    {
      label: () =>
        context.formatter.text(
          context.presentation.powerSamples === 'exclude-zero'
            ? 'power averages · zeros excluded'
            : 'power averages · zeros included',
        ),
      hint: 'power',
      keys: 'power watts zero zeros include exclude coasting freewheel downhill traffic stop',
      run: () => {
        toggleTriPowerFilter(context.preferences)
        render()
      },
    },
    {
      label: () =>
        context.presentation.locale === 'fr' ? 'langue · english' : 'language · français',
      hint: 'locale',
      keys: 'language langue locale english french francais français en fr i18n',
      run: () => {
        context.preferences.update({ locale: context.presentation.locale === 'fr' ? 'en' : 'fr' })
        render()
      },
    },
    {
      label: () => {
        const next = nextTriMapStyle()
        return `map style · ${next === 'mono' ? 'monochrome' : next}`
      },
      hint: 'map',
      keys: 'map style roads streets monochrome mono satellite imagery mapbox route road',
      run: () => {
        toggleTriMapStyle()
        render()
      },
    },
    {
      label: () =>
        context.formatter.text(
          root.classList.contains('tri-panels-fullscreen')
            ? 'panels · windowed'
            : 'panels · full screen',
        ),
      hint: 'layout',
      keys: 'toggle panels fullscreen full screen windowed desktop mobile analytics map training layout',
      run: () => {
        toggleTriPanelsFullscreen(root)
        render()
      },
    },
  ]

  let items: Cmd[] = cmds
  let sel = 0
  let isOpen = false

  const selectPressure = (change: TirePressureChange, step: TirePressurePaletteStep): void => {
    updatePressureSelection(change)
    setPressureMode(step)
  }

  const pressureCommands = (): Cmd[] => {
    if (mode === 'weightUnit')
      return TIRE_PRESSURE_WEIGHT_UNITS.map(unit => ({
        label: () => `${unit === pressureSelection.weightUnit ? '✓ ' : ''}${unit}`,
        hint: unit === 'kg' ? 'kilograms' : 'pounds',
        keys: `${unit} kilograms pounds lbs rider weight mass`,
        run: () => selectPressure({ field: 'weightUnit', value: unit }, 'riderMass'),
      }))
    if (mode === 'riderMass') {
      const value = Number(input.value)
      const riderKg = tirePressureWeightToKg(value, pressureSelection.weightUnit)
      const valid = isTirePressureRiderKg(riderKg)
      return [
        {
          label: () =>
            valid
              ? `rider weight · ${input.value} ${pressureSelection.weightUnit}`
              : `enter ${pressureSelection.weightUnit === 'kg' ? '25–200 kg' : '55.1–440.9 lb'}`,
          hint: valid ? 'use current weight' : 'valid weight required',
          keys: 'rider body weight mass kilograms kg pounds lb custom',
          run: () => {
            if (valid) selectPressure({ field: 'riderMass', valueKg: riderKg }, 'bike')
          },
        },
      ]
    }
    if (mode === 'bike')
      return TIRE_PRESSURE_BIKES.map(bike => ({
        label: () => `${bike.id === pressureSelection.bike ? '✓ ' : ''}${bike.label}`,
        hint: () => `${pressureSelection.bikeMassesLb[bike.id]} lb equipped`,
        keys: `${bike.id} ${bike.label} custom weight mass pounds lb`,
        run: () => selectPressure({ field: 'bike', value: bike.id }, 'bikeMass'),
      }))
    if (mode === 'bikeMass') {
      const bike = tirePressureBike(pressureSelection.bike)
      const mass = Number(input.value)
      const valid = isTirePressureBikeMassLb(mass)
      return [
        {
          label: () => (valid ? `${bike.label} · ${mass} lb` : `${bike.label} · enter 10–80 lb`),
          hint: valid ? 'use custom weight' : 'custom value required',
          keys: `${bike.label} weight mass pounds lb custom`,
          run: () => {
            if (valid) selectPressure({ field: 'bikeMass', bike: bike.id, value: mass }, 'balance')
          },
        },
      ]
    }
    if (mode === 'balance')
      return TIRE_PRESSURE_BALANCES.map(balance => ({
        label: () => `${balance.id === pressureSelection.balance ? '✓ ' : ''}${balance.label}`,
        hint: `${balance.frontPercent}% front · ${balance.rearPercent}% rear`,
        keys: `${balance.id} ${balance.label} balance distribution front rear`,
        run: () => selectPressure({ field: 'balance', value: balance.id }, 'wheel'),
      }))
    if (mode === 'wheel')
      return TIRE_PRESSURE_WHEELS.map(wheel => ({
        label: () => `${wheel.id === pressureSelection.wheel ? '✓ ' : ''}${wheel.label}`,
        hint: (() => {
          const selectedWheel =
            wheel.id === 'custom' ? { ...wheel, ...pressureSelection.customWheel } : wheel
          return selectedWheel.frontInnerWidthMm === selectedWheel.rearInnerWidthMm
            ? `${selectedWheel.frontInnerWidthMm} mm internal`
            : `${selectedWheel.frontInnerWidthMm}/${selectedWheel.rearInnerWidthMm} mm internal`
        })(),
        keys: `${wheel.id} ${wheel.label}`,
        run: () =>
          selectPressure(
            { field: 'wheel', value: wheel.id },
            wheel.id === 'custom' ? 'customWheelFront' : 'measuredTireFront',
          ),
      }))
    if (mode === 'customWheelFront' || mode === 'customWheelRear') {
      const axle = mode === 'customWheelFront' ? 'front' : 'rear'
      const width = Number(input.value)
      const valid = isTirePressureInnerWidthMm(width)
      return [
        {
          label: () =>
            valid ? `${axle} internal width · ${width} mm` : `${axle} width · enter 13–35 mm`,
          hint: valid ? 'use measured width' : 'valid width required',
          keys: `${axle} wheel rim internal width millimetres mm custom`,
          run: () => {
            if (valid)
              selectPressure(
                { field: 'customWheelWidth', axle, value: width },
                mode === 'customWheelFront' ? 'customWheelRear' : 'measuredTireFront',
              )
          },
        },
      ]
    }
    if (mode === 'measuredTireFront' || mode === 'measuredTireRear') {
      const axle = mode === 'measuredTireFront' ? 'front' : 'rear'
      const width = Number(input.value)
      const valid = isTirePressureMeasuredWidthMm(width)
      return [
        {
          label: () =>
            valid
              ? `${axle} measured tire width · ${width} mm`
              : `${axle} measured tire width · enter 20–65 mm`,
          hint: valid ? 'use measured width' : 'whole millimetres required',
          keys: `${axle} measured tire casing width millimetres mm`,
          run: () => {
            if (valid)
              selectPressure(
                { field: 'measuredTireWidth', axle, value: width },
                mode === 'measuredTireFront' ? 'measuredTireRear' : 'tire',
              )
          },
        },
      ]
    }
    if (mode === 'tire')
      return TIRE_PRESSURE_TIRES.map(tire => ({
        label: () => `${tire.id === pressureSelection.tire ? '✓ ' : ''}${tire.label}`,
        hint: tire.detail,
        keys: `${tire.id} ${tire.label} ${tire.detail}`,
        run: () => selectPressure({ field: 'tire', value: tire.id }, 'surface'),
      }))
    if (mode === 'surface')
      return TIRE_PRESSURE_SURFACES.map(surface => ({
        label: () => `${surface.id === pressureSelection.surface ? '✓ ' : ''}${surface.label}`,
        hint: 'dry',
        keys: `${surface.id} ${surface.label}`,
        run: () => selectPressure({ field: 'surface', value: surface.id }, 'speed'),
      }))
    if (mode === 'speed')
      return TIRE_PRESSURE_SPEEDS.map(speed => ({
        label: () => `${speed.mph === pressureSelection.speedMph ? '✓ ' : ''}${speed.label}`,
        hint: `${speed.mph} mph average`,
        keys: `${speed.label} ${speed.mph} mph`,
        run: () => selectPressure({ field: 'speed', value: speed.mph }, 'result'),
      }))
    if (mode === 'result') {
      const recommendation = pressureRecommendation()
      const bike = tirePressureBike(pressureSelection.bike)
      const balance = tirePressureBalance(pressureSelection.balance)
      const wheel = selectedTirePressureWheel(pressureSelection)
      const tire = tirePressureTire(pressureSelection.tire)
      const surface = tirePressureSurface(pressureSelection.surface)
      return [
        {
          label: () =>
            recommendation
              ? `front ${formatTirePressurePsi(recommendation.frontPsi)} PSI · rear ${formatTirePressurePsi(recommendation.rearPsi)} PSI`
              : 'morning weight unavailable',
          hint: recommendation
            ? `${formatTirePressureWeight(recommendation.systemKg, pressureSelection.weightUnit)} ${pressureSelection.weightUnit} · open calculator`
            : 'open calculator',
          keys: 'result pressure front rear open',
          run: () => {
            if (root.querySelector('.tri-pressure')) {
              close()
              root.dispatchEvent(new CustomEvent(TRI_TIRE_PRESSURE_OPEN_EVENT))
            } else navTo('/triathlon/calc#tire-pressure')()
          },
        },
        {
          label: () =>
            pressureSelection.riderKg == null
              ? 'rider weight · unavailable'
              : `rider weight · ${formatTirePressureWeight(pressureSelection.riderKg, pressureSelection.weightUnit)} ${pressureSelection.weightUnit}`,
          hint: 'change',
          keys: 'rider body weight mass kilograms kg pounds lb change',
          run: () => setPressureMode('riderMass'),
        },
        {
          label: () => `weight unit · ${pressureSelection.weightUnit}`,
          hint: 'change',
          keys: 'weight unit kilograms kg pounds lb change',
          run: () => setPressureMode('weightUnit'),
        },
        {
          label: () => `bike · ${bike.label}`,
          hint: 'change',
          keys: 'bike change',
          run: () => setPressureMode('bike'),
        },
        {
          label: () => `bike weight · ${pressureSelection.bikeMassesLb[bike.id]} lb`,
          hint: 'change',
          keys: 'bike weight mass pounds lb custom change',
          run: () => setPressureMode('bikeMass'),
        },
        {
          label: () => `balance · ${balance.label}`,
          hint: `${balance.frontPercent}% front · ${balance.rearPercent}% rear`,
          keys: 'balance distribution front rear change',
          run: () => setPressureMode('balance'),
        },
        {
          label: () => `wheel · ${wheel.label}`,
          hint: 'change',
          keys: 'wheel change',
          run: () => setPressureMode('wheel'),
        },
        ...(wheel.id === 'custom'
          ? [
              {
                label: () =>
                  `internal width · ${wheel.frontInnerWidthMm}/${wheel.rearInnerWidthMm} mm`,
                hint: 'front / rear',
                keys: 'custom wheel rim internal width front rear change',
                run: () => setPressureMode('customWheelFront'),
              },
            ]
          : []),
        {
          label: () =>
            `measured tire width · ${pressureSelection.measuredTire.frontWidthMm}/${pressureSelection.measuredTire.rearWidthMm} mm`,
          hint: 'front / rear',
          keys: 'measured tire casing width front rear millimetres mm change',
          run: () => setPressureMode('measuredTireFront'),
        },
        {
          label: () => `tire · ${tire.label}`,
          hint: tire.detail,
          keys: 'tire setup tubeless tpu change',
          run: () => setPressureMode('tire'),
        },
        {
          label: () => `surface · ${surface.label}`,
          hint: 'change',
          keys: 'surface change',
          run: () => setPressureMode('surface'),
        },
        {
          label: () => `speed · ${pressureSelection.speedMph} mph`,
          hint: 'change',
          keys: 'speed change',
          run: () => setPressureMode('speed'),
        },
      ]
    }
    return cmds
  }

  function setPressureMode(next: PaletteMode): void {
    mode = next
    sel = mode === 'commands' ? 0 : tirePressurePaletteSelectionIndex(mode, pressureSelection)
    const editable =
      mode === 'riderMass' ||
      mode === 'bikeMass' ||
      mode === 'customWheelFront' ||
      mode === 'customWheelRear' ||
      mode === 'measuredTireFront' ||
      mode === 'measuredTireRear'
    input.readOnly = mode !== 'commands' && !editable
    input.inputMode = editable ? 'decimal' : 'search'
    input.value =
      mode === 'commands'
        ? ''
        : mode === 'riderMass'
          ? pressureSelection.riderKg == null
            ? ''
            : formatTirePressureWeight(pressureSelection.riderKg, pressureSelection.weightUnit)
          : mode === 'bikeMass'
            ? String(pressureSelection.bikeMassesLb[pressureSelection.bike])
            : mode === 'customWheelFront'
              ? String(pressureSelection.customWheel.frontInnerWidthMm)
              : mode === 'customWheelRear'
                ? String(pressureSelection.customWheel.rearInnerWidthMm)
                : mode === 'measuredTireFront'
                  ? String(pressureSelection.measuredTire.frontWidthMm)
                  : mode === 'measuredTireRear'
                    ? String(pressureSelection.measuredTire.rearWidthMm)
                    : `tire pressure / ${mode}`
    input.setAttribute(
      'aria-label',
      mode === 'commands'
        ? 'command'
        : mode === 'riderMass'
          ? `rider weight in ${pressureSelection.weightUnit}`
          : mode === 'bikeMass'
            ? `${tirePressureBike(pressureSelection.bike).label} weight in pounds`
            : mode === 'customWheelFront'
              ? 'front internal rim width in millimetres'
              : mode === 'customWheelRear'
                ? 'rear internal rim width in millimetres'
                : mode === 'measuredTireFront'
                  ? 'front measured tire width in millimetres'
                  : mode === 'measuredTireRear'
                    ? 'rear measured tire width in millimetres'
                    : `tire pressure ${mode}`,
    )
    render(true)
    if (editable) input.select()
  }

  const paint = (): void => {
    const rows = list.querySelectorAll<HTMLElement>('.tri-cmdk-row')
    rows.forEach((r, i) => {
      r.classList.toggle('tri-cmdk-row--on', i === sel)
      r.setAttribute('aria-selected', String(i === sel))
    })
    rows[sel]?.scrollIntoView({ block: 'nearest' })
  }
  const render = (continuity = false): void => {
    const q = input.value.trim().toLowerCase()
    items =
      mode === 'commands'
        ? q
          ? cmds.filter(c => `${c.label()} ${commandHint(c)} ${c.keys}`.toLowerCase().includes(q))
          : cmds
        : pressureCommands()
    if (sel >= items.length) sel = Math.max(0, items.length - 1)
    list.replaceChildren(
      ...items.map((c, i) => {
        const row = el(
          'div',
          i === sel ? 'tri-cmdk-row tri-cmdk-row--on' : 'tri-cmdk-row',
          undefined,
          { role: 'option', 'aria-selected': String(i === sel) },
        )
        row.append(
          el('span', 'tri-cmdk-row-label', c.label()),
          el('span', 'tri-cmdk-row-hint', commandHint(c)),
        )
        row.addEventListener('mousemove', () => {
          if (sel !== i) {
            sel = i
            paint()
          }
        })
        row.addEventListener('click', () => c.run())
        return row
      }),
    )
    if (!items.length)
      list.appendChild(el('div', 'tri-cmdk-empty', context.formatter.text('no commands')))
    if (continuity) {
      list.classList.remove('tri-cmdk-list--continuity')
      void list.offsetWidth
      list.classList.add('tri-cmdk-list--continuity')
    }
  }
  const openPalette = (): void => {
    if (isOpen) return
    isOpen = true
    mode = 'commands'
    pressureSelection = tirePressureSelectionFromRoot(root)
    input.readOnly = false
    input.inputMode = 'search'
    input.setAttribute('aria-label', 'command')
    input.value = ''
    sel = 0
    render()
    overlay.classList.add('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'false')
    trigger?.setAttribute('aria-expanded', 'true')
    input.focus()
  }
  function close(): void {
    if (!isOpen) return
    isOpen = false
    overlay.classList.remove('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'true')
    trigger?.setAttribute('aria-expanded', 'false')
    input.blur()
  }

  const togglePalette = (): void => {
    if (isOpen) close()
    else openPalette()
  }

  const onInput = (): void => {
    if (
      mode !== 'commands' &&
      mode !== 'riderMass' &&
      mode !== 'bikeMass' &&
      mode !== 'customWheelFront' &&
      mode !== 'customWheelRear' &&
      mode !== 'measuredTireFront' &&
      mode !== 'measuredTireRear'
    )
      return
    sel = 0
    render()
  }
  const onInputKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    } else if (
      (mode === 'riderMass' ||
        mode === 'bikeMass' ||
        mode === 'customWheelFront' ||
        mode === 'customWheelRear' ||
        mode === 'measuredTireFront' ||
        mode === 'measuredTireRear') &&
      e.key === 'Backspace' &&
      input.value === ''
    ) {
      e.preventDefault()
      setPressureMode(previousTirePressurePaletteStep(mode, pressureSelection))
    } else if (
      mode !== 'commands' &&
      mode !== 'riderMass' &&
      mode !== 'bikeMass' &&
      mode !== 'customWheelFront' &&
      mode !== 'customWheelRear' &&
      mode !== 'measuredTireFront' &&
      mode !== 'measuredTireRear' &&
      (e.key === 'ArrowLeft' || e.key === 'Backspace')
    ) {
      e.preventDefault()
      setPressureMode(previousTirePressurePaletteStep(mode, pressureSelection))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      items[sel]?.run()
    } else if (e.key === 'ArrowDown' || (e.ctrlKey && e.key.toLowerCase() === 'n')) {
      e.preventDefault()
      if (items.length) sel = (sel + 1) % items.length
      paint()
    } else if (e.key === 'ArrowUp' || (e.ctrlKey && e.key.toLowerCase() === 'p')) {
      e.preventDefault()
      if (items.length) sel = (sel - 1 + items.length) % items.length
      paint()
    }
  }
  const onDocKey = (e: KeyboardEvent): void => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      e.stopImmediatePropagation()
      if (toggleSearchFocus(root, null) || currentSearchShortcut(root)) return
      if (root.matches('.tri-analytics-open, .tri-map-open, .tri-training-open, .tri-calc-open'))
        return
      togglePalette()
    }
  }
  const onScrim = (e: MouseEvent): void => {
    if (e.target === overlay) close()
  }
  input.addEventListener('input', onInput)
  input.addEventListener('keydown', onInputKey)
  overlay.addEventListener('mousedown', onScrim)
  trigger?.addEventListener('click', togglePalette)
  document.addEventListener('keydown', onDocKey, true)
  const analyticsPath = root.dataset.analyticsPath
  if (pressureSelection.riderKg == null && analyticsPath)
    void context.resources.analytics.load(analyticsPath).then(result => {
      if (result.status !== 'ready') return
      const weight = latestMorningBodyWeight(result.value.body.composition)
      if (!weight) return
      pressureSelection = { ...pressureSelection, riderKg: weight.kg }
      if (isOpen) render()
    })
  return () => {
    document.removeEventListener('keydown', onDocKey, true)
    trigger?.removeEventListener('click', togglePalette)
    overlay.remove()
  }
}
