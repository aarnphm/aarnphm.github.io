import type { TriathlonContext } from '../runtime/context'
import {
  calculateTirePressure,
  formatTirePressurePsi,
  isTirePressureBikeId,
  isTirePressureSpeed,
  isTirePressureSurfaceId,
  isTirePressureTireId,
  isTirePressureWheelId,
  latestMorningBodyWeight,
  tirePressureBike,
  tirePressureSurface,
  tirePressureTire,
  tirePressureWheel,
  TIRE_PRESSURE_BIKES,
  TRI_TIRE_PRESSURE_CHANGE_EVENT,
  TRI_TIRE_PRESSURE_OPEN_EVENT,
  TIRE_PRESSURE_SPEEDS,
  TIRE_PRESSURE_SURFACES,
  TIRE_PRESSURE_TIRES,
  TIRE_PRESSURE_WHEELS,
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
  { path: '/triathlon/calc', label: 'calculator', hint: 'race calc' },
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

export type TirePressurePaletteStep = 'bike' | 'wheel' | 'tire' | 'surface' | 'speed' | 'result'

export const nextTirePressurePaletteStep = (
  step: Exclude<TirePressurePaletteStep, 'result'>,
): TirePressurePaletteStep => {
  if (step === 'bike') return 'wheel'
  if (step === 'wheel') return 'tire'
  if (step === 'tire') return 'surface'
  if (step === 'surface') return 'speed'
  return 'result'
}

export const previousTirePressurePaletteStep = (
  step: TirePressurePaletteStep,
): TirePressurePaletteStep | 'commands' => {
  if (step === 'result') return 'speed'
  if (step === 'speed') return 'surface'
  if (step === 'surface') return 'tire'
  if (step === 'tire') return 'wheel'
  if (step === 'wheel') return 'bike'
  return 'commands'
}

export const tirePressurePaletteSelectionIndex = (
  step: TirePressurePaletteStep,
  selection: TirePressureSelection,
): number => {
  if (step === 'bike')
    return Math.max(
      0,
      TIRE_PRESSURE_BIKES.findIndex(bike => bike.id === selection.bike),
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
  const stored = readTirePressureSelection()
  const bike = calculator?.dataset.bike
  const wheel = calculator?.dataset.wheel
  const tire = calculator?.dataset.tire
  const surface = calculator?.dataset.surface
  const speedMph = Number(calculator?.dataset.speedMph)
  return {
    bike: bike && isTirePressureBikeId(bike) ? bike : stored.bike,
    wheel: wheel && isTirePressureWheelId(wheel) ? wheel : stored.wheel,
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
  let pressureRiderKg = Number(root.querySelector<HTMLElement>('.tri-pressure')?.dataset.riderKg)

  const commandHint = (command: Cmd): string =>
    typeof command.hint === 'function' ? command.hint() : command.hint

  const navTo = (path: string) => (): void => {
    close()
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }

  const updatePressureSelection = (change: TirePressureChange): void => {
    if (change.field === 'bike') pressureSelection = { ...pressureSelection, bike: change.value }
    else if (change.field === 'wheel')
      pressureSelection = { ...pressureSelection, wheel: change.value }
    else if (change.field === 'tire')
      pressureSelection = { ...pressureSelection, tire: change.value }
    else if (change.field === 'surface')
      pressureSelection = { ...pressureSelection, surface: change.value }
    else pressureSelection = { ...pressureSelection, speedMph: change.value }
    storeTirePressureSelection(pressureSelection)
    root.dispatchEvent(new CustomEvent(TRI_TIRE_PRESSURE_CHANGE_EVENT, { detail: change }))
  }

  const pressureRecommendation = () => {
    return calculateTirePressure(pressureRiderKg, pressureSelection)
  }

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
      label: () => 'tire pressure',
      hint: tirePressureHint,
      keys: 'tire tyre pressure psi front rear pirelli silca cervelo speedmax princeton reserve wheel bike tubeless tpu tube',
      run: () => setPressureMode('bike'),
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

  const selectPressure = (
    change: TirePressureChange,
    step: Exclude<TirePressurePaletteStep, 'bike'>,
  ): void => {
    updatePressureSelection(change)
    setPressureMode(step)
  }

  const pressureCommands = (): Cmd[] => {
    if (mode === 'bike')
      return TIRE_PRESSURE_BIKES.map(bike => ({
        label: () => `${bike.id === pressureSelection.bike ? '✓ ' : ''}${bike.label}`,
        hint: `${bike.massLb} lb equipped`,
        keys: `${bike.id} ${bike.label}`,
        run: () => selectPressure({ field: 'bike', value: bike.id }, 'wheel'),
      }))
    if (mode === 'wheel')
      return TIRE_PRESSURE_WHEELS.map(wheel => ({
        label: () => `${wheel.id === pressureSelection.wheel ? '✓ ' : ''}${wheel.label}`,
        hint:
          wheel.frontInnerWidthMm === wheel.rearInnerWidthMm
            ? `${wheel.frontInnerWidthMm} mm internal`
            : `${wheel.frontInnerWidthMm}/${wheel.rearInnerWidthMm} mm internal`,
        keys: `${wheel.id} ${wheel.label}`,
        run: () => selectPressure({ field: 'wheel', value: wheel.id }, 'tire'),
      }))
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
      const wheel = tirePressureWheel(pressureSelection.wheel)
      const tire = tirePressureTire(pressureSelection.tire)
      const surface = tirePressureSurface(pressureSelection.surface)
      return [
        {
          label: () =>
            recommendation
              ? `front ${formatTirePressurePsi(recommendation.frontPsi)} PSI · rear ${formatTirePressurePsi(recommendation.rearPsi)} PSI`
              : 'morning weight unavailable',
          hint: recommendation
            ? `${recommendation.systemKg.toFixed(1)} kg · open calculator`
            : 'open calculator',
          keys: 'result pressure front rear open',
          run: () => {
            if (root.querySelector('.tri-pressure')) {
              close()
              root.dispatchEvent(new CustomEvent(TRI_TIRE_PRESSURE_OPEN_EVENT))
            } else navTo('/triathlon/tools#tire-pressure')()
          },
        },
        {
          label: () => `bike · ${bike.label}`,
          hint: 'change',
          keys: 'bike change',
          run: () => setPressureMode('bike'),
        },
        {
          label: () => `wheel · ${wheel.label}`,
          hint: 'change',
          keys: 'wheel change',
          run: () => setPressureMode('wheel'),
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
    input.readOnly = mode !== 'commands'
    input.value = mode === 'commands' ? '' : `tire pressure / ${mode}`
    input.setAttribute('aria-label', mode === 'commands' ? 'command' : `tire pressure ${mode}`)
    render(true)
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
    if (mode !== 'commands') return
    sel = 0
    render()
  }
  const onInputKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    } else if (mode !== 'commands' && (e.key === 'ArrowLeft' || e.key === 'Backspace')) {
      e.preventDefault()
      setPressureMode(previousTirePressurePaletteStep(mode))
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
  if ((!Number.isFinite(pressureRiderKg) || pressureRiderKg <= 0) && analyticsPath)
    void context.resources.analytics.load(analyticsPath).then(result => {
      if (result.status !== 'ready') return
      const weight = latestMorningBodyWeight(result.value.body.composition)
      if (!weight) return
      pressureRiderKg = weight.kg
      if (isOpen) render()
    })
  return () => {
    document.removeEventListener('keydown', onDocKey, true)
    trigger?.removeEventListener('click', togglePalette)
    overlay.remove()
  }
}
