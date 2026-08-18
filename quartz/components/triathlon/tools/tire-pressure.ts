import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressurePsi,
  isTirePressureBalanceId,
  isTirePressureBikeId,
  isTirePressureBikeMassLb,
  isTirePressureChange,
  isTirePressureSpeed,
  isTirePressureSurfaceId,
  isTirePressureTireId,
  isTirePressureWheelId,
  tirePressureSurface,
  TRI_TIRE_PRESSURE_CHANGE_EVENT,
  TRI_TIRE_PRESSURE_OPEN_EVENT,
  type TirePressureChange,
  type TirePressureSelection,
} from '../../../util/triathlon-tire-pressure'

const TIRE_PRESSURE_BIKE_KEY = 'triathlon-tire-pressure-bike'
const TIRE_PRESSURE_CERVELO_MASS_KEY = 'triathlon-tire-pressure-bike-mass-cervelo'
const TIRE_PRESSURE_SPEEDMAX_MASS_KEY = 'triathlon-tire-pressure-bike-mass-speedmax'
const TIRE_PRESSURE_CUSTOM_MASS_KEY = 'triathlon-tire-pressure-bike-mass-custom'
const TIRE_PRESSURE_BALANCE_KEY = 'triathlon-tire-pressure-balance'
const TIRE_PRESSURE_WHEEL_KEY = 'triathlon-tire-pressure-wheel'
const TIRE_PRESSURE_TIRE_KEY = 'triathlon-tire-pressure-tire'
const TIRE_PRESSURE_SURFACE_KEY = 'triathlon-tire-pressure-surface'
const TIRE_PRESSURE_SPEED_KEY = 'triathlon-tire-pressure-speed'

const storedBikeMass = (key: string, fallback: number): number => {
  const mass = Number(localStorage.getItem(key))
  return isTirePressureBikeMassLb(mass) ? mass : fallback
}

const renderSurfaceTip = (calculator: HTMLElement, surfaceId: TirePressureSelection['surface']) => {
  const surface = tirePressureSurface(surfaceId)
  const coefficient = calculator.querySelector<HTMLElement>('[data-pressure-surface-coefficient]')
  const note = calculator.querySelector<HTMLElement>('[data-pressure-surface-note]')
  if (coefficient) coefficient.textContent = String(surface.coefficient)
  if (note) note.textContent = surface.note
}

export const readTirePressureSelection = (): TirePressureSelection => {
  const storedBike = localStorage.getItem(TIRE_PRESSURE_BIKE_KEY)
  const storedWheel = localStorage.getItem(TIRE_PRESSURE_WHEEL_KEY)
  const storedBalance = localStorage.getItem(TIRE_PRESSURE_BALANCE_KEY)
  const storedTire = localStorage.getItem(TIRE_PRESSURE_TIRE_KEY)
  const storedSurface = localStorage.getItem(TIRE_PRESSURE_SURFACE_KEY)
  const storedSpeed = Number(localStorage.getItem(TIRE_PRESSURE_SPEED_KEY))
  return {
    bike:
      storedBike && isTirePressureBikeId(storedBike)
        ? storedBike
        : DEFAULT_TIRE_PRESSURE_SELECTION.bike,
    bikeMassesLb: {
      cervelo: storedBikeMass(
        TIRE_PRESSURE_CERVELO_MASS_KEY,
        DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB.cervelo,
      ),
      speedmax: storedBikeMass(
        TIRE_PRESSURE_SPEEDMAX_MASS_KEY,
        DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB.speedmax,
      ),
      custom: storedBikeMass(
        TIRE_PRESSURE_CUSTOM_MASS_KEY,
        DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB.custom,
      ),
    },
    balance:
      storedBalance && isTirePressureBalanceId(storedBalance)
        ? storedBalance
        : DEFAULT_TIRE_PRESSURE_SELECTION.balance,
    wheel:
      storedWheel && isTirePressureWheelId(storedWheel)
        ? storedWheel
        : DEFAULT_TIRE_PRESSURE_SELECTION.wheel,
    tire:
      storedTire && isTirePressureTireId(storedTire)
        ? storedTire
        : DEFAULT_TIRE_PRESSURE_SELECTION.tire,
    surface:
      storedSurface && isTirePressureSurfaceId(storedSurface)
        ? storedSurface
        : DEFAULT_TIRE_PRESSURE_SELECTION.surface,
    speedMph: isTirePressureSpeed(storedSpeed)
      ? storedSpeed
      : DEFAULT_TIRE_PRESSURE_SELECTION.speedMph,
  }
}

export const storeTirePressureSelection = (selection: TirePressureSelection): void => {
  localStorage.setItem(TIRE_PRESSURE_BIKE_KEY, selection.bike)
  localStorage.setItem(TIRE_PRESSURE_CERVELO_MASS_KEY, String(selection.bikeMassesLb.cervelo))
  localStorage.setItem(TIRE_PRESSURE_SPEEDMAX_MASS_KEY, String(selection.bikeMassesLb.speedmax))
  localStorage.setItem(TIRE_PRESSURE_CUSTOM_MASS_KEY, String(selection.bikeMassesLb.custom))
  localStorage.setItem(TIRE_PRESSURE_BALANCE_KEY, selection.balance)
  localStorage.setItem(TIRE_PRESSURE_WHEEL_KEY, selection.wheel)
  localStorage.setItem(TIRE_PRESSURE_TIRE_KEY, selection.tire)
  localStorage.setItem(TIRE_PRESSURE_SURFACE_KEY, selection.surface)
  localStorage.setItem(TIRE_PRESSURE_SPEED_KEY, String(selection.speedMph))
}

const updateSelection = (
  selection: TirePressureSelection,
  change: TirePressureChange,
): TirePressureSelection => {
  if (change.field === 'bike') return { ...selection, bike: change.value }
  if (change.field === 'bikeMass')
    return {
      ...selection,
      bike: change.bike,
      bikeMassesLb: { ...selection.bikeMassesLb, [change.bike]: change.value },
    }
  if (change.field === 'balance') return { ...selection, balance: change.value }
  if (change.field === 'wheel') return { ...selection, wheel: change.value }
  if (change.field === 'tire') return { ...selection, tire: change.value }
  if (change.field === 'surface') return { ...selection, surface: change.value }
  return { ...selection, speedMph: change.value }
}

export const setupTirePressure = (root: HTMLElement): (() => void) | null => {
  const calculators = Array.from(root.querySelectorAll<HTMLElement>('.tri-pressure'))
  if (calculators.length === 0) return null
  let selection = readTirePressureSelection()

  const render = (): void => {
    for (const calculator of calculators) {
      calculator.dataset.bike = selection.bike
      calculator.dataset.bikeMassLb = String(selection.bikeMassesLb[selection.bike])
      calculator.dataset.balance = selection.balance
      calculator.dataset.wheel = selection.wheel
      calculator.dataset.tire = selection.tire
      calculator.dataset.surface = selection.surface
      calculator.dataset.speedMph = String(selection.speedMph)
      renderSurfaceTip(calculator, selection.surface)
      for (const input of calculator.querySelectorAll<HTMLInputElement>(
        'input[data-pressure-field]',
      )) {
        const field = input.dataset.pressureField
        if (field === 'bike') input.checked = input.value === selection.bike
        else if (field === 'bikeMass' && document.activeElement !== input) {
          const bike = input.dataset.pressureBike
          if (bike && isTirePressureBikeId(bike)) input.value = String(selection.bikeMassesLb[bike])
        } else if (field === 'balance') input.checked = input.value === selection.balance
        else if (field === 'wheel') input.checked = input.value === selection.wheel
        else if (field === 'tire') input.checked = input.value === selection.tire
        else if (field === 'surface') input.checked = input.value === selection.surface
        else if (field === 'speed' && document.activeElement !== input)
          input.value = String(selection.speedMph)
      }

      const riderKg = Number(calculator.dataset.riderKg)
      const recommendation = calculateTirePressure(riderKg, selection)
      const front = calculator.querySelector<HTMLOutputElement>('[data-pressure-output="front"]')
      const rear = calculator.querySelector<HTMLOutputElement>('[data-pressure-output="rear"]')
      const system = calculator.querySelector<HTMLElement>('[data-pressure-system]')
      const warning = calculator.querySelector<HTMLElement>('[data-pressure-warning]')
      if (front)
        front.textContent = recommendation ? formatTirePressurePsi(recommendation.frontPsi) : '—'
      if (rear)
        rear.textContent = recommendation ? formatTirePressurePsi(recommendation.rearPsi) : '—'
      if (system)
        system.textContent = recommendation
          ? `${recommendation.riderKg.toFixed(1)} + ${recommendation.bikeKg.toFixed(1)} = ${recommendation.systemKg.toFixed(1)} kg system`
          : 'add a morning body-composition measurement'
      if (warning) warning.hidden = !recommendation?.wheelCompatibilityWarning
      calculator.dataset.frontPsi = recommendation ? String(recommendation.frontPsi) : ''
      calculator.dataset.rearPsi = recommendation ? String(recommendation.rearPsi) : ''
      calculator.dataset.systemKg = recommendation ? recommendation.systemKg.toFixed(1) : ''
      calculator.classList.toggle('tri-pressure--unavailable', recommendation === null)
    }
  }

  const apply = (change: TirePressureChange): void => {
    selection = updateSelection(selection, change)
    storeTirePressureSelection(selection)
    render()
  }

  const onChange = (event: Event): void => {
    if (!(event.target instanceof HTMLInputElement)) return
    const field = event.target.dataset.pressureField
    if (field === 'bike' && isTirePressureBikeId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'wheel' && isTirePressureWheelId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'tire' && isTirePressureTireId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'balance' && isTirePressureBalanceId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'surface' && isTirePressureSurfaceId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'bikeMass') {
      const bike = event.target.dataset.pressureBike
      const mass = Number(event.target.value)
      if (bike && isTirePressureBikeId(bike) && isTirePressureBikeMassLb(mass))
        apply({ field, bike, value: mass })
      else if (bike && isTirePressureBikeId(bike))
        event.target.value = String(selection.bikeMassesLb[bike])
    } else if (field === 'speed') {
      const speed = Number(event.target.value)
      if (isTirePressureSpeed(speed)) apply({ field, value: speed })
      else event.target.value = String(selection.speedMph)
    }
  }

  const onInput = (event: Event): void => {
    if (!(event.target instanceof HTMLInputElement)) return
    const field = event.target.dataset.pressureField
    if (field === 'speed') {
      const speed = Number(event.target.value)
      if (isTirePressureSpeed(speed)) apply({ field, value: speed })
    } else if (field === 'bikeMass') {
      const bike = event.target.dataset.pressureBike
      const mass = Number(event.target.value)
      if (bike && isTirePressureBikeId(bike) && isTirePressureBikeMassLb(mass))
        apply({ field, bike, value: mass })
    }
  }

  const onExternalChange = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return
    const detail: unknown = event.detail
    if (isTirePressureChange(detail)) apply(detail)
  }

  const onFocusIn = (event: FocusEvent): void => {
    if (!(event.target instanceof HTMLInputElement)) return
    const field = event.target.dataset.pressureField
    if (field === 'speed' || field === 'bikeMass') event.target.select()
    else if (field === 'surface' && isTirePressureSurfaceId(event.target.value)) {
      const calculator = event.target.closest<HTMLElement>('.tri-pressure')
      if (calculator) renderSurfaceTip(calculator, event.target.value)
    }
  }

  const onPointerOver = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const option = event.target.closest<HTMLElement>('[data-pressure-surface-option]')
    const surfaceId = option?.dataset.pressureSurfaceOption
    if (!option || !surfaceId || !isTirePressureSurfaceId(surfaceId)) return
    const calculator = option.closest<HTMLElement>('.tri-pressure')
    if (calculator) renderSurfaceTip(calculator, surfaceId)
  }

  const onKeyDown = (event: KeyboardEvent): void => {
    if (!(event.target instanceof HTMLInputElement)) return
    const field = event.target.dataset.pressureField
    if ((field !== 'speed' && field !== 'bikeMass') || event.key !== 'Enter') return
    event.preventDefault()
    event.target.blur()
  }

  const onOpen = (): void => {
    const calculator = calculators[0]
    if (!calculator) return
    const wrap = calculator.closest<HTMLElement>('.tri-gear-wrap')
    const trigger = wrap?.querySelector<HTMLButtonElement>('.tri-gear-btn')
    if (trigger && trigger.getAttribute('aria-expanded') !== 'true') trigger.click()
    requestAnimationFrame(() => {
      calculator.scrollIntoView({ block: 'nearest' })
      calculator.querySelector<HTMLInputElement>('input[data-pressure-field="bike"]')?.focus()
    })
  }

  root.addEventListener('change', onChange)
  root.addEventListener('input', onInput)
  root.addEventListener('focusin', onFocusIn)
  root.addEventListener('pointerover', onPointerOver)
  root.addEventListener('keydown', onKeyDown)
  root.addEventListener(TRI_TIRE_PRESSURE_CHANGE_EVENT, onExternalChange)
  root.addEventListener(TRI_TIRE_PRESSURE_OPEN_EVENT, onOpen)
  render()
  return () => {
    root.removeEventListener('change', onChange)
    root.removeEventListener('input', onInput)
    root.removeEventListener('focusin', onFocusIn)
    root.removeEventListener('pointerover', onPointerOver)
    root.removeEventListener('keydown', onKeyDown)
    root.removeEventListener(TRI_TIRE_PRESSURE_CHANGE_EVENT, onExternalChange)
    root.removeEventListener(TRI_TIRE_PRESSURE_OPEN_EVENT, onOpen)
  }
}
