import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressurePsi,
  formatTirePressureWeight,
  isTirePressureBalanceId,
  isTirePressureBikeId,
  isTirePressureBikeMassLb,
  isTirePressureChange,
  isTirePressureInnerWidthMm,
  isTirePressureMeasuredWidthMm,
  isTirePressureRiderKg,
  isTirePressureSpeed,
  isTirePressureSurfaceId,
  isTirePressureTireId,
  isTirePressureWheelId,
  isTirePressureWeightUnit,
  tirePressureSurface,
  tirePressureWeightToKg,
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
const TIRE_PRESSURE_CUSTOM_WHEEL_FRONT_KEY = 'triathlon-tire-pressure-wheel-custom-front'
const TIRE_PRESSURE_CUSTOM_WHEEL_REAR_KEY = 'triathlon-tire-pressure-wheel-custom-rear'
const TIRE_PRESSURE_MEASURED_TIRE_FRONT_KEY = 'triathlon-tire-pressure-measured-front'
const TIRE_PRESSURE_MEASURED_TIRE_REAR_KEY = 'triathlon-tire-pressure-measured-rear'
const TIRE_PRESSURE_TIRE_KEY = 'triathlon-tire-pressure-tire'
const TIRE_PRESSURE_SURFACE_KEY = 'triathlon-tire-pressure-surface'
const TIRE_PRESSURE_SPEED_KEY = 'triathlon-tire-pressure-speed'
const TIRE_PRESSURE_RIDER_KG_KEY = 'triathlon-tire-pressure-rider-kg'
const TIRE_PRESSURE_RIDER_DATE_KEY = 'triathlon-tire-pressure-rider-date'
const TIRE_PRESSURE_WEIGHT_UNIT_KEY = 'triathlon-tire-pressure-weight-unit'

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

export const readTirePressureSelection = (
  fallbackRiderKg: number | null = null,
  weightDate?: string,
): TirePressureSelection => {
  const storedBike = localStorage.getItem(TIRE_PRESSURE_BIKE_KEY)
  const storedWheel = localStorage.getItem(TIRE_PRESSURE_WHEEL_KEY)
  const storedBalance = localStorage.getItem(TIRE_PRESSURE_BALANCE_KEY)
  const storedTire = localStorage.getItem(TIRE_PRESSURE_TIRE_KEY)
  const storedSurface = localStorage.getItem(TIRE_PRESSURE_SURFACE_KEY)
  const storedSpeed = Number(localStorage.getItem(TIRE_PRESSURE_SPEED_KEY))
  const storedRiderKg = Number(localStorage.getItem(TIRE_PRESSURE_RIDER_KG_KEY))
  const storedRiderDate = localStorage.getItem(TIRE_PRESSURE_RIDER_DATE_KEY)
  const storedWeightUnit = localStorage.getItem(TIRE_PRESSURE_WEIGHT_UNIT_KEY)
  const storedCustomFront = Number(localStorage.getItem(TIRE_PRESSURE_CUSTOM_WHEEL_FRONT_KEY))
  const storedCustomRear = Number(localStorage.getItem(TIRE_PRESSURE_CUSTOM_WHEEL_REAR_KEY))
  const storedMeasuredFront = Number(localStorage.getItem(TIRE_PRESSURE_MEASURED_TIRE_FRONT_KEY))
  const storedMeasuredRear = Number(localStorage.getItem(TIRE_PRESSURE_MEASURED_TIRE_REAR_KEY))
  const useStoredRider =
    isTirePressureRiderKg(storedRiderKg) && (!weightDate || storedRiderDate === weightDate)
  return {
    riderKg: useStoredRider
      ? storedRiderKg
      : fallbackRiderKg != null && isTirePressureRiderKg(fallbackRiderKg)
        ? fallbackRiderKg
        : null,
    weightUnit:
      storedWeightUnit && isTirePressureWeightUnit(storedWeightUnit)
        ? storedWeightUnit
        : DEFAULT_TIRE_PRESSURE_SELECTION.weightUnit,
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
    customWheel: {
      frontInnerWidthMm: isTirePressureInnerWidthMm(storedCustomFront)
        ? storedCustomFront
        : DEFAULT_TIRE_PRESSURE_SELECTION.customWheel.frontInnerWidthMm,
      rearInnerWidthMm: isTirePressureInnerWidthMm(storedCustomRear)
        ? storedCustomRear
        : DEFAULT_TIRE_PRESSURE_SELECTION.customWheel.rearInnerWidthMm,
    },
    measuredTire: {
      frontWidthMm: isTirePressureMeasuredWidthMm(storedMeasuredFront)
        ? storedMeasuredFront
        : DEFAULT_TIRE_PRESSURE_SELECTION.measuredTire.frontWidthMm,
      rearWidthMm: isTirePressureMeasuredWidthMm(storedMeasuredRear)
        ? storedMeasuredRear
        : DEFAULT_TIRE_PRESSURE_SELECTION.measuredTire.rearWidthMm,
    },
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

export const storeTirePressureSelection = (
  selection: TirePressureSelection,
  weightDate?: string,
): void => {
  if (selection.riderKg != null) {
    localStorage.setItem(TIRE_PRESSURE_RIDER_KG_KEY, String(selection.riderKg))
    localStorage.setItem(TIRE_PRESSURE_RIDER_DATE_KEY, weightDate ?? '')
  }
  localStorage.setItem(TIRE_PRESSURE_WEIGHT_UNIT_KEY, selection.weightUnit)
  localStorage.setItem(TIRE_PRESSURE_BIKE_KEY, selection.bike)
  localStorage.setItem(TIRE_PRESSURE_CERVELO_MASS_KEY, String(selection.bikeMassesLb.cervelo))
  localStorage.setItem(TIRE_PRESSURE_SPEEDMAX_MASS_KEY, String(selection.bikeMassesLb.speedmax))
  localStorage.setItem(TIRE_PRESSURE_CUSTOM_MASS_KEY, String(selection.bikeMassesLb.custom))
  localStorage.setItem(TIRE_PRESSURE_BALANCE_KEY, selection.balance)
  localStorage.setItem(TIRE_PRESSURE_WHEEL_KEY, selection.wheel)
  localStorage.setItem(
    TIRE_PRESSURE_CUSTOM_WHEEL_FRONT_KEY,
    String(selection.customWheel.frontInnerWidthMm),
  )
  localStorage.setItem(
    TIRE_PRESSURE_CUSTOM_WHEEL_REAR_KEY,
    String(selection.customWheel.rearInnerWidthMm),
  )
  localStorage.setItem(
    TIRE_PRESSURE_MEASURED_TIRE_FRONT_KEY,
    String(selection.measuredTire.frontWidthMm),
  )
  localStorage.setItem(
    TIRE_PRESSURE_MEASURED_TIRE_REAR_KEY,
    String(selection.measuredTire.rearWidthMm),
  )
  localStorage.setItem(TIRE_PRESSURE_TIRE_KEY, selection.tire)
  localStorage.setItem(TIRE_PRESSURE_SURFACE_KEY, selection.surface)
  localStorage.setItem(TIRE_PRESSURE_SPEED_KEY, String(selection.speedMph))
}

const updateSelection = (
  selection: TirePressureSelection,
  change: TirePressureChange,
): TirePressureSelection => {
  if (change.field === 'riderMass') return { ...selection, riderKg: change.valueKg }
  if (change.field === 'weightUnit') return { ...selection, weightUnit: change.value }
  if (change.field === 'bike') return { ...selection, bike: change.value }
  if (change.field === 'bikeMass')
    return {
      ...selection,
      bike: change.bike,
      bikeMassesLb: { ...selection.bikeMassesLb, [change.bike]: change.value },
    }
  if (change.field === 'balance') return { ...selection, balance: change.value }
  if (change.field === 'wheel') return { ...selection, wheel: change.value }
  if (change.field === 'customWheelWidth')
    return {
      ...selection,
      wheel: 'custom',
      customWheel: {
        ...selection.customWheel,
        [change.axle === 'front' ? 'frontInnerWidthMm' : 'rearInnerWidthMm']: change.value,
      },
    }
  if (change.field === 'measuredTireWidth')
    return {
      ...selection,
      measuredTire: {
        ...selection.measuredTire,
        [change.axle === 'front' ? 'frontWidthMm' : 'rearWidthMm']: change.value,
      },
    }
  if (change.field === 'tire') return { ...selection, tire: change.value }
  if (change.field === 'surface') return { ...selection, surface: change.value }
  return { ...selection, speedMph: change.value }
}

export const setupTirePressure = (root: HTMLElement): (() => void) | null => {
  const calculators = Array.from(root.querySelectorAll<HTMLElement>('.tri-pressure'))
  if (calculators.length === 0) return null
  const initialRiderKg = Number(calculators[0]?.dataset.riderKg)
  const weightDate = calculators[0]?.dataset.weightDate
  let selection = readTirePressureSelection(
    isTirePressureRiderKg(initialRiderKg) ? initialRiderKg : null,
    weightDate,
  )

  const render = (): void => {
    for (const calculator of calculators) {
      calculator.dataset.bike = selection.bike
      calculator.dataset.riderKg = selection.riderKg == null ? '' : String(selection.riderKg)
      calculator.dataset.weightUnit = selection.weightUnit
      calculator.dataset.bikeMassLb = String(selection.bikeMassesLb[selection.bike])
      calculator.dataset.balance = selection.balance
      calculator.dataset.wheel = selection.wheel
      calculator.dataset.customWheelFrontMm = String(selection.customWheel.frontInnerWidthMm)
      calculator.dataset.customWheelRearMm = String(selection.customWheel.rearInnerWidthMm)
      calculator.dataset.measuredTireFrontMm = String(selection.measuredTire.frontWidthMm)
      calculator.dataset.measuredTireRearMm = String(selection.measuredTire.rearWidthMm)
      calculator.dataset.tire = selection.tire
      calculator.dataset.surface = selection.surface
      calculator.dataset.speedMph = String(selection.speedMph)
      renderSurfaceTip(calculator, selection.surface)
      for (const input of calculator.querySelectorAll<HTMLInputElement>(
        'input[data-pressure-field]',
      )) {
        const field = input.dataset.pressureField
        if (field === 'riderMass' && document.activeElement !== input)
          input.value =
            selection.riderKg == null
              ? ''
              : formatTirePressureWeight(selection.riderKg, selection.weightUnit)
        else if (field === 'weightUnit') input.checked = input.value === selection.weightUnit
        else if (field === 'bike') input.checked = input.value === selection.bike
        else if (field === 'bikeMass' && document.activeElement !== input) {
          const bike = input.dataset.pressureBike
          if (bike && isTirePressureBikeId(bike)) input.value = String(selection.bikeMassesLb[bike])
        } else if (field === 'balance') input.checked = input.value === selection.balance
        else if (field === 'wheel') input.checked = input.value === selection.wheel
        else if (field === 'customWheelWidth' && document.activeElement !== input) {
          const axle = input.dataset.pressureAxle
          if (axle === 'front') input.value = String(selection.customWheel.frontInnerWidthMm)
          else if (axle === 'rear') input.value = String(selection.customWheel.rearInnerWidthMm)
        } else if (field === 'measuredTireWidth' && document.activeElement !== input) {
          const axle = input.dataset.pressureAxle
          if (axle === 'front') input.value = String(selection.measuredTire.frontWidthMm)
          else if (axle === 'rear') input.value = String(selection.measuredTire.rearWidthMm)
        } else if (field === 'tire') input.checked = input.value === selection.tire
        else if (field === 'surface') input.checked = input.value === selection.surface
        else if (field === 'speed' && document.activeElement !== input)
          input.value = String(selection.speedMph)
      }

      const recommendation = calculateTirePressure(selection)
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
          ? `${formatTirePressureWeight(recommendation.riderKg, selection.weightUnit)} + ${formatTirePressureWeight(recommendation.bikeKg, selection.weightUnit)} = ${formatTirePressureWeight(recommendation.systemKg, selection.weightUnit)} ${selection.weightUnit} system`
          : 'add a morning body-composition measurement'
      if (warning) {
        warning.hidden = !recommendation?.wheelCompatibilityWarning
        const minimum = recommendation?.wheel.recommendedMinimumTireWidthMm
        if (recommendation?.wheelCompatibilityWarning && minimum != null) {
          const incompatible = [
            recommendation.frontMeasuredWidthMm < minimum
              ? `${recommendation.frontMeasuredWidthMm} mm front`
              : null,
            recommendation.rearMeasuredWidthMm < minimum
              ? `${recommendation.rearMeasuredWidthMm} mm rear`
              : null,
          ].filter(width => width != null)
          const tires =
            incompatible.length === 1
              ? `The ${incompatible[0]} tire sits`
              : `The ${incompatible.join(' and ')} tires sit`
          warning.textContent = `${recommendation.wheel.label} specifies ${minimum} mm as its minimum recommended tire width. ${tires} below that published range.`
        }
      }
      calculator.dataset.frontPsi = recommendation ? String(recommendation.frontPsi) : ''
      calculator.dataset.rearPsi = recommendation ? String(recommendation.rearPsi) : ''
      calculator.dataset.systemKg = recommendation ? recommendation.systemKg.toFixed(1) : ''
      calculator.classList.toggle('tri-pressure--unavailable', recommendation === null)
    }
  }

  const apply = (change: TirePressureChange): void => {
    selection = updateSelection(selection, change)
    storeTirePressureSelection(selection, weightDate)
    render()
  }

  const onChange = (event: Event): void => {
    if (!(event.target instanceof HTMLInputElement)) return
    const field = event.target.dataset.pressureField
    if (field === 'weightUnit' && isTirePressureWeightUnit(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'bike' && isTirePressureBikeId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'wheel' && isTirePressureWheelId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'tire' && isTirePressureTireId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'balance' && isTirePressureBalanceId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'surface' && isTirePressureSurfaceId(event.target.value))
      apply({ field, value: event.target.value })
    else if (field === 'riderMass') {
      const riderKg = tirePressureWeightToKg(Number(event.target.value), selection.weightUnit)
      if (isTirePressureRiderKg(riderKg)) apply({ field, valueKg: riderKg })
      else
        event.target.value =
          selection.riderKg == null
            ? ''
            : formatTirePressureWeight(selection.riderKg, selection.weightUnit)
    } else if (field === 'customWheelWidth') {
      const axle = event.target.dataset.pressureAxle
      const width = Number(event.target.value)
      if ((axle === 'front' || axle === 'rear') && isTirePressureInnerWidthMm(width))
        apply({ field, axle, value: width })
      else if (axle === 'front')
        event.target.value = String(selection.customWheel.frontInnerWidthMm)
      else if (axle === 'rear') event.target.value = String(selection.customWheel.rearInnerWidthMm)
    } else if (field === 'measuredTireWidth') {
      const axle = event.target.dataset.pressureAxle
      const width = Number(event.target.value)
      if ((axle === 'front' || axle === 'rear') && isTirePressureMeasuredWidthMm(width))
        apply({ field, axle, value: width })
      else if (axle === 'front') event.target.value = String(selection.measuredTire.frontWidthMm)
      else if (axle === 'rear') event.target.value = String(selection.measuredTire.rearWidthMm)
    } else if (field === 'bikeMass') {
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
    if (field === 'riderMass') {
      const riderKg = tirePressureWeightToKg(Number(event.target.value), selection.weightUnit)
      if (isTirePressureRiderKg(riderKg)) apply({ field, valueKg: riderKg })
    } else if (field === 'customWheelWidth') {
      const axle = event.target.dataset.pressureAxle
      const width = Number(event.target.value)
      if ((axle === 'front' || axle === 'rear') && isTirePressureInnerWidthMm(width))
        apply({ field, axle, value: width })
    } else if (field === 'measuredTireWidth') {
      const axle = event.target.dataset.pressureAxle
      const width = Number(event.target.value)
      if ((axle === 'front' || axle === 'rear') && isTirePressureMeasuredWidthMm(width))
        apply({ field, axle, value: width })
    } else if (field === 'speed') {
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
    if (
      field === 'riderMass' ||
      field === 'customWheelWidth' ||
      field === 'measuredTireWidth' ||
      field === 'speed' ||
      field === 'bikeMass'
    )
      event.target.select()
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
    if (
      (field !== 'riderMass' &&
        field !== 'customWheelWidth' &&
        field !== 'measuredTireWidth' &&
        field !== 'speed' &&
        field !== 'bikeMass') ||
      event.key !== 'Enter'
    )
      return
    event.preventDefault()
    event.target.blur()
  }

  const onOpen = (): void => {
    const calculator = calculators[0]
    if (!calculator) return
    calculator
      .closest<HTMLElement>('.tri-calc')
      ?.querySelector<HTMLButtonElement>('[data-calc-tab="tire-pressure"]')
      ?.click()
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
