export type TirePressureBikeId = 'cervelo' | 'speedmax' | 'custom'
export type TirePressureWheelId = 'hunt' | 'reserve' | 'custom'
export type TirePressureTireId = 'tpu' | 'tubeless'
export type TirePressureBalanceId = '50-50' | '48-52' | '47-53' | '46.5-53.5'
export type TirePressureWeightUnit = 'kg' | 'lb'
export type TirePressureWheelAxle = 'front' | 'rear'
export type TirePressureSurfaceId =
  | 'new-pavement'
  | 'worn-pavement'
  | 'poor-pavement'
  | 'cobblestone'

export interface TirePressureBike {
  id: TirePressureBikeId
  label: string
  massLb: number
}

export interface TirePressureBalance {
  id: TirePressureBalanceId
  label: string
  frontPercent: number
  rearPercent: number
  frontPressureCoefficient: number
  rearPressureCoefficient: number
}

export interface TirePressureWheel {
  id: TirePressureWheelId
  label: string
  diameterMm: number
  frontInnerWidthMm: number
  rearInnerWidthMm: number
  recommendedMinimumTireWidthMm: number | null
}

export interface TirePressureCustomWheel {
  frontInnerWidthMm: number
  rearInnerWidthMm: number
}

export interface TirePressureMeasuredTire {
  frontWidthMm: number
  rearWidthMm: number
}

export interface TirePressureSurface {
  id: TirePressureSurfaceId
  label: string
  coefficient: number
  note: string
}

export interface TirePressureTire {
  id: TirePressureTireId
  label: string
  detail: string
  pressureCoefficient: number
}

export interface TirePressureSpeed {
  label: string
  mph: number
}

export interface TirePressureSelection {
  riderKg: number | null
  weightUnit: TirePressureWeightUnit
  bike: TirePressureBikeId
  bikeMassesLb: Readonly<Record<TirePressureBikeId, number>>
  balance: TirePressureBalanceId
  wheel: TirePressureWheelId
  customWheel: TirePressureCustomWheel
  measuredTire: TirePressureMeasuredTire
  tire: TirePressureTireId
  surface: TirePressureSurfaceId
  speedMph: number
}

export type TirePressureChange =
  | { field: 'riderMass'; valueKg: number }
  | { field: 'weightUnit'; value: TirePressureWeightUnit }
  | { field: 'bike'; value: TirePressureBikeId }
  | { field: 'bikeMass'; bike: TirePressureBikeId; value: number }
  | { field: 'balance'; value: TirePressureBalanceId }
  | { field: 'wheel'; value: TirePressureWheelId }
  | { field: 'customWheelWidth'; axle: TirePressureWheelAxle; value: number }
  | { field: 'measuredTireWidth'; axle: TirePressureWheelAxle; value: number }
  | { field: 'tire'; value: TirePressureTireId }
  | { field: 'surface'; value: TirePressureSurfaceId }
  | { field: 'speed'; value: number }

export interface TirePressureRecommendation {
  frontPsi: number
  rearPsi: number
  riderKg: number
  bikeMassLb: number
  bikeKg: number
  systemKg: number
  frontMeasuredWidthMm: number
  rearMeasuredWidthMm: number
  diameterMm: number
  wheel: TirePressureWheel
  bike: TirePressureBike
  balance: TirePressureBalance
  tire: TirePressureTire
  surface: TirePressureSurface
  speedMph: number
  wheelCompatibilityWarning: boolean
}

export interface MorningBodyWeight {
  date: string
  kg: number
}

export interface BodyCompositionWeight {
  date: string
  kg: number | null
}

export const KG_PER_LB = 0.45359237
export const TRI_TIRE_PRESSURE_CHANGE_EVENT = 'tri:tire-pressure-change'
export const TRI_TIRE_PRESSURE_OPEN_EVENT = 'tri:tire-pressure-open'
export const TIRE_PRESSURE_SOURCE_URL = 'https://silca.cc/en-ca/pages/pro-tire-pressure-calculator'
export const PIRELLI_PRESSURE_SOURCE_URL = 'https://www.pirelli.com/tires/en-us/bike/pressure-tool'
export const TIRE_PRESSURE_WEIGHT_UNITS: readonly TirePressureWeightUnit[] = ['kg', 'lb']

export const TIRE_PRESSURE_BIKES: readonly TirePressureBike[] = [
  { id: 'cervelo', label: 'Cervélo Soloist', massLb: 26.2 },
  { id: 'speedmax', label: 'Canyon Speedmax', massLb: 26 },
  { id: 'custom', label: 'Custom', massLb: 20 },
]

export const TIRE_PRESSURE_BALANCES: readonly TirePressureBalance[] = [
  {
    id: '50-50',
    label: '50 / 50',
    frontPercent: 50,
    rearPercent: 50,
    frontPressureCoefficient: 1,
    rearPressureCoefficient: 1,
  },
  {
    id: '48-52',
    label: '48 / 52',
    frontPercent: 48,
    rearPercent: 52,
    frontPressureCoefficient: 0.985,
    rearPressureCoefficient: 1.01,
  },
  {
    id: '47-53',
    label: '47 / 53',
    frontPercent: 47,
    rearPercent: 53,
    frontPressureCoefficient: 0.975,
    rearPressureCoefficient: 1.02,
  },
  {
    id: '46.5-53.5',
    label: '46.5 / 53.5',
    frontPercent: 46.5,
    rearPercent: 53.5,
    frontPressureCoefficient: 0.97,
    rearPressureCoefficient: 1.03,
  },
]

export const TIRE_PRESSURE_WHEELS: readonly TirePressureWheel[] = [
  {
    id: 'hunt',
    label: 'HUNT 54_58 Aerodynamicist UD',
    diameterMm: 622,
    frontInnerWidthMm: 22,
    rearInnerWidthMm: 22,
    recommendedMinimumTireWidthMm: 25,
  },
  {
    id: 'reserve',
    label: 'Reserve 42|49 TA',
    diameterMm: 622,
    frontInnerWidthMm: 25.4,
    rearInnerWidthMm: 24.8,
    recommendedMinimumTireWidthMm: 29,
  },
  {
    id: 'custom',
    label: 'Custom Wheelset',
    diameterMm: 622,
    frontInnerWidthMm: 23,
    rearInnerWidthMm: 23,
    recommendedMinimumTireWidthMm: null,
  },
]

export const TIRE_PRESSURE_SURFACES: readonly TirePressureSurface[] = [
  {
    id: 'new-pavement',
    label: 'new pavement',
    coefficient: 261,
    note: 'fresh resurfacing, sealed and smooth',
  },
  {
    id: 'worn-pavement',
    label: 'worn pavement',
    coefficient: 246.5,
    note: 'most Toronto roads, aged asphalt and seams',
  },
  {
    id: 'poor-pavement',
    label: 'poor pavement',
    coefficient: 225,
    note: 'rough Toronto streets, patches, cracks and potholes',
  },
  {
    id: 'cobblestone',
    label: 'cobblestone',
    coefficient: 199,
    note: 'setts, deep joints and persistent vibration',
  },
]

export const TIRE_PRESSURE_TIRES: readonly TirePressureTire[] = [
  { id: 'tpu', label: 'P Zero Race SL-R', detail: 'P Zero TPU tube', pressureCoefficient: 1 },
  { id: 'tubeless', label: 'P Zero Race TLR SL-R', detail: 'tubeless', pressureCoefficient: 1 },
]

export const TIRE_PRESSURE_SPEEDS: readonly TirePressureSpeed[] = [
  { label: 'endurance', mph: 17 },
  { label: 'fast group', mph: 19.5 },
  { label: 'race', mph: 23 },
]

export const DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB: Readonly<Record<TirePressureBikeId, number>> = {
  cervelo: 26.2,
  speedmax: 26,
  custom: 20,
}

export const DEFAULT_TIRE_PRESSURE_SELECTION: TirePressureSelection = {
  riderKg: null,
  weightUnit: 'kg',
  bike: 'cervelo',
  bikeMassesLb: DEFAULT_TIRE_PRESSURE_BIKE_MASSES_LB,
  balance: '48-52',
  wheel: 'hunt',
  customWheel: { frontInnerWidthMm: 23, rearInnerWidthMm: 23 },
  measuredTire: { frontWidthMm: 32, rearWidthMm: 28 },
  tire: 'tpu',
  surface: 'worn-pavement',
  speedMph: 19.5,
}

export const isTirePressureBikeId = (value: string): value is TirePressureBikeId =>
  TIRE_PRESSURE_BIKES.some(bike => bike.id === value)

export const isTirePressureWheelId = (value: string): value is TirePressureWheelId =>
  TIRE_PRESSURE_WHEELS.some(wheel => wheel.id === value)

export const isTirePressureTireId = (value: string): value is TirePressureTireId =>
  TIRE_PRESSURE_TIRES.some(tire => tire.id === value)

export const isTirePressureSurfaceId = (value: string): value is TirePressureSurfaceId =>
  TIRE_PRESSURE_SURFACES.some(surface => surface.id === value)

export const isTirePressureBalanceId = (value: string): value is TirePressureBalanceId =>
  TIRE_PRESSURE_BALANCES.some(balance => balance.id === value)

export const isTirePressureWeightUnit = (value: string): value is TirePressureWeightUnit =>
  value === 'kg' || value === 'lb'

export const isTirePressureSpeed = (value: number): boolean =>
  Number.isFinite(value) && value >= 10 && value <= 33

export const isTirePressureBikeMassLb = (value: number): boolean =>
  Number.isFinite(value) && value >= 10 && value <= 80

export const isTirePressureRiderKg = (value: number): boolean =>
  Number.isFinite(value) && value >= 25 && value <= 200

export const isTirePressureInnerWidthMm = (value: number): boolean =>
  Number.isFinite(value) && value >= 13 && value <= 35

export const isTirePressureMeasuredWidthMm = (value: number): boolean =>
  Number.isInteger(value) && value >= 20 && value <= 65

export const tirePressureWeightFromKg = (valueKg: number, unit: TirePressureWeightUnit): number =>
  unit === 'kg' ? valueKg : valueKg / KG_PER_LB

export const tirePressureWeightToKg = (value: number, unit: TirePressureWeightUnit): number =>
  unit === 'kg' ? value : value * KG_PER_LB

export const formatTirePressureWeight = (valueKg: number, unit: TirePressureWeightUnit): string =>
  tirePressureWeightFromKg(valueKg, unit).toFixed(unit === 'kg' ? 2 : 1)

export const isTirePressureChange = (value: unknown): value is TirePressureChange => {
  if (value === null || typeof value !== 'object' || !('field' in value)) return false
  if (value.field === 'riderMass')
    return (
      'valueKg' in value &&
      typeof value.valueKg === 'number' &&
      isTirePressureRiderKg(value.valueKg)
    )
  if (!('value' in value)) return false
  if (value.field === 'weightUnit')
    return typeof value.value === 'string' && isTirePressureWeightUnit(value.value)
  if (value.field === 'bike')
    return typeof value.value === 'string' && isTirePressureBikeId(value.value)
  if (value.field === 'bikeMass')
    return (
      'bike' in value &&
      typeof value.bike === 'string' &&
      isTirePressureBikeId(value.bike) &&
      typeof value.value === 'number' &&
      isTirePressureBikeMassLb(value.value)
    )
  if (value.field === 'balance')
    return typeof value.value === 'string' && isTirePressureBalanceId(value.value)
  if (value.field === 'wheel')
    return typeof value.value === 'string' && isTirePressureWheelId(value.value)
  if (value.field === 'customWheelWidth')
    return (
      'axle' in value &&
      (value.axle === 'front' || value.axle === 'rear') &&
      typeof value.value === 'number' &&
      isTirePressureInnerWidthMm(value.value)
    )
  if (value.field === 'measuredTireWidth')
    return (
      'axle' in value &&
      (value.axle === 'front' || value.axle === 'rear') &&
      typeof value.value === 'number' &&
      isTirePressureMeasuredWidthMm(value.value)
    )
  if (value.field === 'tire')
    return typeof value.value === 'string' && isTirePressureTireId(value.value)
  if (value.field === 'surface')
    return typeof value.value === 'string' && isTirePressureSurfaceId(value.value)
  return (
    value.field === 'speed' && typeof value.value === 'number' && isTirePressureSpeed(value.value)
  )
}

export const tirePressureBike = (id: TirePressureBikeId): TirePressureBike =>
  TIRE_PRESSURE_BIKES.find(bike => bike.id === id) ?? TIRE_PRESSURE_BIKES[0]

export const tirePressureBalance = (id: TirePressureBalanceId): TirePressureBalance =>
  TIRE_PRESSURE_BALANCES.find(balance => balance.id === id) ?? TIRE_PRESSURE_BALANCES[0]

export const tirePressureWheel = (id: TirePressureWheelId): TirePressureWheel =>
  TIRE_PRESSURE_WHEELS.find(wheel => wheel.id === id) ?? TIRE_PRESSURE_WHEELS[0]

export const selectedTirePressureWheel = (selection: TirePressureSelection): TirePressureWheel => {
  const wheel = tirePressureWheel(selection.wheel)
  return wheel.id === 'custom' ? { ...wheel, ...selection.customWheel } : wheel
}

export const tirePressureTire = (id: TirePressureTireId): TirePressureTire =>
  TIRE_PRESSURE_TIRES.find(tire => tire.id === id) ?? TIRE_PRESSURE_TIRES[0]

export const tirePressureSurface = (id: TirePressureSurfaceId): TirePressureSurface =>
  TIRE_PRESSURE_SURFACES.find(surface => surface.id === id) ?? TIRE_PRESSURE_SURFACES[0]

export const latestMorningBodyWeight = (
  composition: readonly BodyCompositionWeight[],
): MorningBodyWeight | null => {
  let latest: MorningBodyWeight | null = null
  for (const sample of composition) {
    if (sample.kg == null || !Number.isFinite(sample.kg) || sample.kg <= 0) continue
    if (!latest || sample.date > latest.date) latest = { date: sample.date, kg: sample.kg }
  }
  return latest
}

const roundHalfPsi = (value: number): number => Math.round(value * 2) / 2

const tirePressureForWidth = (
  width: number,
  diameter: number,
  stiffness: number,
  speedCoefficient: number,
  loadCoefficient: number,
  tireCoefficient: number,
): number => {
  const numerator =
    (-0.00006 * width ** 3 + 0.0079 * width ** 2 - 0.4102 * width + 12.725) * -226.44
  const unloadedRadius = width + diameter / 2
  const loadedRadius = (-0.5 * 9.81) / (stiffness * (20 / width)) + unloadedRadius
  const denominator = loadedRadius ** 2 - unloadedRadius ** 2
  return roundHalfPsi(
    (numerator / denominator) * speedCoefficient * loadCoefficient * tireCoefficient,
  )
}

export const calculateTirePressure = (
  selection: TirePressureSelection,
): TirePressureRecommendation | null => {
  const riderKg = selection.riderKg
  if (riderKg == null || !isTirePressureRiderKg(riderKg)) return null
  if (
    selection.wheel === 'custom' &&
    (!isTirePressureInnerWidthMm(selection.customWheel.frontInnerWidthMm) ||
      !isTirePressureInnerWidthMm(selection.customWheel.rearInnerWidthMm))
  )
    return null
  if (
    !isTirePressureMeasuredWidthMm(selection.measuredTire.frontWidthMm) ||
    !isTirePressureMeasuredWidthMm(selection.measuredTire.rearWidthMm)
  )
    return null
  const bike = tirePressureBike(selection.bike)
  const balance = tirePressureBalance(selection.balance)
  const wheel = selectedTirePressureWheel(selection)
  const tire = tirePressureTire(selection.tire)
  const surface = tirePressureSurface(selection.surface)
  const bikeMassLb = selection.bikeMassesLb[selection.bike]
  if (!isTirePressureBikeMassLb(bikeMassLb)) return null
  const bikeKg = bikeMassLb * KG_PER_LB
  const systemKg = riderKg + bikeKg
  if (systemKg < 34 || systemKg > 205) return null
  if (!isTirePressureSpeed(selection.speedMph)) return null

  const stiffness = 0.5 * (systemKg - 50) + surface.coefficient
  const speedCoefficient = 0.97 + (selection.speedMph - 10) * (0.06 / 23)
  const frontPsi = tirePressureForWidth(
    selection.measuredTire.frontWidthMm,
    wheel.diameterMm,
    stiffness,
    speedCoefficient,
    balance.frontPressureCoefficient,
    tire.pressureCoefficient,
  )
  const rearPsi = tirePressureForWidth(
    selection.measuredTire.rearWidthMm,
    wheel.diameterMm,
    stiffness,
    speedCoefficient,
    balance.rearPressureCoefficient,
    tire.pressureCoefficient,
  )
  if (!Number.isFinite(frontPsi) || !Number.isFinite(rearPsi)) return null

  return {
    frontPsi,
    rearPsi,
    riderKg,
    bikeMassLb,
    bikeKg,
    systemKg,
    frontMeasuredWidthMm: selection.measuredTire.frontWidthMm,
    rearMeasuredWidthMm: selection.measuredTire.rearWidthMm,
    diameterMm: wheel.diameterMm,
    wheel,
    bike,
    balance,
    tire,
    surface,
    speedMph: selection.speedMph,
    wheelCompatibilityWarning:
      wheel.recommendedMinimumTireWidthMm != null &&
      (selection.measuredTire.frontWidthMm < wheel.recommendedMinimumTireWidthMm ||
        selection.measuredTire.rearWidthMm < wheel.recommendedMinimumTireWidthMm),
  }
}

export const formatTirePressurePsi = (value: number): string =>
  Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)
