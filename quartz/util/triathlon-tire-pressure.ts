export type TirePressureBikeId = 'cervelo' | 'speedmax'
export type TirePressureWheelId = 'princeton' | 'reserve'
export type TirePressureTireId = 'tpu' | 'tubeless'
export type TirePressureSurfaceId =
  | 'new-pavement'
  | 'worn-pavement'
  | 'poor-pavement'
  | 'cobblestone'

export interface TirePressureBike {
  id: TirePressureBikeId
  label: string
  massLb: number
  frontLoadCoefficient: number
  rearLoadCoefficient: number
}

export interface TirePressureWheel {
  id: TirePressureWheelId
  label: string
  diameterMm: number
  frontInnerWidthMm: number
  rearInnerWidthMm: number
  recommendedMinimumTireWidthMm: number
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
  bike: TirePressureBikeId
  wheel: TirePressureWheelId
  tire: TirePressureTireId
  surface: TirePressureSurfaceId
  speedMph: number
}

export type TirePressureChange =
  | { field: 'bike'; value: TirePressureBikeId }
  | { field: 'wheel'; value: TirePressureWheelId }
  | { field: 'tire'; value: TirePressureTireId }
  | { field: 'surface'; value: TirePressureSurfaceId }
  | { field: 'speed'; value: number }

export interface TirePressureRecommendation {
  frontPsi: number
  rearPsi: number
  riderKg: number
  bikeKg: number
  systemKg: number
  measuredWidthMm: number
  diameterMm: number
  wheel: TirePressureWheel
  bike: TirePressureBike
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
export const TIRE_PRESSURE_MEASURED_WIDTH_MM = 28
export const TRI_TIRE_PRESSURE_CHANGE_EVENT = 'tri:tire-pressure-change'
export const TRI_TIRE_PRESSURE_OPEN_EVENT = 'tri:tire-pressure-open'
export const TIRE_PRESSURE_SOURCE_URL =
  'https://silca.cc/en-ca/pages/pro-tire-pressure-calculator?_eab=1'
export const PIRELLI_PRESSURE_SOURCE_URL = 'https://www.pirelli.com/tires/en-us/bike/pressure-tool'

export const TIRE_PRESSURE_BIKES: readonly TirePressureBike[] = [
  {
    id: 'cervelo',
    label: 'Cervélo Soloist',
    massLb: 22,
    frontLoadCoefficient: 0.985,
    rearLoadCoefficient: 1.01,
  },
  {
    id: 'speedmax',
    label: 'Canyon Speedmax',
    massLb: 26,
    frontLoadCoefficient: 1,
    rearLoadCoefficient: 1,
  },
]

export const TIRE_PRESSURE_WHEELS: readonly TirePressureWheel[] = [
  {
    id: 'princeton',
    label: 'Princeton Mach 7580',
    diameterMm: 622,
    frontInnerWidthMm: 22,
    rearInnerWidthMm: 22,
    recommendedMinimumTireWidthMm: 28,
  },
  {
    id: 'reserve',
    label: 'Reserve 42|49 TA',
    diameterMm: 622,
    frontInnerWidthMm: 25.4,
    rearInnerWidthMm: 24.8,
    recommendedMinimumTireWidthMm: 29,
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

export const DEFAULT_TIRE_PRESSURE_SELECTION: TirePressureSelection = {
  bike: 'cervelo',
  wheel: 'princeton',
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

export const isTirePressureSpeed = (value: number): boolean =>
  Number.isFinite(value) && value >= 10 && value <= 33

export const isTirePressureChange = (value: unknown): value is TirePressureChange => {
  if (value === null || typeof value !== 'object' || !('field' in value) || !('value' in value))
    return false
  if (value.field === 'bike')
    return typeof value.value === 'string' && isTirePressureBikeId(value.value)
  if (value.field === 'wheel')
    return typeof value.value === 'string' && isTirePressureWheelId(value.value)
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

export const tirePressureWheel = (id: TirePressureWheelId): TirePressureWheel =>
  TIRE_PRESSURE_WHEELS.find(wheel => wheel.id === id) ?? TIRE_PRESSURE_WHEELS[0]

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

export const calculateTirePressure = (
  riderKg: number,
  selection: TirePressureSelection,
): TirePressureRecommendation | null => {
  const bike = tirePressureBike(selection.bike)
  const wheel = tirePressureWheel(selection.wheel)
  const tire = tirePressureTire(selection.tire)
  const surface = tirePressureSurface(selection.surface)
  const bikeKg = bike.massLb * KG_PER_LB
  const systemKg = riderKg + bikeKg
  if (!Number.isFinite(riderKg) || systemKg < 34 || systemKg > 205) return null
  if (!isTirePressureSpeed(selection.speedMph)) return null

  const width = TIRE_PRESSURE_MEASURED_WIDTH_MM
  const stiffness = 0.5 * (systemKg - 50) + surface.coefficient
  const numerator =
    (-0.00006 * width ** 3 + 0.0079 * width ** 2 - 0.4102 * width + 12.725) * -226.44
  const loadedRadius = (-0.5 * 9.81) / (stiffness * (20 / width)) + (width + wheel.diameterMm / 2)
  const denominator = loadedRadius ** 2 - (width + wheel.diameterMm / 2) ** 2
  const centerPressurePsi = numerator / denominator
  const speedCoefficient = 0.97 + (selection.speedMph - 10) * (0.06 / 23)
  const frontPsi = roundHalfPsi(
    centerPressurePsi * speedCoefficient * bike.frontLoadCoefficient * tire.pressureCoefficient,
  )
  const rearPsi = roundHalfPsi(
    centerPressurePsi * speedCoefficient * bike.rearLoadCoefficient * tire.pressureCoefficient,
  )
  if (!Number.isFinite(frontPsi) || !Number.isFinite(rearPsi)) return null

  return {
    frontPsi,
    rearPsi,
    riderKg,
    bikeKg,
    systemKg,
    measuredWidthMm: width,
    diameterMm: wheel.diameterMm,
    wheel,
    bike,
    tire,
    surface,
    speedMph: selection.speedMph,
    wheelCompatibilityWarning: width < wheel.recommendedMinimumTireWidthMm,
  }
}

export const formatTirePressurePsi = (value: number): string =>
  Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)
