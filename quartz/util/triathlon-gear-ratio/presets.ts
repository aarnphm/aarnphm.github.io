export interface GearCassettePreset {
  id: string
  label: string
  speeds: number
  cogs: readonly number[]
  maximumChainrings: 1 | 2
}

export interface GearCassettePresetGroup {
  label: string
  presets: readonly GearCassettePreset[]
}

export interface GearChainringPreset {
  id: string
  label: string
  chainrings: readonly [number, number]
}

export const cassette = (
  id: string,
  label: string,
  speeds: number,
  cogs: readonly number[],
  maximumChainrings: 1 | 2 = 2,
): GearCassettePreset => ({ id, label, speeds, cogs, maximumChainrings })

export const DEFAULT_GEAR_CASSETTE: GearCassettePreset = cassette(
  'shimano-ultegra-r8100-11-34',
  'Ultegra R8100 · 11–34 · 12s',
  12,
  [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34],
)

export const chainrings = (
  id: string,
  label: string,
  first: number,
  second: number,
): GearChainringPreset => ({ id, label, chainrings: [first, second] })

export const DEFAULT_GEAR_CHAINRING_PRESET: GearChainringPreset = chainrings(
  '54-40',
  '54–40',
  54,
  40,
)

export const GEAR_CHAINRING_PRESETS: readonly GearChainringPreset[] = [
  DEFAULT_GEAR_CHAINRING_PRESET,
  chainrings('53-39', '53–39', 53, 39),
  chainrings('52-36', '52–36', 52, 36),
]

export const gearChainringPreset = (id: string): GearChainringPreset | null => {
  return GEAR_CHAINRING_PRESETS.find(preset => preset.id === id) ?? null
}
