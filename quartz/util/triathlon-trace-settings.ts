export type TriathlonTraceSettings = Readonly<Record<string, boolean>>

const TRACE_NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/

export const TRIATHLON_TRACE_DISPLAY_SETTINGS = Object.freeze({
  detailed: Object.freeze({}),
  simplified: Object.freeze({
    'power-balance': false,
    'torque-effectiveness': false,
    'pedal-smoothness': false,
    'power-phase': false,
    'rider-position': false,
    stamina: false,
    'electronic-shifting': false,
    'heat-strain-index': false,
    'core-temperature': false,
    'skin-temperature': false,
    temperature: false,
  }),
})

export const triathlonTraceName = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

export const parseTriathlonTraceSettings = (
  value: string | undefined,
): TriathlonTraceSettings | null => {
  const raw = value?.startsWith('settings=') ? value.slice('settings='.length) : value
  if (!raw) return null
  if (raw === 'display:detailed') return TRIATHLON_TRACE_DISPLAY_SETTINGS.detailed
  if (raw === 'display:simplified') return TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified
  if (raw.startsWith('display:')) return null
  const settings: Record<string, boolean> = {}
  for (const entry of raw.split('&')) {
    const separator = entry.lastIndexOf(':')
    if (separator <= 0) return null
    const name = entry.slice(0, separator)
    const enabled = entry.slice(separator + 1)
    if (!TRACE_NAME_RE.test(name) || (enabled !== 'true' && enabled !== 'false')) return null
    if (Object.hasOwn(settings, name)) return null
    settings[name] = enabled === 'true'
  }
  return settings
}

export const serializeTriathlonTraceSettings = (settings: TriathlonTraceSettings): string =>
  Object.entries(settings)
    .map(([name, enabled]) => `${name}:${enabled}`)
    .join('&')

export const triathlonTraceEnabled = (
  settings: TriathlonTraceSettings | undefined,
  trace: string,
): boolean => settings?.[triathlonTraceName(trace)] !== false
