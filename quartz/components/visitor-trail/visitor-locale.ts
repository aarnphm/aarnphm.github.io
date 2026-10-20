// What the visitor register knows about the viewer. The UTC offset is the one field shared with
// the room (camp-presence.ts publishes it); the zone and country stay in this browser and only
// label the viewer's own offset cell.
export type VisitorLocale = {
  zone: { label: string; iana: string; offset: string } | null
  country: { label: string; code: string } | null
}

export const OFFSET_PATTERN = /^UTC(?:[+−]\d{1,2}(?::\d{2})?)?$/

function offsetLabel(zone: string, now: number, locale: string | undefined) {
  const part = new Intl.DateTimeFormat(locale, { timeZone: zone, timeZoneName: 'longOffset' })
    .formatToParts(now)
    .find(item => item.type === 'timeZoneName')?.value
  const match = part?.match(/^GMT(?:([+\-−])(\d{1,2})(?::(\d{2}))?)?$/)
  if (!match) return null
  if (!match[1] || (Number(match[2]) === 0 && Number(match[3] ?? 0) === 0)) return 'UTC'
  const hours = String(Number(match[2]))
  return `UTC${match[1] === '+' ? '+' : '−'}${hours}${match[3] && match[3] !== '00' ? `:${match[3]}` : ''}`
}

// Prefers the country the weather response names (the visit's location), then the region of
// the browser language (a guess: it follows the language, not the place).
export function visitorLocale(
  view: Pick<Window, 'navigator'>,
  weatherCountry: string | null,
  now = Date.now(),
): VisitorLocale {
  const language = view.navigator.language || undefined
  let zone: VisitorLocale['zone'] = null
  try {
    const iana = new Intl.DateTimeFormat().resolvedOptions().timeZone
    const offset = iana ? offsetLabel(iana, now, language) : null
    if (iana && offset)
      zone = {
        iana,
        offset,
        label: (iana.split('/').pop() ?? iana).replaceAll('_', ' ').toLowerCase(),
      }
  } catch {
    zone = null
  }
  let country: VisitorLocale['country'] = null
  try {
    const code = (
      weatherCountry && /^[A-Za-z]{2}$/.test(weatherCountry)
        ? weatherCountry
        : language
          ? new Intl.Locale(language).maximize().region
          : undefined
    )?.toUpperCase()
    const label = code
      ? new Intl.DisplayNames(language ? [language] : undefined, { type: 'region' }).of(code)
      : undefined
    if (code && label && label !== code) country = { code, label: label.toLowerCase() }
  } catch {
    country = null
  }
  return { zone, country }
}

export function localUtcOffset(view: Pick<Window, 'navigator'>, now = Date.now()) {
  return visitorLocale(view, null, now).zone?.offset ?? null
}

// The own offset cell's tooltip: zone and country, read here and never sent.
export function visitorLocaleTitle(locale: VisitorLocale) {
  const parts = [locale.zone?.iana, locale.country?.label].filter(Boolean)
  return parts.length > 0 ? `${parts.join(' · ')}. only you see this` : ''
}
