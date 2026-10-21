import type { TriathlonCalendar } from './details'

const icalText = (value: string): string =>
  value
    .replaceAll('\\', '\\\\')
    .replace(/\r\n|\r|\n/g, '\\n')
    .replaceAll(';', '\\;')
    .replaceAll(',', '\\,')

const foldLine = (line: string): string => {
  const encoder = new TextEncoder()
  let folded = ''
  let bytes = 0
  for (const character of line) {
    const size = encoder.encode(character).length
    if (bytes + size > 75) {
      folded += '\r\n '
      bytes = 1
    }
    folded += character
    bytes += size
  }
  return folded
}

const icalDate = (date: string): string => date.replaceAll('-', '')

const exclusiveEndDate = (date: string): string => {
  const end = new Date(`${date}T00:00:00.000Z`)
  end.setUTCDate(end.getUTCDate() + 1)
  return icalDate(end.toISOString().slice(0, 10))
}

export const serializeTriathlonCalendar = (calendar: TriathlonCalendar | null): string => {
  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//aarnphm.xyz//Race calendar//EN',
    'CALSCALE:GREGORIAN',
    `X-WR-CALNAME:${calendar ? `${calendar.year} race calendar` : 'Race calendar'}`,
  ]
  for (const event of calendar?.events ?? []) {
    if (event.date === null) continue
    const description = [
      event.format,
      event.participated ? 'Participated' : null,
      event.note,
      `Official event: ${event.url}`,
      calendar?.checked ? `Dates checked: ${calendar.checked}` : null,
    ]
      .filter(Boolean)
      .join('\n')
    lines.push(
      'BEGIN:VEVENT',
      `UID:${event.id}@aarnphm.xyz`,
      `DTSTAMP:${icalDate(calendar?.checked ?? event.date)}T000000Z`,
      `DTSTART;VALUE=DATE:${icalDate(event.date)}`,
      `DTEND;VALUE=DATE:${exclusiveEndDate(event.endDate ?? event.date)}`,
      `SUMMARY:${icalText(event.name)}`,
      `LOCATION:${icalText(event.location)}`,
      `DESCRIPTION:${icalText(description)}`,
      `URL:${event.url}`,
      'END:VEVENT',
    )
  }
  lines.push('END:VCALENDAR')
  return `${lines.map(foldLine).join('\r\n')}\r\n`
}
