export type CuriusDateFilter =
  | { kind: 'all' }
  | { kind: 'invalid' }
  | { kind: 'day'; key: string; label: string }

export function curiusDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

/** A day follows the reader's timezone, including daylight-saving boundaries. */
export function curiusDateFilter(anchor: string, now = new Date()): CuriusDateFilter {
  const value = anchor.replace(/^#/, '').toLowerCase()
  if (!value) return { kind: 'all' }
  let date = now
  if (value !== 'today') {
    const match = /^(\d{1,2})-(\d{1,2})-(\d{4})$/.exec(value)
    if (!match) return { kind: 'invalid' }
    const [, month, day, year] = match.map(Number)
    date = new Date(0)
    date.setFullYear(year, month - 1, day)
    date.setHours(0, 0, 0, 0)
    if (
      year < 1 ||
      date.getFullYear() !== year ||
      date.getMonth() !== month - 1 ||
      date.getDate() !== day
    )
      return { kind: 'invalid' }
  }
  const key = curiusDateKey(date)
  return { kind: 'day', key, label: value === 'today' ? `today · ${key}` : key }
}
