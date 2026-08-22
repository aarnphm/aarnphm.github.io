export function parseClockSeconds(value: string): number {
  const trimmed = value.trim()
  if (!trimmed) return 0
  const rawParts = trimmed.split(':')
  if (rawParts.length === 2 || rawParts.length === 3) {
    const parts = rawParts.map(part => Number(part))
    if (parts.some(part => !Number.isFinite(part))) return 0
    if (parts.length === 3) return (parts[0] || 0) * 3600 + (parts[1] || 0) * 60 + (parts[2] || 0)
    return (parts[0] || 0) * 60 + (parts[1] || 0)
  }
  const seconds = Number(trimmed)
  return Number.isFinite(seconds) ? seconds : 0
}
