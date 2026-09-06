export const AUDIO_ICON_PATHS = {
  play: 'M8 5v14l11-7z',
  pause: 'M7 5h3.4v14H7zm6.6 0H17v14h-3.4z',
  repeat: 'M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z',
} as const

export const AUDIO_BAR_COUNT = 200

export function audioIconSvg(path: string): string {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`
}

export function audioBarHeight(i: number): number {
  return 30 + Math.round(60 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.55)))
}

export function memoPeaksUrl(url: string): string | undefined {
  const parsed = new URL(url)
  if (!/^\/triathlon\/memos\/[^/]+\.m4a$/.test(parsed.pathname)) return undefined
  parsed.pathname = parsed.pathname.replace(/\.m4a$/, '.peaks.json')
  return parsed.toString()
}

export function resampleAudioPeaks(value: unknown, count: number): number[] | null {
  if (typeof value !== 'object' || value === null || !('peaks' in value)) return null
  const peaks = value.peaks
  if (
    !Array.isArray(peaks) ||
    !peaks.length ||
    !Number.isInteger(count) ||
    count < 1 ||
    !peaks.every(
      (peak): peak is number =>
        typeof peak === 'number' && Number.isFinite(peak) && peak >= 0 && peak <= 1,
    )
  )
    return null
  return Array.from({ length: count }, (_, index) => {
    const start = Math.floor((index * peaks.length) / count)
    const end = Math.max(start + 1, Math.floor(((index + 1) * peaks.length) / count))
    return Math.max(...peaks.slice(start, end))
  })
}
