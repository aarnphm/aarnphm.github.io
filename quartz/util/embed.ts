const MEDIA_EXT = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.bmp',
  '.svg',
  '.webp',
  '.avif',
  '.ico',
  '.mp4',
  '.webm',
  '.ogv',
  '.mov',
  '.mkv',
  '.mp3',
  '.wav',
  '.ogg',
  '.m4a',
  '.flac',
  '.aac',
  '.pdf',
])

const PROVIDERS: { match: (host: string) => boolean; name: string }[] = [
  {
    match: host =>
      host === 'strava-embeds.com' || host === 'strava.com' || host.endsWith('.strava.com'),
    name: 'strava',
  },
]

export interface ExternalEmbed {
  src: string
  provider?: string
}

// Strava's "latest rides" widget: /athletes/<id>/latest-rides/<hash>. The hash only keys the
// widget; the athlete id selects which synced cache can stand in for it.
export const stravaLatestRidesAthlete = (rawUrl: string): number | undefined => {
  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    return undefined
  }
  const host = url.hostname.toLowerCase().replace(/^www\./, '')
  if (host !== 'strava.com') return undefined
  const match = /^\/athletes\/(\d+)\/latest-rides(?:\/|$)/.exec(url.pathname)
  return match ? Number(match[1]) : undefined
}

export const buildExternalEmbed = (rawUrl: string): ExternalEmbed | undefined => {
  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    return undefined
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return undefined
  const pathname = url.pathname.toLowerCase()
  const dot = pathname.lastIndexOf('.')
  if (dot >= 0 && MEDIA_EXT.has(pathname.slice(dot))) return undefined
  const host = url.hostname.toLowerCase().replace(/^www\./, '')
  return { src: rawUrl, provider: PROVIDERS.find(p => p.match(host))?.name }
}
