import { hostnameMatches } from './url'

export type ArenaExternalEmbedMode = 'auto' | 'iframe' | 'fetch' | 'none'

export function readArenaExternalEmbedMode(value: unknown): ArenaExternalEmbedMode | undefined {
  if (typeof value !== 'string') return undefined

  const normalized = value.trim().toLowerCase()
  if (normalized === 'auto') return 'auto'
  if (normalized === 'iframe' || normalized === 'frame') return 'iframe'
  if (normalized === 'fetch' || normalized === 'snapshot' || normalized === 'proxy') return 'fetch'
  if (
    normalized === 'none' ||
    normalized === 'off' ||
    normalized === 'false' ||
    normalized === 'disabled'
  ) {
    return 'none'
  }

  return undefined
}

export function defaultArenaExternalEmbedMode(
  rawUrl: string | undefined,
  markerDisabled: boolean,
): ArenaExternalEmbedMode {
  if (markerDisabled) return 'none'
  if (!rawUrl) return 'auto'

  try {
    const url = new URL(rawUrl)
    if (hostnameMatches(url, 'github.com')) return 'none'
  } catch {
    return 'auto'
  }

  return 'auto'
}

export function arenaEmbedHtmlPath(rawUrl: string): string {
  return `/api/arena-embed/html?url=${encodeURIComponent(rawUrl)}`
}

export function arenaEmbedCapabilityPath(rawUrl: string): string {
  return `/api/arena-embed/capability?url=${encodeURIComponent(rawUrl)}`
}
