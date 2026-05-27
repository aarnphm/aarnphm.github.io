import { hostnameMatches } from './url'

export type ArenaExternalEmbedMode = 'auto' | 'iframe' | 'fetch' | 'capture' | 'none'

export interface ArenaEmbedCaptureOptions {
  width?: number
  height?: number
  dpr?: number
}

export function readArenaExternalEmbedMode(value: unknown): ArenaExternalEmbedMode | undefined {
  if (typeof value !== 'string') return undefined

  const normalized = value.trim().toLowerCase()
  if (normalized === 'auto') return 'auto'
  if (normalized === 'iframe' || normalized === 'frame') return 'iframe'
  if (normalized === 'fetch' || normalized === 'proxy' || normalized === 'html') return 'fetch'
  if (normalized === 'capture' || normalized === 'screenshot' || normalized === 'snapshot') {
    return 'capture'
  }
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

function pushCaptureParam(parts: string[], key: string, value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) return
  parts.push(`${key}=${encodeURIComponent(String(value))}`)
}

export function arenaEmbedCapturePath(rawUrl: string, opts: ArenaEmbedCaptureOptions = {}): string {
  const parts = [`url=${encodeURIComponent(rawUrl)}`]
  pushCaptureParam(parts, 'w', opts.width)
  pushCaptureParam(parts, 'h', opts.height)
  pushCaptureParam(parts, 'dpr', opts.dpr)
  return `/api/arena-embed/capture?${parts.join('&')}`
}
