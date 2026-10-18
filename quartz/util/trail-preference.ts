const TRAIL_PREFERENCE_KEY = 'garden:trail-open'

// The scene relies on Chromium-only rendering and scheduling. User-Agent Client Hints exist only in
// Chromium browsers, and their brand list names Chromium even where the UA string is spoofed.
export function trailSupported(): boolean {
  const hints = (navigator as Navigator & { userAgentData?: { brands?: { brand: string }[] } })
    .userAgentData
  return hints?.brands?.some(({ brand }) => brand === 'Chromium') ?? false
}

export function readTrailPreference(): boolean {
  if (!trailSupported()) return false
  try {
    return localStorage.getItem(TRAIL_PREFERENCE_KEY) === 'true'
  } catch {
    return false
  }
}

export function writeTrailPreference(open: boolean) {
  try {
    localStorage.setItem(TRAIL_PREFERENCE_KEY, String(open))
  } catch {
    // A blocked preference store must leave the current-page control usable.
  }
}

// Keep the forest identity separate from visibility so a closed trail can reuse its trees.
export function trailSceneSeed(): number {
  const key = 'garden:trail-seed'
  const seed = Math.floor(Math.random() * 4294967296)
  try {
    const stored = localStorage.getItem(key)
    const previous = stored === null ? Number.NaN : Number(stored)
    if (Number.isInteger(previous) && previous >= 0 && previous <= 0xffffffff) return previous
    localStorage.setItem(key, String(seed))
  } catch {
    // A blocked store still gives the current page a usable scene.
  }
  return seed
}
