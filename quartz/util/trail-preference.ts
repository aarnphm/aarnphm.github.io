const TRAIL_PREFERENCE_KEY = 'garden:trail-open'

export function readTrailPreference(): boolean {
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

// A saved-open trail restores its layout so its generated trees can survive a reload.
export function trailSceneSeed(): number {
  const key = 'garden:trail-seed'
  const seed = Math.floor(Math.random() * 4294967296)
  try {
    const stored = localStorage.getItem(key)
    const previous = stored === null ? Number.NaN : Number(stored)
    if (
      readTrailPreference() &&
      Number.isInteger(previous) &&
      previous >= 0 &&
      previous <= 0xffffffff
    )
      return previous
    localStorage.setItem(key, String(seed))
  } catch {
    // A blocked store still gives the current page a usable scene.
  }
  return seed
}
