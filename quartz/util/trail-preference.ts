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
