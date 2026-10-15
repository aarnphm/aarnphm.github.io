import type { TrailSeason } from './visitor-trail-weather'

export const trailSeasons: readonly TrailSeason[] = ['spring', 'summer', 'autumn', 'winter']

const key = 'trail-season'

type SeasonEvents = { 'trail:season': CustomEvent<TrailSeason | null> }

declare global {
  interface CustomEventMap extends SeasonEvents {}
  interface DocumentEventMap extends SeasonEvents {}
}

export function isTrailSeason(value: unknown): value is TrailSeason {
  return trailSeasons.includes(value as TrailSeason)
}

// The visitor's chosen season, or null for the forecast's. Kept per browser.
export function readTrailSeason(view: Window | null | undefined): TrailSeason | null {
  try {
    const value = view?.localStorage.getItem(key)
    return isTrailSeason(value) ? value : null
  } catch {
    return null
  }
}

export function setTrailSeason(document: Document, season: TrailSeason | null) {
  try {
    const storage = document.defaultView?.localStorage
    if (season) storage?.setItem(key, season)
    else storage?.removeItem(key)
  } catch {
    // Private windows and blocked storage keep the choice for the page only.
  }
  document.dispatchEvent(new CustomEvent('trail:season', { detail: season }))
}

export function subscribeTrailSeason(
  document: Document,
  listener: (season: TrailSeason | null) => void,
  signal: AbortSignal,
) {
  document.addEventListener('trail:season', event => listener(event.detail), { signal })
}
