import { almanac, type Site } from '../components/scripts/404-almanac'

// The homepage can preview its sky at another time of today. The preview is an offset from the
// real clock on the root element, so the trail frame reads it through its parent document. Sky
// and light read sceneNow; people, visits and the local clock keep the real time.
export type ScenePhase = 'sunrise' | 'midday' | 'sunset' | 'evening'
export type ScenePreview = { phase: ScenePhase; offset: number }

const phases: ScenePhase[] = ['sunrise', 'midday', 'sunset', 'evening']
const changeEvent = 'scene-clock:change'
export const sceneClockStepEvent = 'scene-clock:step'

type SceneClockEvents = {
  'scene-clock:change': CustomEvent<void>
  'scene-clock:step': CustomEvent<void>
}

declare global {
  interface CustomEventMap extends SceneClockEvents {}
  interface DocumentEventMap extends SceneClockEvents {}
}

const isPhase = (value: string | undefined): value is ScenePhase =>
  phases.includes(value as ScenePhase)

export function scenePreview(target: Document): ScenePreview | null {
  const { scenePreview: phase, scenePreviewOffset } = target.documentElement.dataset
  const offset = Number(scenePreviewOffset)
  return isPhase(phase) && Number.isFinite(offset) ? { phase, offset } : null
}

export const sceneNow = (target: Document, now = Date.now()) =>
  now + (scenePreview(target)?.offset ?? 0)

export function setScenePreview(target: Document, preview: ScenePreview | null) {
  const root = target.documentElement
  if (preview) {
    root.dataset.scenePreview = preview.phase
    root.dataset.scenePreviewOffset = String(Math.round(preview.offset))
  } else {
    if (!root.dataset.scenePreview) return
    delete root.dataset.scenePreview
    delete root.dataset.scenePreviewOffset
  }
  target.dispatchEvent(new CustomEvent<void>(changeEvent))
}

export function subscribeSceneClock(target: Document, receive: () => void, signal: AbortSignal) {
  target.addEventListener(changeEvent, receive, { signal })
}

// Phases are sun altitudes, since clock hours put a December sunrise at 60° N in the dark. The
// sun sits 2° up at sunrise and sunset so its disc clears the far ridge, and evening is 10° below
// the horizon, past the page's dark threshold of -4°.
export function scenePhaseTimes(site: Site, now = Date.now()): Record<ScenePhase, number | null> {
  const start = new Date(now)
  start.setHours(0, 0, 0, 0)
  const altitude = (at: number) => almanac(new Date(at), site).sun.alt
  const step = 10 * 60_000
  const samples = Array.from({ length: 24 * 6 + 1 }, (_, index) => {
    const at = start.getTime() + index * step
    return { at, altitude: altitude(at) }
  })
  const crossing = (level: number, rising: boolean) => {
    for (let index = 1; index < samples.length; index++) {
      const before = samples[index - 1]
      const after = samples[index]
      if (
        rising
          ? before.altitude >= level || after.altitude < level
          : before.altitude < level || after.altitude >= level
      )
        continue
      let low = before.at
      let high = after.at
      while (high - low > 60_000) {
        const middle = (low + high) / 2
        if (altitude(middle) < level === rising) low = middle
        else high = middle
      }
      return Math.round(high / 60_000) * 60_000
    }
    return null
  }
  const noon = samples.reduce((best, sample) => (sample.altitude > best.altitude ? sample : best))
  return {
    sunrise: crossing(2, true),
    midday: noon.at,
    sunset: crossing(2, false),
    evening: crossing(-10, false),
  }
}

// The next phase after the current preview with a time today, or null to return to now.
export function nextScenePreview(target: Document, site: Site, now = Date.now()) {
  const current = scenePreview(target)?.phase
  const times = scenePhaseTimes(site, now)
  for (const phase of phases.slice(current ? phases.indexOf(current) + 1 : 0)) {
    const at = times[phase]
    if (at !== null) return { phase, offset: at - now }
  }
  return null
}
