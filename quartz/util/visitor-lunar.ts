import { lunarPhase, lunarPhaseName, type LunarPhase } from './lunar-phase'
import { isRecord } from './type-guards'

export type VisitorMoon = LunarPhase & { at: number }

type LunarEvents = {
  'visitor-moon:request': CustomEvent<void>
  'visitor-moon:change': CustomEvent<VisitorMoon>
}
declare global {
  interface CustomEventMap extends LunarEvents {}
  interface DocumentEventMap extends LunarEvents {}
}

export function createVisitorMoonSource(document: Document, signal: AbortSignal) {
  let latest: VisitorMoon = { ...lunarPhase(Date.now()), at: Date.now() }
  let signature = ''
  const dispatch = () =>
    document.dispatchEvent(new CustomEvent('visitor-moon:change', { detail: latest }))
  document.addEventListener('visitor-moon:request', dispatch, { signal })
  return {
    publish(moon: VisitorMoon) {
      if (signal.aborted) return
      const value = JSON.stringify(moon)
      if (signature === value) return
      signature = value
      latest = moon
      dispatch()
    },
  }
}

export function subscribeVisitorMoon(
  document: Document,
  receive: (moon: VisitorMoon) => void,
  signal: AbortSignal,
) {
  document.addEventListener(
    'visitor-moon:change',
    event => {
      if (!('detail' in event)) return
      const value: unknown = event.detail
      if (
        !isRecord(value) ||
        typeof value.at !== 'number' ||
        !Number.isFinite(value.at) ||
        typeof value.phase !== 'number' ||
        !Number.isFinite(value.phase) ||
        value.phase < 0 ||
        value.phase >= 1 ||
        (value.source !== 'usno' && value.source !== 'estimate')
      )
        return
      receive({
        phase: value.phase,
        illumination: (1 - Math.cos(value.phase * Math.PI * 2)) / 2,
        waxing: value.phase < 0.5,
        name: lunarPhaseName(value.phase),
        source: value.source,
        at: value.at,
      })
    },
    { signal },
  )
  document.dispatchEvent(new CustomEvent<void>('visitor-moon:request'))
}
