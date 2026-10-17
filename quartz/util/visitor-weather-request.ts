type VisitorWeatherMemo = { at: number; promise: Promise<unknown> }

declare global {
  interface Window {
    __visitorWeather?: VisitorWeatherMemo
  }
}

export const VISITOR_WEATHER_TTL_MS = 15 * 60_000

export function fetchVisitorWeather(options?: {
  signal?: AbortSignal
  force?: boolean
}): Promise<unknown> {
  const signal = options?.signal
  if (signal?.aborted) return Promise.resolve(null)
  const now = Date.now()
  const existing = typeof window === 'object' ? window.__visitorWeather : undefined
  const fresh =
    !options?.force &&
    existing !== undefined &&
    now - existing.at >= 0 &&
    now - existing.at < VISITOR_WEATHER_TTL_MS
  let shared: Promise<unknown>
  if (fresh) {
    shared = existing.promise
  } else {
    const promise: Promise<unknown> = fetch('/api/weather', { cache: 'no-store' })
      .then(response => (response.ok ? response.json() : null))
      .catch(() => null)
      .then(body => {
        if (
          body === null &&
          typeof window === 'object' &&
          window.__visitorWeather?.promise === promise
        ) {
          delete window.__visitorWeather
        }
        return body
      })
    if (typeof window === 'object') {
      window.__visitorWeather = { at: now, promise }
    }
    shared = promise
  }
  if (!signal) return shared
  return new Promise(resolve => {
    const onAbort = () => resolve(null)
    signal.addEventListener('abort', onAbort, { once: true })
    void shared.then(value => {
      signal.removeEventListener('abort', onAbort)
      resolve(signal.aborted ? null : value)
    })
  })
}
