// The country the weather response already names for this browser, handed to the visitor
// register through a document event. It lives in memory only: no storage, no request, and it
// never enters the shared presence channel.
type CountryEvents = {
  'visitor-country:request': CustomEvent<void>
  'visitor-country:change': CustomEvent<string | null>
}

declare global {
  interface CustomEventMap extends CountryEvents {}
  interface DocumentEventMap extends CountryEvents {}
}

export function createVisitorCountrySource(target: Document, signal: AbortSignal) {
  let latest: string | null = null
  const dispatch = () =>
    target.dispatchEvent(new CustomEvent('visitor-country:change', { detail: latest }))
  target.addEventListener('visitor-country:request', dispatch, { signal })
  signal.addEventListener(
    'abort',
    () => {
      latest = null
      dispatch()
    },
    { once: true },
  )
  return {
    publish(country: string | null) {
      if (signal.aborted || country === latest) return
      latest = country
      dispatch()
    },
  }
}

export function subscribeVisitorCountry(
  target: Document,
  receive: (country: string | null) => void,
  signal: AbortSignal,
) {
  target.addEventListener(
    'visitor-country:change',
    event => {
      const detail = 'detail' in event ? event.detail : null
      receive(typeof detail === 'string' && /^[A-Za-z]{2}$/.test(detail) ? detail : null)
    },
    { signal },
  )
  target.dispatchEvent(new CustomEvent<void>('visitor-country:request'))
}
