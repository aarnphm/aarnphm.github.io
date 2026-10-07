import type { MountedTriathlon } from '../triathlon/runtime/mount'
import { TRI_ANALYTICS_BOOT_CLASS } from '../triathlon/analytics/boot'
import { currentNavSignal } from './nav-lifecycle'

type TriathlonRuntime = typeof import('../triathlon/runtime/mount')

// Every page loads this chunk, so the runtime (~1 MB) ships as the LazyScripts bundle
// `triathlon.js` (quartz.config.ts), fetched only by triathlon content.
const TRIATHLON_CONTENT =
  '.triathlon, .tri-day-embed, .tri-compare-embed, .tri-calc-embed, [data-calendar-set]'

let runtime: Promise<TriathlonRuntime> | undefined
const loadRuntime = () =>
  (runtime ??= (
    import(new URL('triathlon.js', import.meta.url).href) as Promise<TriathlonRuntime>
  ).catch((error: unknown) => {
    runtime = undefined
    throw error
  }))

// Module scripts run after parsing, so a hard load starts the fetch before the first `nav`.
if (document.querySelector(TRIATHLON_CONTENT)) loadRuntime().catch(() => {})

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  document.documentElement.classList.toggle(
    TRI_ANALYTICS_BOOT_CLASS,
    root?.dataset.triView === 'analytics',
  )
  const signal = currentNavSignal()
  let mounted: Promise<MountedTriathlon | null> | undefined
  const mount = () =>
    (mounted ??= loadRuntime().then(({ mountTriathlon }) =>
      signal.aborted ? null : mountTriathlon(signal),
    ))
  const onDecrypted = ({ detail }: CustomEventMap['contentdecrypted']) => {
    if (detail.content.querySelector(TRIATHLON_CONTENT)) mount().catch(console.error)
  }
  document.addEventListener('contentdecrypted', onDecrypted)
  window.addCleanup(() => document.removeEventListener('contentdecrypted', onDecrypted))
  if (document.querySelector(TRIATHLON_CONTENT)) mount().catch(console.error)
})
