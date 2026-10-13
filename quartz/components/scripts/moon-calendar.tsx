import { render, type KeyboardEventHandler } from 'preact'
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks'
import { lunarDiscPath, lunarPhase, readLunarEvents, type LunarEvent } from '../../util/lunar-phase'
import { trailScene } from '../../util/trail-scene'
import { subscribeVisitorMoon } from '../../util/visitor-lunar'
import { subscribeVisitorTrailWeather } from '../../util/visitor-trail-weather'
import { englishDateLabels } from '../controls/date-picker'
import {
  addMonths,
  isoDateFromLocal,
  isoMonth,
  monthGridDates,
  parseIsoMonth,
  shiftIsoDate,
} from '../controls/dates'
import { MonthGrid } from '../controls/month-grid'
import { registerEscapeHandler } from './escape-handler'

const localNoon = (date: string) => new Date(`${date}T12:00:00`).getTime()
const dateLabel = (date: string) =>
  new Date(`${date}T12:00:00`)
    .toLocaleDateString('en', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
    .toLowerCase()

const MoonDisc = ({ phase }: { phase: number }) => (
  <svg viewBox="-8 -8 16 16" aria-hidden="true" focusable="false">
    <circle r="7" class="moon-calendar-outline" />
    <path d={lunarDiscPath(phase, 7, 0.25)} class="moon-calendar-lit" />
  </svg>
)

function MoonCalendar({ open, signal }: { open: boolean; signal: AbortSignal }) {
  const today = isoDateFromLocal(new Date())
  const [month, setMonth] = useState(today.slice(0, 7))
  const [selected, setSelected] = useState(today)
  const [ephemeris, setEphemeris] = useState<{
    month: string
    events: LunarEvent[]
    state: 'loading' | 'ready' | 'unavailable'
  }>({ month, events: [], state: 'loading' })
  const [retry, setRetry] = useState(0)
  const cache = useRef(new Map<string, LunarEvent[]>())
  const host = useRef<HTMLDivElement>(null)
  const focusDate = useRef<string | null>(null)
  const view = parseIsoMonth(month)
  const all = view ? monthGridDates(view) : []
  const dates = all[35]?.startsWith(month) ? all : all.slice(0, 35)
  const saved = cache.current.get(month)
  const events = ephemeris.month === month ? ephemeris.events : (saved ?? [])
  const state = ephemeris.month === month ? ephemeris.state : saved ? 'ready' : 'loading'

  useEffect(() => {
    if (!open || !dates.length) return
    const saved = cache.current.get(month)
    if (saved) {
      setEphemeris({ month, events: saved, state: 'ready' })
      return
    }
    const controller = new AbortController()
    const requestSignal = AbortSignal.any([signal, controller.signal, AbortSignal.timeout(6000)])
    setEphemeris({ month, events: [], state: 'loading' })
    const load = async () => {
      // Enclose the spillover days too, so every disc can interpolate between published quarters.
      const start = new Date(localNoon(dates[0]) - 12 * 86400000)
      const url = new URL('https://aa.usno.navy.mil/api/moon/phases/date')
      url.searchParams.set('date', start.toISOString().slice(0, 10))
      url.searchParams.set('nump', '12')
      const response = await fetch(url, {
        signal: requestSignal,
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
      }).catch(() => null)
      const received = response?.ok
        ? readLunarEvents(await response.json().catch(() => null))
        : null
      if (signal.aborted || controller.signal.aborted) return
      const end = received?.at(-1)
      if (
        received &&
        end &&
        received[0].at <= localNoon(dates[0]) &&
        end.at > localNoon(dates.at(-1)!)
      ) {
        if (cache.current.size >= 4) cache.current.delete(cache.current.keys().next().value ?? '')
        cache.current.set(month, received)
        setEphemeris({ month, events: received, state: 'ready' })
      } else setEphemeris({ month, events: [], state: 'unavailable' })
    }
    void load()
    return () => controller.abort()
  }, [open, month, retry])

  useLayoutEffect(() => {
    if (!focusDate.current) return
    host.current
      ?.querySelector<HTMLButtonElement>(`[data-moon-date="${focusDate.current}"]`)
      ?.focus()
    focusDate.current = null
  }, [month, selected])

  const select = (date: string, focus = false) => {
    if (focus) focusDate.current = date
    setSelected(date)
    setMonth(date.slice(0, 7))
  }
  const step = (delta: number) => {
    if (!view) return
    const next = isoMonth(addMonths(view, delta))
    setMonth(next)
    setSelected(`${next}-01`)
  }
  const onKeyDown: KeyboardEventHandler<HTMLDivElement> = event => {
    if (!(event.target instanceof HTMLButtonElement) || !event.target.dataset.moonDate) return
    const date = event.target.dataset.moonDate
    const weekday = new Date(`${date}T12:00:00`).getDay()
    const offsets: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowUp: -7,
      ArrowDown: 7,
      Home: -weekday,
      End: 6 - weekday,
    }
    let next: string | undefined
    if (event.key in offsets)
      next = shiftIsoDate(date, offsets[event.key], undefined, undefined)?.date
    else if ((event.key === 'PageUp' || event.key === 'PageDown') && view) {
      const target = addMonths(view, event.key === 'PageUp' ? -1 : 1)
      const last = new Date(target.year, target.month, 0).getDate()
      next = `${isoMonth(target)}-${String(Math.min(Number(date.slice(8)), last)).padStart(2, '0')}`
    }
    if (!next) return
    event.preventDefault()
    select(next, true)
  }

  // Primary phase dates show the exact event disc; other days show the local-noon estimate.
  const moonFor = (date: string) => {
    const primary = events.find(event => isoDateFromLocal(new Date(event.at)) === date)
    return { primary, moon: lunarPhase(primary?.at ?? localNoon(date), events) }
  }
  const { primary, moon } = moonFor(selected)
  return (
    <div ref={host} class="moon-calendar" data-state={state} data-month={month}>
      <div class="g-cal-head">
        <span class="g-cal-title" aria-live="polite">
          {englishDateLabels.monthYear(`${month}-01`)}
        </span>
        {[-1, 1].map(delta => (
          <button
            type="button"
            class="g-icon-button"
            aria-label={delta < 0 ? 'previous month' : 'next month'}
            onClick={() => step(delta)}
          >
            <svg class="g-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              <path d={delta < 0 ? 'M10 3.5 5.5 8l4.5 4.5' : 'M6 3.5 10.5 8 6 12.5'} />
            </svg>
          </button>
        ))}
      </div>
      <MonthGrid
        id="moon-calendar-grid"
        dates={dates}
        month={month}
        today={today}
        rowMinHeight="2.75rem"
        weekdayLabel={date =>
          new Date(`${date}T12:00:00`)
            .toLocaleDateString('en', { weekday: 'short' })
            .slice(0, 2)
            .toLowerCase()
        }
        dayLabel={dateLabel}
        dayAttributes={date => ({ 'data-selected': date === selected ? 'true' : undefined })}
        onKeyDown={onKeyDown}
        renderDay={date => {
          const { primary, moon } = moonFor(date)
          return (
            <button
              type="button"
              class="moon-calendar-day"
              data-moon-date={date}
              data-moon-event={primary?.name}
              aria-label={`${dateLabel(date)}, ${moon.name}${moon.source === 'estimate' ? ', estimated' : ''}`}
              aria-pressed={date === selected}
              tabIndex={date === selected ? 0 : -1}
              onClick={() => select(date)}
              title={
                primary
                  ? `${primary.name} · ${new Date(primary.at).toLocaleTimeString('en', { hour: 'numeric', minute: '2-digit' })}`
                  : moon.name
              }
            >
              <MoonDisc phase={moon.phase} />
            </button>
          )
        }}
      />
      <div class="moon-calendar-readout" role="status">
        <MoonDisc phase={moon.phase} />
        <div>
          <span>{moon.name}</span>
          <time dateTime={selected}>
            {new Date(localNoon(selected))
              .toLocaleDateString('en', { month: 'short', day: 'numeric' })
              .toLowerCase()}
            {' · '}
            {primary
              ? new Date(primary.at)
                  .toLocaleTimeString('en', { hour: 'numeric', minute: '2-digit' })
                  .toLowerCase()
              : `approx. ${Math.round(moon.illumination * 100)}% lit`}
          </time>
        </div>
        <button type="button" class="g-text-button" onClick={() => select(today)}>
          today
        </button>
      </div>
      <div class="moon-calendar-source">
        <span>
          {state === 'loading'
            ? 'loading phases'
            : state === 'unavailable'
              ? 'estimated phases'
              : 'local dates · USNO phases'}
        </span>
        {state === 'unavailable' && (
          <button type="button" class="g-text-button" onClick={() => setRetry(retry + 1)}>
            retry
          </button>
        )}
      </div>
    </div>
  )
}

export function mountMoonCalendar(root: HTMLElement, signal: AbortSignal) {
  const panel = root.querySelector<HTMLElement>('#homepage-moon-panel')
  const toggle = root.querySelector<HTMLButtonElement>('.homepage-moon-toggle')
  if (!panel || !toggle) return () => {}
  const document = root.ownerDocument
  let daylight: boolean | null = null
  const updateVisibility = () => {
    const hour = new Date().getHours()
    toggle.hidden = daylight ?? (hour >= 6 && hour < 18)
    if (toggle.hidden && panel.matches(':popover-open')) panel.hidePopover()
  }
  const label = (name: string) => {
    toggle.textContent = name
    toggle.setAttribute('aria-label', `Moon calendar · ${name}`)
  }
  label(lunarPhase(Date.now()).name)
  subscribeVisitorMoon(document, moon => label(moon.name), signal)
  subscribeVisitorTrailWeather(
    document,
    weather => {
      daylight = weather?.day ?? null
      updateVisibility()
    },
    signal,
  )
  updateVisibility()
  const timer = window.setInterval(updateVisibility, 60000)
  document.addEventListener('visibilitychange', updateVisibility, { signal })

  // The scene host stands beside body, outside the trail root.
  const host = document.querySelector<HTMLElement>('.homepage-trail-scene')
  const settingsOpen = () =>
    !host?.hidden &&
    Boolean(
      trailScene(document)?.querySelector(
        '.trail-tree-panel:not([hidden]), .trail-camp-tools:not([hidden])',
      ),
    )
  const syncSettings = () => {
    toggle.disabled = settingsOpen()
    if (toggle.disabled && panel.matches(':popover-open')) panel.hidePopover()
  }
  const settings = new MutationObserver(records => {
    if (
      records.some(
        record =>
          record.target instanceof HTMLElement &&
          record.target.matches('.trail-tree-panel, .trail-camp-tools, .homepage-trail-scene'),
      )
    )
      syncSettings()
  })
  if (host) settings.observe(host, { attributes: true, attributeFilter: ['hidden'] })
  const observeSettings = () => {
    const scene = trailScene(document)
    if (scene)
      settings.observe(scene, { subtree: true, attributes: true, attributeFilter: ['hidden'] })
  }
  observeSettings()
  host?.addEventListener('trail:mounted', observeSettings, { signal })
  panel.addEventListener(
    'beforetoggle',
    event => {
      if ('newState' in event && event.newState === 'open' && settingsOpen()) event.preventDefault()
    },
    { signal },
  )
  const update = () => {
    const open = panel.matches(':popover-open')
    toggle.setAttribute('aria-expanded', String(open))
    render(<MoonCalendar open={open} signal={signal} />, panel)
    if (open)
      panel
        .querySelector<HTMLButtonElement>('[data-moon-date][tabindex="0"]')
        ?.focus({ preventScroll: true })
  }
  panel.addEventListener('toggle', update, { signal })
  update()
  const disposeEscape = registerEscapeHandler(
    panel,
    () => {
      panel.hidePopover()
      toggle.focus({ preventScroll: true })
    },
    () => panel.matches(':popover-open'),
  )
  return () => {
    window.clearInterval(timer)
    settings.disconnect()
    disposeEscape()
    render(null, panel)
  }
}
