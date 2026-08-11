import type { ActivitySummary } from '../../../plugins/stores/analytics'
import type { ActivityKind } from '../../../plugins/stores/strava'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import { predDateFromLocal } from '../tools/date-picker'

export const SEARCH_SECTIONS: { label: string; chart: string; hay: string }[] = [
  {
    label: 'body weight',
    chart: 'body',
    hay: 'body weight kg lbs mass cut goal fat bmi ffmi fat-free mass muscle bone water composition scale index',
  },
  { label: 'form · ramp', chart: 'gauge', hay: 'form ramp gauge taper peak projection' },
  {
    label: 'recovery · hrv · rhr',
    chart: 'recovery',
    hay: 'recovery hrv heart rate variability rhr resting autonomic illness temperature overreaching suppressed fatigue oura',
  },
  {
    label: 'heat strain · acclimatisation',
    chart: 'heat',
    hay: 'heat strain hsi core body skin temperature ambient thermal hot weather weatherkit acclimatisation acclimation exposure proxy',
  },
  {
    label: 'sleep · debt',
    chart: 'sleep',
    hay: 'sleep debt duration score short streak need target rest hours oura',
  },
  {
    label: 'vo2max · fitness age',
    chart: 'vo2max',
    hay: 'vo2max vo2 max aerobic fitness age friend percentile engine ftp map trend',
  },
  {
    label: 'lactate threshold projection',
    chart: 'lactate',
    hay: 'lactate threshold projection lt lthr lt2 mlss heart rate bpm pace power proxy forecast',
  },
  {
    label: 'best efforts · power curve',
    chart: 'power',
    hay: 'best efforts power curve critical power cycling watts duration ftp six weeks year',
  },
  {
    label: 'ftp hypothesis',
    chart: 'ftp',
    hay: 'ftp watts power vo2 hypothesis slider acsm efficiency threshold lt2 vt2 cycling',
  },
  {
    label: 'abilities',
    chart: 'abilities',
    hay: 'abilities radar sprint threshold endurance climb stride length cadence recovery vertical oscillation power profile vam wkg swim bike run pace css stroke average',
  },
  {
    label: 'training distributions',
    chart: 'distributions',
    hay: 'heart rate zones distribution majority training time average power cadence skin temperature heat strain index hsi telemetry swim bike run date range',
  },
  {
    label: 'cardiovascular health',
    chart: 'cardio',
    hay: 'cardio cardiovascular heart rhr hrv efficiency factor decoupling aerobic drift',
  },
  {
    label: 'TSS · fitness · fatigue · form',
    chart: 'pmc',
    hay: 'pmc training stress score tss fitness fatigue form ctl atl tsb discipline swim bike run',
  },
  { label: 'weekly load', chart: 'weekly', hay: 'weekly load volume tss' },
  { label: 'relative effort', chart: 'effort', hay: 'relative effort suffer score weekly' },
  {
    label: 'race readiness',
    chart: 'readiness',
    hay: 'race readiness predicted time sprint olympic 70.3 ironman binding leg',
  },
  {
    label: 'pace trend + forecast',
    chart: 'trend',
    hay: 'pace trend forecast threshold faster slower ewma ols',
  },
  { label: 'things to improve', chart: 'actions', hay: 'actions things improve weakest' },
]

export const GLOSS_CHART: Record<string, string> = {
  tss: 'pmc',
  ctl: 'pmc',
  atl: 'pmc',
  tsb: 'gauge',
  acwr: 'gauge',
  ramp: 'gauge',
  monotony: 'gauge',
  strain: 'gauge',
  load: 'weekly',
  effort: 'effort',
  score: 'readiness',
  binding: 'readiness',
  predtime: 'readiness',
  conf: 'trend',
  threshold: 'trend',
  trend: 'trend',
  weight: 'body',
  wtrend: 'body',
  wgoal: 'body',
  bodyfat: 'body',
  ffmi: 'body',
  dexa: 'dexa',
  bmi: 'body',
  hrv: 'recovery',
  rhr: 'recovery',
  tempdev: 'recovery',
  ambienttemp: 'heat',
  heatstrain: 'heat',
  heatdose: 'heat',
  heatacclimation: 'heat',
  overreaching: 'recovery',
  oreadiness: 'recovery',
  sleepdebt: 'sleep',
  vo2max: 'vo2max',
  lactate: 'lactate',
  ftp: 'ftp',
  watts: 'power',
  fitage: 'vo2max',
  vam: 'abilities',
  radar: 'abilities',
  hrzones: 'distributions',
  activitytelemetry: 'distributions',
  ef: 'cardio',
  decouple: 'cardio',
}

export const searchCommandTitle = (prefix: string, value?: string): HTMLElement => {
  const wrap = el('span', 'tri-search-command')
  wrap.appendChild(el('span', 'tri-search-command-token', prefix))
  if (value) wrap.appendChild(el('span', 'tri-search-command-value', value))
  return wrap
}

export const activityResultItem = (title: HTMLElement | string, sub: string): HTMLElement => {
  const item = el('button', 'tri-ana-ritem')
  item.setAttribute('type', 'button')
  const label = el('span', 'tri-ana-ritem-t')
  if (typeof title === 'string') label.textContent = title
  else label.appendChild(title)
  item.append(label, el('span', 'tri-ana-ritem-s', sub))
  return item
}

export const matchesActivityTokens = (haystack: string, tokens: string[]): boolean =>
  tokens.every(token => haystack.includes(token))

export const activityResultItems = (results: HTMLElement | null): HTMLElement[] =>
  results ? Array.from(results.querySelectorAll<HTMLElement>('.tri-ana-ritem')) : []

export const setActivityResultSelection = (results: HTMLElement | null, index: number): number => {
  const items = activityResultItems(results)
  if (items.length === 0) return -1
  const selected = ((index % items.length) + items.length) % items.length
  items.forEach((item, itemIndex) =>
    item.classList.toggle('tri-ana-ritem--sel', itemIndex === selected),
  )
  items[selected].scrollIntoView({ block: 'nearest' })
  return selected
}

export const ACTIVITY_FILTER_SPORTS: readonly ActivityKind[] = [
  'bike',
  'run',
  'swim',
  'walk',
  'strength',
  'yoga',
]

export const ACTIVITY_FILTER_ALIASES: Readonly<Record<string, ActivityKind>> = {
  hike: 'walk',
  weight: 'strength',
  gym: 'strength',
  pilates: 'yoga',
}

export const ACTIVITY_SORT_KEYS: readonly string[] = ['distance', 'cadence', 'pace']

export const DATE_FILTER_KEYWORDS: readonly string[] = ['today', 'yesterday', 'week', 'month']

export const DATE_FILTER_UNIT = /^(days?|d|weeks?|w|months?|mo)$/

export const DATE_HINT_SUBS: Record<string, string> = {
  today: 'completed today',
  yesterday: 'completed yesterday',
  week: 'last 7 days',
  month: 'last month',
}

export interface DateSpan {
  start: string
  end: string
}

export interface ActivityQuery {
  filterSport: string | null
  filterDate: DateSpan | null
  sortKey: string | null
  tokens: string[]
}

export const daySpan = (days: number): DateSpan => {
  const now = new Date()
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - (days - 1))
  return { start: predDateFromLocal(start), end: predDateFromLocal(now) }
}

export const monthSpan = (months: number): DateSpan => {
  const now = new Date()
  const start = new Date(now.getFullYear(), now.getMonth() - months, now.getDate())
  return { start: predDateFromLocal(start), end: predDateFromLocal(now) }
}

export const dateFilterSpan = (value: string): DateSpan | null => {
  if (value === 'today') return daySpan(1)
  if (value === 'yesterday') {
    const { start } = daySpan(2)
    return { start, end: start }
  }
  if (value === 'week') return daySpan(7)
  if (value === 'month') return monthSpan(1)
  const m = /^(\d+)\s*(days?|d|weeks?|w|months?|mo)?$/.exec(value)
  if (!m) return null
  const n = Number(m[1])
  if (n < 1) return null
  const unit = m[2] ?? 'd'
  if (unit.startsWith('mo')) return monthSpan(n)
  return daySpan(unit.startsWith('w') ? n * 7 : n)
}

export const resolveActivityFilterSport = (value: string): ActivityKind | null =>
  ACTIVITY_FILTER_ALIASES[value] ?? ACTIVITY_FILTER_SPORTS.find(sport => sport === value) ?? null

export const aliasesForActivityFilterSport = (sport: ActivityKind): string[] =>
  Object.entries(ACTIVITY_FILTER_ALIASES).flatMap(([alias, canonical]) =>
    canonical === sport ? [alias] : [],
  )

export const isActivityFilterSport = (value: string, sports: readonly ActivityKind[]): boolean => {
  const sport = resolveActivityFilterSport(value)
  return sport !== null && sports.includes(sport)
}

export const parseActivityQuery = (rawTokens: string[]): ActivityQuery => {
  let filterSport: string | null = null
  let filterDate: DateSpan | null = null
  let sortKey: string | null = null
  const tokens: string[] = []
  for (let i = 0; i < rawTokens.length; i++) {
    const t = rawTokens[i]
    if (t.startsWith('filter:')) {
      let fv = t.slice(7)
      if (/^\d+$/.test(fv) && rawTokens[i + 1] && DATE_FILTER_UNIT.test(rawTokens[i + 1])) {
        fv = `${fv} ${rawTokens[i + 1]}`
        i++
      }
      const span = dateFilterSpan(fv)
      if (span) filterDate = span
      else filterSport = resolveActivityFilterSport(fv) ?? fv
    } else if (t.startsWith('sort:')) {
      sortKey = t.slice(5)
    } else if (t) tokens.push(t)
  }
  return { filterSport, filterDate, sortKey, tokens }
}

export const sortActivitiesBy = <
  T extends Pick<ActivitySummary, 'date' | 'distanceKm' | 'cadence' | 'movingTimeS'>,
>(
  acts: T[],
  sortKey: string | null,
): T[] => {
  if (!sortKey) return acts
  return acts.sort((a, b) => {
    if (sortKey === 'date') return b.date.localeCompare(a.date)
    if (sortKey === 'distance') return b.distanceKm - a.distanceKm
    if (sortKey === 'cadence') return (b.cadence ?? 0) - (a.cadence ?? 0)
    if (sortKey === 'pace') {
      const sa = a.movingTimeS > 0 ? a.distanceKm / a.movingTimeS : 0
      const sb = b.movingTimeS > 0 ? b.distanceKm / b.movingTimeS : 0
      return sb - sa
    }
    return 0
  })
}

export const activityCommandHints = (
  lastToken: string,
  noun: string,
  filterSports: readonly ActivityKind[] = ACTIVITY_FILTER_SPORTS,
): HTMLElement[] => {
  const hints: HTMLElement[] = []
  const filterValue = lastToken.startsWith('filter:') ? lastToken.slice(7) : null
  const sortValue = lastToken.startsWith('sort:') ? lastToken.slice(5) : null
  if (filterValue !== null && /^\d+$/.test(filterValue)) {
    const units = filterValue === '1' ? ['day', 'week', 'month'] : ['days', 'weeks', 'months']
    for (const u of units) {
      const it = activityResultItem(
        searchCommandTitle('filter:', `${filterValue} ${u}`),
        `last ${filterValue} ${u}`,
      )
      it.dataset.insert = `filter:${filterValue} ${u}`
      hints.push(it)
    }
  } else if (
    filterValue !== null &&
    !isActivityFilterSport(filterValue, filterSports) &&
    !dateFilterSpan(filterValue)
  ) {
    for (const sport of filterSports) {
      const alias = aliasesForActivityFilterSport(sport).find(value =>
        value.startsWith(filterValue),
      )
      const value = sport.startsWith(filterValue) ? sport : alias
      if (value) {
        const sub = value === sport ? `filter ${noun}` : `filter ${noun} (${sport})`
        const it = activityResultItem(searchCommandTitle('filter:', value), sub)
        it.dataset.insert = `filter:${value}`
        hints.push(it)
      }
    }
    for (const k of DATE_FILTER_KEYWORDS)
      if (k.startsWith(filterValue)) {
        const it = activityResultItem(searchCommandTitle('filter:', k), DATE_HINT_SUBS[k])
        it.dataset.insert = `filter:${k}`
        hints.push(it)
      }
  } else if (sortValue !== null && !ACTIVITY_SORT_KEYS.includes(sortValue)) {
    for (const s of ACTIVITY_SORT_KEYS)
      if (s.startsWith(sortValue)) {
        const it = activityResultItem(searchCommandTitle('sort:', s), `sort ${noun}`)
        it.dataset.insert = `sort:${s}`
        hints.push(it)
      }
  } else if (lastToken.length > 0 && 'filter:'.startsWith(lastToken) && lastToken !== 'filter:') {
    const it = activityResultItem(
      searchCommandTitle('filter:'),
      'filter by sport or date (bike, strength, yoga, today, 3 days)',
    )
    it.dataset.insert = 'filter:'
    hints.push(it)
  } else if (lastToken.length > 0 && 'sort:'.startsWith(lastToken) && lastToken !== 'sort:') {
    const it = activityResultItem(searchCommandTitle('sort:'), 'sort by distance, cadence, pace')
    it.dataset.insert = 'sort:'
    hints.push(it)
  }
  return hints
}

export const marqueeCtl = (): { run: (name: HTMLElement) => void; stop: () => void } => {
  let host: HTMLElement | null = null
  let raf = 0
  let start = 0
  const stop = () => {
    if (raf) {
      cancelAnimationFrame(raf)
      raf = 0
    }
    if (host) {
      host.scrollLeft = 0
      host.style.textOverflow = ''
      host = null
    }
    start = 0
  }
  const run = (name: HTMLElement) => {
    if (name === host) return
    stop()
    const max = name.scrollWidth - name.clientWidth
    if (max <= 2) return
    host = name
    name.style.textOverflow = 'clip'
    const leg = (max / 36) * 1000
    const cycle = leg * 2 + 1400
    const step = (ts: number) => {
      if (!name.isConnected) {
        stop()
        return
      }
      if (!start) start = ts
      const t = (ts - start) % cycle
      name.scrollLeft =
        t < leg
          ? (t / leg) * max
          : t < leg + 700
            ? max
            : t < leg * 2 + 700
              ? max - ((t - leg - 700) / leg) * max
              : 0
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
  }
  return { run, stop }
}

export const detailHead = (
  date: string,
  title?: string,
  backLabel = 'go back',
): { head: HTMLElement; back: HTMLElement; actions: HTMLElement } => {
  const head = el('div', 'tri-pop-head tri-pop-head--detail')
  const row = el('div', 'tri-pop-head-row')
  const actions = el('div', 'tri-pop-head-actions')
  const back = el('button', 'tri-ana-back tri-ana-back--ico')
  back.setAttribute('type', 'button')
  back.setAttribute('aria-label', backLabel)
  const ico = svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' })
  ico.appendChild(svg('path', { d: 'M19 12H5M11 6l-6 6 6 6' }))
  back.appendChild(ico)
  actions.appendChild(back)
  row.append(el('span', 'tri-pop-date', date), actions)
  head.appendChild(row)
  if (!title) return { head, back, actions }
  const titleEl = el('span', 'tri-pop-title', title)
  const marquee = marqueeCtl()
  titleEl.addEventListener('mouseenter', () => marquee.run(titleEl))
  titleEl.addEventListener('mouseleave', marquee.stop)
  head.appendChild(titleEl)
  return { head, back, actions }
}
