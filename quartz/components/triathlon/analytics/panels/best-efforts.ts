import type { Analytics } from '../../../../plugins/stores/analytics'
import type {
  BestEffortCategory,
  BestEffortEntry,
  BestEffortGroup,
  BestEffortSport,
} from '../../../../util/best-efforts'
import type { TriathlonContext } from '../../runtime/context'
import { shiftIsoDay } from '../../../../util/local-date'
import { powerCurveActivityLinkAttributes } from '../../../../util/triathlon-power-activity'
import { isRecord } from '../../../../util/type-guards'
import { setupPowerCurveActivityLinks } from '../../activity/power-links'
import { buildIcon } from '../../activity/primitives'
import { ELEV_RAMP, HEAT_RAMP, SPD_RAMP, STRIDE_RAMP } from '../../maps/palette'
import { el } from '../../runtime/dom'
import { nextMapMetricShortcutIndex } from '../../shell/command-palette'
import { anaTitle, markGlossDefinition } from '../shared'
import { buildBestEffortsChart, type BestEffortsChartView } from './best-efforts-chart'
import {
  bestEffortCategoryLabel,
  bestEffortDisplay,
  bestEffortShortLabel,
  buildBestEffortRank,
} from './best-efforts-format'

export const TRI_BEST_EFFORTS_KEY = 'tri-best-efforts'

type BestEffortsView = 'analysis' | 'top'
type BestEffortsSort = 'newest' | 'oldest' | 'high' | 'low'
type BestEffortsRange = 'all' | '7d' | '14d' | '30d' | '60d' | 'month'

interface BestEffortsState {
  sport: BestEffortSport
  view: BestEffortsView
  runCategory: string
  bikeGroup: BestEffortGroup
  bikeCategory: Partial<Record<BestEffortGroup, string>>
  year: number | null
  sort: BestEffortsSort
  rideRange: BestEffortsRange
  rideMonth: string
}

interface Picker {
  name: 'year' | 'sort' | 'month'
  wrap: HTMLElement
  trigger: HTMLButtonElement
  menu: HTMLElement
  options: HTMLButtonElement[]
}

interface CategoryTabOption {
  value: string
  label: string
  short: string
}

interface CategoryTablist {
  name: string
  element: HTMLElement
  tabs: HTMLButtonElement[]
  values: string[]
}

const SPORTS: readonly BestEffortSport[] = ['run', 'bike']
const VIEWS: readonly { key: BestEffortsView; label: string }[] = [
  { key: 'analysis', label: 'analysis' },
  { key: 'top', label: 'top 10' },
]
const RANGES: readonly { key: BestEffortsRange; label: string; days: number | null }[] = [
  { key: 'all', label: 'all-time', days: null },
  { key: '7d', label: '7d', days: 7 },
  { key: '14d', label: '14d', days: 14 },
  { key: '30d', label: '30d', days: 30 },
  { key: '60d', label: '60d', days: 60 },
  { key: 'month', label: 'month', days: null },
]
// High and low order by the category's own value (time, distance, metres, watts).
const SORTS: readonly { key: BestEffortsSort; label: string }[] = [
  { key: 'newest', label: 'newest first' },
  { key: 'oldest', label: 'oldest first' },
  { key: 'high', label: 'high → low' },
  { key: 'low', label: 'low → high' },
]
const BIKE_GROUPS: readonly BestEffortGroup[] = ['longest', 'elevation', 'distance', 'power']
const BIKE_GROUP_LABEL: Record<BestEffortGroup, string> = {
  longest: 'longest ride',
  elevation: 'elevation gain',
  distance: 'distance',
  power: 'power',
}
// W for power, as on the activity map tabs.
const BIKE_GROUP_SHORT: Record<BestEffortGroup, string> = {
  longest: 'L',
  elevation: 'E',
  distance: 'D',
  power: 'W',
}
const BIKE_GROUP_COLOR: Record<BestEffortGroup, string> = {
  longest: STRIDE_RAMP[6],
  elevation: ELEV_RAMP[6],
  distance: SPD_RAMP[6],
  power: HEAT_RAMP[6],
}
/** Olympic bike leg and the FTP window open first when their group has them. */
const BIKE_GROUP_DEFAULT: Partial<Record<BestEffortGroup, string>> = {
  distance: 'bike:distance:40K',
  power: 'bike:power:1200',
}

const entryYear = (entry: BestEffortEntry): number => Number(entry.date.slice(0, 4))

const byRank = (efforts: readonly BestEffortEntry[]): BestEffortEntry[] =>
  [...efforts].sort((a, b) => a.rank - b.rank)

export const buildBestEfforts = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const { formatter } = context
  const text = (key: string): string => formatter.text(key)
  const { categories, years } = data.bestEfforts
  const block = el('section', 'tri-best-efforts', undefined, { 'data-best-efforts': '' })
  const title = markGlossDefinition(
    anaTitle(formatter, 'best efforts'),
    text('best efforts definition'),
  )

  const sports = SPORTS.filter(sport => categories.some(category => category.sport === sport))
  if (sports.length === 0) {
    block.dataset.beEmpty = 'true'
    block.append(title, el('div', 'tri-ana-empty', text('no best efforts yet')))
    return { element: block, mount: () => () => {} }
  }

  const sportCategories = (sport: BestEffortSport): BestEffortCategory[] =>
    categories.filter(category => category.sport === sport)
  const runCategories = sportCategories('run')
  const groupCategories = (group: BestEffortGroup): BestEffortCategory[] =>
    categories.filter(category => category.sport === 'bike' && category.group === group)
  const bikeGroups = BIKE_GROUPS.filter(group => groupCategories(group).length > 0)
  const categoryByKey = new Map(categories.map(category => [category.key, category]))
  const rideMonths = [
    ...new Set(
      categories
        .filter(category => category.group === 'longest' || category.group === 'elevation')
        .flatMap(category => category.efforts.map(entry => entry.date.slice(0, 7))),
    ),
  ].sort((a, b) => b.localeCompare(a))

  const validRunCategory = (key: unknown): key is string =>
    typeof key === 'string' && categoryByKey.get(key)?.sport === 'run'
  const validBikeCategory = (group: BestEffortGroup, key: unknown): key is string =>
    typeof key === 'string' &&
    categoryByKey.get(key)?.sport === 'bike' &&
    categoryByKey.get(key)?.group === group
  const groupCategoryKey = (group: BestEffortGroup): string | undefined => {
    const stored = state.bikeCategory[group]
    if (validBikeCategory(group, stored)) return stored
    const preferred = BIKE_GROUP_DEFAULT[group]
    if (validBikeCategory(group, preferred)) return preferred
    return groupCategories(group)[0]?.key
  }

  const restore = (): BestEffortsState => {
    let stored: Record<string, unknown> = {}
    try {
      const parsed: unknown = JSON.parse(localStorage.getItem(TRI_BEST_EFFORTS_KEY) ?? 'null')
      if (isRecord(parsed)) stored = parsed
    } catch {}
    const sport = sports.find(option => option === stored.sport) ?? sports[0]
    const view = VIEWS.find(option => option.key === stored.view)?.key ?? 'analysis'
    const runCategory = validRunCategory(stored.runCategory)
      ? stored.runCategory
      : validRunCategory('run:5K')
        ? 'run:5K'
        : (runCategories[0]?.key ?? '')
    const bikeGroup =
      bikeGroups.find(group => group === stored.bikeGroup) ??
      (bikeGroups.includes('longest') ? 'longest' : (bikeGroups[0] ?? 'longest'))
    const bikeCategory: Partial<Record<BestEffortGroup, string>> = {}
    if (isRecord(stored.bikeCategory))
      for (const group of bikeGroups) {
        const key = stored.bikeCategory[group]
        if (validBikeCategory(group, key)) bikeCategory[group] = key
      }
    const year = typeof stored.year === 'number' && years.includes(stored.year) ? stored.year : null
    const sort = SORTS.find(option => option.key === stored.sort)?.key ?? 'newest'
    const rideRange = RANGES.find(option => option.key === stored.rideRange)?.key ?? 'all'
    const rideMonth = rideMonths.find(month => month === stored.rideMonth) ?? rideMonths[0] ?? ''
    return { sport, view, runCategory, bikeGroup, bikeCategory, year, sort, rideRange, rideMonth }
  }

  const state = restore()

  const persist = (): void => {
    try {
      localStorage.setItem(TRI_BEST_EFFORTS_KEY, JSON.stringify(state))
    } catch {}
  }

  const rideRangeBounds = (): { from: string; to: string } | undefined => {
    if (state.rideRange === 'all') return undefined
    if (state.rideRange === 'month') {
      const monthEnd = new Date(
        Date.UTC(Number(state.rideMonth.slice(0, 4)), Number(state.rideMonth.slice(5)), 0),
      )
        .toISOString()
        .slice(0, 10)
      return {
        from: `${state.rideMonth}-01`,
        to: monthEnd < data.meta.today ? monthEnd : data.meta.today,
      }
    }
    const days = RANGES.find(option => option.key === state.rideRange)?.days ?? 1
    return { from: shiftIsoDay(data.meta.today, 1 - days), to: data.meta.today }
  }

  const activeCategory = (): BestEffortCategory => {
    const key = state.sport === 'run' ? state.runCategory : groupCategoryKey(state.bikeGroup)
    const category = (key ? categoryByKey.get(key) : undefined) ?? sportCategories(state.sport)[0]
    if (category.group !== 'longest' && category.group !== 'elevation') return category
    const range = rideRangeBounds()
    if (!range) return category
    return {
      ...category,
      efforts: category.efforts.filter(entry => entry.date >= range.from && entry.date <= range.to),
    }
  }

  const ridePeriodActive = (): boolean =>
    state.sport === 'bike' && (state.bikeGroup === 'longest' || state.bikeGroup === 'elevation')
  const periodLabel = (): string =>
    ridePeriodActive() && state.rideRange !== 'all'
      ? state.rideRange === 'month'
        ? formatter.monthYear(`${state.rideMonth}-01`)
        : text(state.rideRange)
      : text('all-time')

  const activityLink = (
    entry: BestEffortEntry,
    className: string,
    label = entry.name || text('activity'),
    attrs: Record<string, string> = {},
  ): HTMLElement =>
    el('a', className, label, {
      ...attrs,
      ...powerCurveActivityLinkAttributes({ activityId: entry.id, activityDate: entry.date }),
    })

  const dateNode = (entry: BestEffortEntry, label: string): HTMLElement =>
    el('time', 'tri-be-date', label, { datetime: entry.date })

  const valueNode = (category: BestEffortCategory, entry: BestEffortEntry): HTMLElement => {
    const display = bestEffortDisplay(category, entry, formatter)
    const node = el('span', 'tri-be-value')
    node.appendChild(el('strong', 'tri-be-primary', display.primary))
    if (display.secondary) node.appendChild(el('span', 'tri-be-secondary', display.secondary))
    return node
  }

  const controls = el('div', 'tri-dist-controls tri-be-controls')
  const sportControls = el('div', 'tri-dist-sports tri-be-sports', undefined, {
    role: 'group',
    'aria-label': text('best efforts sport'),
  })
  const sportButtons = new Map<BestEffortSport, HTMLButtonElement>()
  for (const sport of sports) {
    const button = el(
      'button',
      `tri-radar-sport tri-dist-sport tri-radar-sport--${sport}`,
      undefined,
      {
        type: 'button',
        'aria-label': text(sport),
        'aria-pressed': String(sport === state.sport),
        title: text(sport),
        'data-be-sport-option': sport,
        'data-be-focus': `sport:${sport}`,
      },
    ) as HTMLButtonElement
    button.appendChild(buildIcon(context.presentation, sport))
    sportButtons.set(sport, button)
    sportControls.appendChild(button)
  }
  const tablist = el('div', 'tri-map-tablist tri-be-tabs', undefined, {
    role: 'tablist',
    'aria-label': text('best efforts view'),
  })
  const tabs = new Map<BestEffortsView, HTMLButtonElement>()
  for (const view of VIEWS) {
    const tab = el('button', 'tri-map-tab tri-be-tab', text(view.label), {
      type: 'button',
      role: 'tab',
      id: `tri-be-tab-${view.key}`,
      'aria-controls': 'tri-be-panel',
      'data-be-view-option': view.key,
      'data-be-focus': `view:${view.key}`,
    }) as HTMLButtonElement
    tabs.set(view.key, tab)
    tablist.appendChild(tab)
  }
  controls.append(sportControls, tablist)

  // Category tabs follow the activity map's metric tabs: a short code that widens to the full
  // label when selected. Each list is kept so the expansion animates in place.
  const categoryTablist = (
    name: string,
    label: string,
    kind: 'category' | 'group' | 'range',
    options: readonly CategoryTabOption[],
  ): CategoryTablist => {
    const element = el('div', 'tri-map-tablist tri-be-cats', undefined, {
      role: 'tablist',
      'aria-label': label,
      'data-be-cats': name,
    })
    const tabs = options.map(option => {
      const tab = el('button', 'tri-map-tab', undefined, {
        type: 'button',
        role: 'tab',
        'aria-label': option.label,
        'aria-controls': 'tri-be-panel',
        [`data-be-${kind}-option`]: option.value,
        'data-shortcut': option.short[0].toLowerCase(),
      }) as HTMLButtonElement
      tab.style.setProperty('--tri-map-tab-shortcut-width', `${option.short.length}ch`)
      tab.style.setProperty('--tri-map-tab-label-width', `${option.label.length}ch`)
      tab.append(
        el('span', 'tri-map-tab-shortcut', option.short, { 'aria-hidden': 'true' }),
        el('span', 'tri-map-tab-label', option.label, { 'aria-hidden': 'true' }),
      )
      element.appendChild(tab)
      return tab
    })
    return { name, element, tabs, values: options.map(option => option.value) }
  }
  const categoryOptions = (options: readonly BestEffortCategory[]): CategoryTabOption[] =>
    options.map(option => ({
      value: option.key,
      label: bestEffortCategoryLabel(option, formatter),
      short: bestEffortShortLabel(option, formatter),
    }))
  const categoryLists = new Map<string, CategoryTablist>()
  const addCategoryList = (list: CategoryTablist): void => {
    categoryLists.set(list.name, list)
  }
  if (runCategories.length > 0)
    addCategoryList(
      categoryTablist(
        'run',
        text('best efforts category'),
        'category',
        categoryOptions(runCategories),
      ),
    )
  if (bikeGroups.length > 0)
    addCategoryList(
      categoryTablist(
        'bike',
        text('best efforts category'),
        'group',
        bikeGroups.map(group => ({
          value: group,
          label: text(BIKE_GROUP_LABEL[group]),
          short: BIKE_GROUP_SHORT[group],
        })),
      ),
    )
  for (const group of ['distance', 'power'] as const)
    if (bikeGroups.includes(group))
      addCategoryList(
        categoryTablist(
          group,
          text(BIKE_GROUP_LABEL[group]),
          'category',
          categoryOptions(groupCategories(group)),
        ),
      )

  if (rideMonths.length > 0)
    addCategoryList(
      categoryTablist(
        'range',
        text('date range'),
        'range',
        RANGES.map(option => ({
          value: option.key,
          label: text(option.label),
          short: text(option.label),
        })),
      ),
    )

  const catbar = el('div', 'tri-be-catbar')
  const categoryRow = el('div', 'tri-be-category-row')
  const bikeTabs = categoryLists.get('bike')
  if (bikeTabs) {
    controls.insertBefore(bikeTabs.element, tablist)
    bikeTabs.tabs.forEach((tab, i) => {
      tab.style.setProperty('--tri-be-tab-color', BIKE_GROUP_COLOR[bikeGroups[i]])
    })
  }
  catbar.append(controls, categoryRow)
  const panel = el('div', 'tri-be-panel', undefined, { role: 'tabpanel', id: 'tri-be-panel' })
  block.append(title, catbar, panel)

  let mounted = false
  let chart: BestEffortsChartView | null = null
  let chartCleanup: (() => void) | null = null
  let chartSlot: HTMLElement | null = null
  let listSlot: HTMLElement | null = null
  let pickers: Picker[] = []

  const pickerFor = (node: unknown): Picker | undefined =>
    node instanceof Node ? pickers.find(picker => picker.wrap.contains(node)) : undefined

  const closeMenu = (picker: Picker, restoreFocus = false): void => {
    picker.menu.hidden = true
    picker.trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) picker.trigger.focus({ preventScroll: true })
  }

  const openMenu = (picker: Picker): void => {
    for (const other of pickers) if (other !== picker) closeMenu(other)
    picker.menu.hidden = false
    picker.trigger.setAttribute('aria-expanded', 'true')
    picker.options.find(option => option.getAttribute('aria-selected') === 'true')?.focus()
  }

  const dropChart = (): void => {
    chartCleanup?.()
    chartCleanup = null
    chart = null
  }
  const mountChart = (): void => {
    if (mounted && chart && !chartCleanup) chartCleanup = chart.mount()
  }

  const visibleCategoryLists = (): CategoryTablist[] => {
    const names =
      state.sport === 'run'
        ? ['run']
        : state.bikeGroup === 'distance' || state.bikeGroup === 'power'
          ? ['bike', state.bikeGroup]
          : ['bike', 'range']
    return names.flatMap(name => {
      const list = categoryLists.get(name)
      return list ? [list] : []
    })
  }

  const renderCategories = (category: BestEffortCategory, animate: boolean): void => {
    const visible = visibleCategoryLists()
    if (bikeTabs) bikeTabs.element.hidden = state.sport !== 'bike'
    const rows = visible.filter(list => list.name !== 'bike').map(list => list.element)
    if (ridePeriodActive())
      rows.push(
        buildPicker(
          'month',
          text('month'),
          rideMonths.map(month => ({ value: month, label: formatter.monthYear(`${month}-01`) })),
          state.rideMonth,
        ),
      )
    const shown = Array.from(categoryRow.children)
    // Rows are swapped only when the set changes: a kept row stays in the DOM, so its tabs
    // transition from the old selection to the new one.
    rows.forEach((row, i) => {
      if (shown[i] === row) return
      if (shown[i]) shown[i].replaceWith(row)
      else categoryRow.appendChild(row)
    })
    for (const row of shown.slice(rows.length)) row.remove()
    for (const list of visible) {
      const selected = Math.max(
        0,
        list.values.indexOf(
          list.name === 'bike'
            ? state.bikeGroup
            : list.name === 'range'
              ? state.rideRange
              : category.key,
        ),
      )
      if (!animate) list.element.dataset.motion = 'instant'
      list.tabs.forEach((tab, i) => {
        tab.setAttribute('aria-selected', String(i === selected))
        tab.tabIndex = i === selected ? 0 : -1
      })
    }
    if (animate) return
    catbar.getBoundingClientRect()
    for (const list of visible) delete list.element.dataset.motion
  }

  const buildHero = (category: BestEffortCategory, label: string): HTMLElement => {
    const ranked = byRank(category.efforts)
    const best = ranked[0]
    if (!best) return el('div', 'tri-ana-empty', text('no efforts in the selected period'))
    const hero = el('section', 'tri-be-hero', undefined, {
      'aria-label': `${periodLabel()} · ${label}`,
    })
    const main = el('div', 'tri-be-hero-main')
    const record = el('div', 'tri-be-hero-pr', undefined, {
      'data-be-entry': String(best.id),
      'data-be-rank': String(best.rank),
    })
    record.append(buildBestEffortRank(best.rank, formatter), valueNode(category, best))
    const meta = el('div', 'tri-be-hero-meta')
    meta.append(
      dateNode(best, formatter.longDate(best.date)),
      activityLink(best, 'tri-be-activity'),
    )
    main.append(el('span', 'tri-be-cap', periodLabel()), record, meta)
    hero.appendChild(main)
    const runners = ranked.slice(1, 3)
    if (runners.length > 0) {
      const list = el('ol', 'tri-be-hero-runners')
      for (const entry of runners) {
        const display = bestEffortDisplay(category, entry, formatter)
        const item = el('li', 'tri-be-hero-runner', undefined, {
          'data-be-entry': String(entry.id),
          'data-be-rank': String(entry.rank),
        })
        const date = formatter.longDate(entry.date)
        item.append(
          buildBestEffortRank(entry.rank, formatter),
          el('strong', 'tri-be-primary', display.primary),
          activityLink(entry, 'tri-be-activity tri-be-hero-runner-link', date, {
            'aria-label': `${entry.name || text('activity')} · ${date}`,
            ...(entry.name ? { title: entry.name } : {}),
          }),
        )
        list.appendChild(item)
      }
      hero.appendChild(list)
    }
    return hero
  }

  // Rows read like the activity feed: rank box, name, then right-aligned date and values.
  const buildRow = (category: BestEffortCategory, entry: BestEffortEntry): HTMLElement => {
    const year = entryYear(entry)
    const display = bestEffortDisplay(category, entry, formatter)
    const item = el('li', 'tri-be-row', undefined, {
      'data-be-entry': String(entry.id),
      'data-be-rank': String(entry.rank),
      'data-be-year-rank': String(entry.yearRank),
    })
    const details = el('details', 'tri-feed-row tri-be-details')
    const summary = el('summary', 'tri-feed-head tri-be-summary')
    const sub = el('span', 'tri-feed-sub')
    sub.append(
      el('time', 'tri-feed-c tri-feed-c--date', entry.date, { datetime: entry.date }),
      el('span', 'tri-feed-c tri-be-c--value', display.primary),
      el('span', 'tri-feed-c tri-be-c--rate', display.secondary ?? '-'),
    )
    summary.append(
      buildBestEffortRank(entry.rank, formatter),
      el('span', 'tri-feed-name', entry.name || text('activity')),
      sub,
    )
    const more = el('div', 'tri-be-more')
    const facts = el('dl', 'tri-be-facts')
    const fact = (label: string, value: string): void => {
      const row = el('div', 'tri-be-fact')
      row.append(el('dt', undefined, label), el('dd', undefined, value))
      facts.appendChild(row)
    }
    fact(text('rank'), `#${entry.rank} ${text('all-time rank')} · #${entry.yearRank} ${year}`)
    if (entry.heartRate != null)
      fact(text('heart rate'), `${formatter.number(Math.round(entry.heartRate))} bpm`)
    if (entry.wattsPerKg != null && category.unit !== 'watts')
      fact('W/kg', `${formatter.number(entry.wattsPerKg, 2)} W/kg`)
    more.append(activityLink(entry, 'tri-be-activity'), facts)
    if (entry.indoor) more.appendChild(el('span', 'tri-be-tag', text('indoor')))
    details.append(summary, more)
    item.appendChild(details)
    return item
  }

  // Arena-style section band: label left, count right; sticky inside the list scroller.
  const band = (label: string, count: number): HTMLElement => {
    const node = el('div', 'tri-be-band')
    node.append(
      el('h3', 'tri-be-band-title', label),
      el('span', 'tri-be-band-count', String(count)),
    )
    return node
  }

  const byDate = (a: BestEffortEntry, b: BestEffortEntry): number =>
    a.date.localeCompare(b.date) || a.id - b.id
  const sortEfforts = (efforts: BestEffortEntry[]): BestEffortEntry[] => {
    switch (state.sort) {
      case 'oldest':
        return efforts.sort(byDate)
      case 'high':
        return efforts.sort((a, b) => b.value - a.value || byDate(b, a))
      case 'low':
        return efforts.sort((a, b) => a.value - b.value || byDate(b, a))
      case 'newest':
        return efforts.sort((a, b) => byDate(b, a))
    }
  }

  const buildList = (category: BestEffortCategory): HTMLElement => {
    const list = el('div', 'tri-be-list')
    const box = el('div', 'tri-be-box')
    const scroller = el('div', 'tri-be-scroll')
    box.appendChild(scroller)
    list.appendChild(box)
    const efforts = sortEfforts(
      category.efforts.filter(entry => state.year == null || entryYear(entry) === state.year),
    )
    if (efforts.length === 0) {
      scroller.appendChild(
        el(
          'div',
          'tri-ana-empty',
          text(
            ridePeriodActive() && state.rideRange !== 'all'
              ? 'no efforts in the selected period'
              : 'no efforts in the selected year',
          ),
        ),
      )
      return list
    }
    // A season of rides is ~120 rows; scrolling them in place keeps the panels below within reach.
    // Value sorts order rows inside each year band; the top 10 tab holds the all-time order.
    const groups = new Map<number, BestEffortEntry[]>()
    const bandYears = [...new Set(efforts.map(entryYear))].sort((a, b) =>
      state.sort === 'oldest' ? a - b : b - a,
    )
    for (const year of bandYears) groups.set(year, [])
    for (const entry of efforts) groups.get(entryYear(entry))?.push(entry)
    for (const [year, entries] of groups) {
      const section = el('section', 'tri-be-year-group', undefined, {
        'data-be-year-group': String(year),
      })
      const rows = el('ul', 'tri-be-rows')
      for (const entry of entries) rows.appendChild(buildRow(category, entry))
      section.append(band(String(year), entries.length), rows)
      scroller.appendChild(section)
    }
    return list
  }

  const renderList = (category: BestEffortCategory): void => {
    listSlot?.replaceChildren(buildList(category))
  }

  const renderYearRegion = (category: BestEffortCategory): void => {
    if (!chartSlot || !listSlot) return
    dropChart()
    const range = ridePeriodActive() ? rideRangeBounds() : undefined
    chart = buildBestEffortsChart({
      category,
      context,
      year: state.year,
      today: data.meta.today,
      range,
    })
    chartSlot.replaceChildren(chart.element)
    renderList(category)
    mountChart()
  }

  // Listbox pickers use the lab date picker's styling and keyboard behavior.
  const buildPicker = (
    name: Picker['name'],
    labelText: string,
    choices: readonly { value: string; label: string }[],
    selected: string,
  ): HTMLElement => {
    const field = el('div', 'tri-be-field')
    const label = el('span', 'tri-be-field-label', labelText, { id: `tri-be-${name}-label` })
    if (name === 'month') label.hidden = true
    const wrap = el('div', 'tri-be-picker')
    const trigger = document.createElement('button')
    trigger.type = 'button'
    trigger.className = 'tri-be-picker-trigger'
    trigger.id = `tri-be-${name}-trigger`
    trigger.dataset.bePicker = name
    trigger.dataset.beFocus = name
    trigger.textContent = choices.find(choice => choice.value === selected)?.label ?? ''
    trigger.setAttribute('aria-labelledby', `${label.id} ${trigger.id}`)
    trigger.setAttribute('aria-haspopup', 'listbox')
    trigger.setAttribute('aria-expanded', 'false')
    trigger.setAttribute('aria-controls', `tri-be-${name}-menu`)
    if (name === 'month') trigger.dataset.active = String(state.rideRange === 'month')
    const menu = el('div', 'tri-be-picker-menu', undefined, {
      id: `tri-be-${name}-menu`,
      role: 'listbox',
      'aria-labelledby': label.id,
    })
    menu.hidden = true
    const options = choices.map(choice => {
      const option = document.createElement('button')
      option.type = 'button'
      option.className = 'tri-be-picker-option'
      option.dataset.beOption = choice.value
      option.setAttribute('role', 'option')
      option.setAttribute('aria-selected', String(choice.value === selected))
      option.tabIndex = -1
      option.append(
        el('span', 'tri-be-picker-check', '✓', { 'aria-hidden': 'true' }),
        el('span', 'tri-be-picker-text', choice.label),
      )
      return option
    })
    menu.append(...options)
    wrap.append(trigger, menu)
    pickers.push({ name, wrap, trigger, menu, options })
    field.append(label, wrap)
    return field
  }

  const buildFilters = (): HTMLElement => {
    const filters = el('div', 'tri-be-filter')
    filters.append(
      buildPicker(
        'year',
        text('filter by year'),
        [
          { value: 'all', label: text('all years') },
          ...years.map(year => ({ value: String(year), label: String(year) })),
        ],
        state.year == null ? 'all' : String(state.year),
      ),
      buildPicker(
        'sort',
        text('sort efforts'),
        SORTS.map(option => ({ value: option.key, label: text(option.label) })),
        state.sort,
      ),
    )
    return filters
  }

  const buildTop = (category: BestEffortCategory, label: string): HTMLElement => {
    const top = byRank(category.efforts).slice(0, 10)
    const box = el('div', 'tri-be-box tri-be-top')
    const scroller = el('div', 'tri-be-scroll')
    const rows = el('ol', 'tri-be-rows', undefined, {
      'aria-label': `${text('top 10')} · ${label}`,
    })
    for (const entry of top) rows.appendChild(buildRow(category, entry))
    const scope = ridePeriodActive() && state.rideRange !== 'all' ? ` · ${periodLabel()}` : ''
    scroller.append(band(`${text('top 10')} · ${label}${scope}`, top.length), rows)
    if (top.length === 0)
      scroller.appendChild(el('div', 'tri-ana-empty', text('no efforts in the selected period')))
    box.appendChild(scroller)
    return box
  }

  const renderPanel = (category: BestEffortCategory): void => {
    dropChart()
    chartSlot = null
    listSlot = null
    const label = bestEffortCategoryLabel(category, formatter)
    panel.setAttribute('aria-labelledby', `tri-be-tab-${state.view}`)
    if (state.view === 'top') {
      panel.replaceChildren(buildTop(category, label))
      return
    }
    chartSlot = el('div', 'tri-be-chart-slot')
    listSlot = el('div', 'tri-be-list-slot')
    panel.replaceChildren(buildHero(category, label), chartSlot, buildFilters(), listSlot)
    renderYearRegion(category)
  }

  const syncState = (category: BestEffortCategory): void => {
    block.style.setProperty(
      '--tri-be-category-color',
      state.sport === 'bike' ? BIKE_GROUP_COLOR[state.bikeGroup] : SPD_RAMP[6],
    )
    block.dataset.beSport = state.sport
    block.dataset.beView = state.view
    block.dataset.beCategory = category.key
    block.dataset.beYear = state.year == null ? 'all' : String(state.year)
    block.dataset.beSort = state.sort
    block.dataset.beRange = ridePeriodActive() ? state.rideRange : 'all'
    block.dataset.beMonth = state.rideMonth
    for (const [sport, button] of sportButtons)
      button.setAttribute('aria-pressed', String(sport === state.sport))
    for (const [view, tab] of tabs) {
      const selected = view === state.view
      tab.setAttribute('aria-selected', String(selected))
      tab.tabIndex = selected ? 0 : -1
    }
  }

  const render = (animate = false): void => {
    const category = activeCategory()
    pickers = []
    syncState(category)
    renderCategories(category, animate)
    renderPanel(category)
  }

  // Rebuilt controls lose focus; move it to the replacement for the activated control.
  const update = (
    change: () => void,
    scope: 'all' | 'year' | 'list' = 'all',
    animate = false,
  ): void => {
    const active = document.activeElement
    const focusKey =
      active instanceof HTMLElement && block.contains(active) ? active.dataset.beFocus : undefined
    change()
    persist()
    if (scope === 'all') render(animate)
    else {
      const category = activeCategory()
      syncState(category)
      if (scope === 'year') renderYearRegion(category)
      else renderList(category)
    }
    if (!focusKey) return
    const target = Array.from(block.querySelectorAll<HTMLElement>('[data-be-focus]')).find(
      node => node.dataset.beFocus === focusKey,
    )
    if (target && target !== document.activeElement) target.focus()
  }

  const selectView = (view: BestEffortsView): void => {
    if (view === state.view) return
    update(() => {
      state.view = view
    })
  }

  // Category tabs stay in the DOM across renders, so focus stays on the activated tab.
  const selectCategoryTab = (tab: HTMLButtonElement, animate: boolean): void => {
    const { beGroupOption, beCategoryOption, beRangeOption } = tab.dataset
    if (beRangeOption) {
      const range = RANGES.find(option => option.key === beRangeOption)?.key
      if (!range || range === state.rideRange) return
      update(
        () => {
          state.rideRange = range
        },
        'all',
        animate,
      )
    } else if (beGroupOption) {
      const group = bikeGroups.find(option => option === beGroupOption)
      if (!group || group === state.bikeGroup) return
      update(
        () => {
          state.bikeGroup = group
        },
        'all',
        animate,
      )
    } else if (beCategoryOption) {
      const category = categoryByKey.get(beCategoryOption)
      if (!category || category.key === activeCategory().key) return
      update(
        () => {
          if (category.sport === 'run') state.runCategory = category.key
          else state.bikeCategory = { ...state.bikeCategory, [category.group]: category.key }
        },
        'all',
        animate,
      )
    }
  }

  // The pickers live outside the list, so a year or sort change keeps them and only syncs them.
  const choosePickerOption = (picker: Picker, value: string): void => {
    if (picker.name === 'month') {
      if (!rideMonths.includes(value)) return
      closeMenu(picker)
      update(() => {
        state.rideRange = 'month'
        state.rideMonth = value
      })
      pickers.find(option => option.name === 'month')?.trigger.focus({ preventScroll: true })
      return
    } else if (picker.name === 'year') {
      const next = value === 'all' ? null : years.find(year => String(year) === value)
      if (next === undefined) return
      if (next !== state.year)
        update(() => {
          state.year = next
        }, 'year')
    } else {
      const next = SORTS.find(option => option.key === value)?.key
      if (!next) return
      if (next !== state.sort)
        update(() => {
          state.sort = next
        }, 'list')
    }
    for (const option of picker.options) {
      const selected = option.dataset.beOption === value
      option.setAttribute('aria-selected', String(selected))
      if (selected)
        picker.trigger.textContent =
          option.querySelector('.tri-be-picker-text')?.textContent ?? value
    }
    closeMenu(picker, true)
  }

  const onClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const control = target.closest<HTMLButtonElement>('button')
    if (!control || !block.contains(control)) return
    const {
      beSportOption,
      beViewOption,
      beGroupOption,
      beCategoryOption,
      beRangeOption,
      beOption,
    } = control.dataset
    const picker = pickerFor(control)
    if (picker && control === picker.trigger) {
      if (picker.menu.hidden) openMenu(picker)
      else closeMenu(picker)
    } else if (picker && beOption != null) {
      choosePickerOption(picker, beOption)
    } else if (beSportOption) {
      const sport = sports.find(option => option === beSportOption)
      if (!sport || sport === state.sport) return
      update(() => {
        state.sport = sport
      })
    } else if (beViewOption) {
      const view = VIEWS.find(option => option.key === beViewOption)?.key
      if (view) selectView(view)
    } else if (beGroupOption || beCategoryOption || beRangeOption) {
      // A pointer click animates the width change; Enter and Space (detail 0) snap like the arrows.
      selectCategoryTab(control, event.detail > 0)
    }
  }

  const onTabKeydown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const target = event.target
    if (!(target instanceof Element) || !target.closest('.tri-be-tab')) return
    const active = VIEWS.findIndex(view => view.key === state.view)
    let next = -1
    if (event.key === 'ArrowLeft') next = (active - 1 + VIEWS.length) % VIEWS.length
    else if (event.key === 'ArrowRight') next = (active + 1) % VIEWS.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = VIEWS.length - 1
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    const view = VIEWS[next].key
    selectView(view)
    tabs.get(view)?.focus()
  }

  // Arrows wrap, Home and End jump, and a letter or digit cycles through the tabs whose code
  // starts with it, as on the activity map.
  const onCategoryKeydown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const target = event.target
    if (!(target instanceof HTMLButtonElement)) return
    const name = target.closest<HTMLElement>('[data-be-cats]')?.dataset.beCats
    const list = name ? categoryLists.get(name) : undefined
    if (!list) return
    const active = list.tabs.indexOf(target)
    const count = list.tabs.length
    const next =
      event.key === 'ArrowLeft'
        ? (active - 1 + count) % count
        : event.key === 'ArrowRight'
          ? (active + 1) % count
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? count - 1
              : nextMapMetricShortcutIndex(
                  list.tabs.map(tab => tab.dataset.shortcut),
                  active,
                  event.key,
                )
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    const tab = list.tabs[next]
    selectCategoryTab(tab, false)
    tab.focus({ preventScroll: true })
    tab.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }

  const onPickerKeydown = (event: KeyboardEvent): void => {
    const picker = pickerFor(event.target)
    if (!picker) return
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return
    if (event.key === 'Escape' && !picker.menu.hidden) {
      event.preventDefault()
      event.stopPropagation()
      closeMenu(picker, true)
      return
    }
    if (event.key === 'Tab') {
      closeMenu(picker, true)
      return
    }
    if (event.target === picker.trigger) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        event.stopPropagation()
        openMenu(picker)
      }
      return
    }
    const active = picker.options.findIndex(option => option === document.activeElement)
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? picker.options.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(picker.options.length - 1, active + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, active - 1)
              : -1
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    picker.options[next]?.focus()
  }

  const onPickerFocusout = (event: FocusEvent): void => {
    for (const picker of pickers)
      if (!(event.relatedTarget instanceof Node && picker.wrap.contains(event.relatedTarget)))
        closeMenu(picker)
  }

  const onDocumentPointerdown = (event: PointerEvent): void => {
    const path = event.composedPath()
    for (const picker of pickers) if (!path.includes(picker.wrap)) closeMenu(picker)
  }

  render()
  return {
    element: block,
    mount: () => {
      mounted = true
      block.addEventListener('click', onClick)
      block.addEventListener('keydown', onPickerKeydown)
      block.addEventListener('focusout', onPickerFocusout)
      document.addEventListener('pointerdown', onDocumentPointerdown)
      tablist.addEventListener('keydown', onTabKeydown)
      catbar.addEventListener('keydown', onCategoryKeydown)
      const linksCleanup = setupPowerCurveActivityLinks(block, context)
      mountChart()
      return () => {
        mounted = false
        block.removeEventListener('click', onClick)
        block.removeEventListener('keydown', onPickerKeydown)
        block.removeEventListener('focusout', onPickerFocusout)
        document.removeEventListener('pointerdown', onDocumentPointerdown)
        for (const picker of pickers) closeMenu(picker)
        tablist.removeEventListener('keydown', onTabKeydown)
        catbar.removeEventListener('keydown', onCategoryKeydown)
        linksCleanup()
        chartCleanup?.()
        chartCleanup = null
      }
    },
  }
}
