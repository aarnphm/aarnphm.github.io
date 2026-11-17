import type { Analytics } from '../../../../plugins/stores/analytics'
import type {
  BestEffortCategory,
  BestEffortGroup,
  BestEffortSport,
} from '../../../../util/best-efforts'
import type { TriathlonContext } from '../../runtime/context'
import { isRecord } from '../../../../util/type-guards'
import { buildSelect, type Select } from '../../../controls/select'
import { setupPowerCurveActivityLinks } from '../../activity/power-links'
import { buildIcon } from '../../activity/primitives'
import { SPD_RAMP } from '../../maps/palette'
import { el } from '../../runtime/dom'
import { anaTitle, markGlossDefinition } from '../shared'
import { buildBestEffortsChart, type BestEffortsChartView } from './best-efforts-chart'
import { bestEffortCategoryLabel } from './best-efforts-format'
import { createCategoryControls } from './best-efforts/categories'
import { createBestEffortsContent } from './best-efforts/content'
import {
  BIKE_GROUPS,
  BIKE_GROUP_COLOR,
  BIKE_GROUP_DEFAULT,
  BestEffortsState,
  BestEffortsView,
  SORTS,
  SPORTS,
  TRI_BEST_EFFORTS_KEY,
  VIEWS,
} from './best-efforts/model'
export { TRI_BEST_EFFORTS_KEY } from './best-efforts/model'

interface Picker {
  name: 'year' | 'sort'
  select: Select<string>
  dispose: (() => void) | null
}

export const buildBestEfforts = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const { formatter } = context
  const text = (key: string): string => formatter.text(key)
  const { buildHero, buildList, buildTop } = createBestEffortsContent(formatter)
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
    return { sport, view, runCategory, bikeGroup, bikeCategory, year, sort }
  }

  const state = restore()

  const persist = (): void => {
    try {
      localStorage.setItem(TRI_BEST_EFFORTS_KEY, JSON.stringify(state))
    } catch {}
  }

  const activeCategory = (): BestEffortCategory => {
    const key = state.sport === 'run' ? state.runCategory : groupCategoryKey(state.bikeGroup)
    return (key ? categoryByKey.get(key) : undefined) ?? sportCategories(state.sport)[0]
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
  const { catbar, renderCategories, onCategoryKeydown } = createCategoryControls({
    formatter,
    state,
    controls,
    tablist,
    runCategories,
    bikeGroups,
    groupCategories,
    selectCategoryTab: (tab, animate) => selectCategoryTab(tab, animate),
  })

  const panel = el('div', 'tri-be-panel', undefined, { role: 'tabpanel', id: 'tri-be-panel' })
  block.append(title, catbar, panel)

  let mounted = false
  let chart: BestEffortsChartView | null = null
  let chartCleanup: (() => void) | null = null
  let chartSlot: HTMLElement | null = null
  let listSlot: HTMLElement | null = null
  let pickers: Picker[] = []

  // Pickers are rebuilt on every render; only the ones in the DOM hold listeners.
  const mountPickers = (): void => {
    if (mounted) for (const picker of pickers) picker.dispose ??= picker.select.mount()
  }
  const dropPickers = (): void => {
    for (const picker of pickers) {
      picker.dispose?.()
      picker.dispose = null
    }
  }

  const dropChart = (): void => {
    chartCleanup?.()
    chartCleanup = null
    chart = null
  }
  const mountChart = (): void => {
    if (mounted && chart && !chartCleanup) chartCleanup = chart.mount()
  }

  const renderList = (category: BestEffortCategory): void => {
    listSlot?.replaceChildren(buildList(category, state))
  }

  const renderYearRegion = (category: BestEffortCategory): void => {
    if (!chartSlot || !listSlot) return
    dropChart()
    chart = buildBestEffortsChart({ category, context, year: state.year, today: data.meta.today })
    chartSlot.replaceChildren(chart.element)
    renderList(category)
    mountChart()
  }

  const buildPicker = (
    name: Picker['name'],
    labelText: string,
    choices: { value: string; label: string }[],
    selected: string,
  ): HTMLElement => {
    const field = el('div', 'tri-be-field')
    // The trigger's own name repeats the label, so the visible copy is hidden from assistive tech.
    const label = el('span', 'tri-be-field-label', labelText, { 'aria-hidden': 'true' })
    const select = buildSelect({
      id: `tri-be-${name}-menu`,
      label: labelText,
      groups: [{ options: choices }],
      selected,
      onSelect: value => choosePickerOption(name, value),
      className: 'tri-be-picker',
    })
    select.trigger.dataset.beFocus = name
    pickers.push({ name, select, dispose: null })
    field.append(label, select.element)
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
    dropPickers()
    pickers = []
    syncState(category)
    renderCategories(category, animate)
    renderPanel(category)
    mountPickers()
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
    const { beGroupOption, beCategoryOption } = tab.dataset
    if (beGroupOption) {
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

  // The year and sort pickers live outside the list, so their changes keep them in place.
  const choosePickerOption = (name: Picker['name'], value: string): void => {
    if (name === 'year') {
      const next = value === 'all' ? null : years.find(year => String(year) === value)
      if (next !== undefined && next !== state.year)
        update(() => {
          state.year = next
        }, 'year')
    } else {
      const next = SORTS.find(option => option.key === value)?.key
      if (next && next !== state.sort)
        update(() => {
          state.sort = next
        }, 'list')
    }
  }

  const onClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const control = target.closest<HTMLButtonElement>('button')
    if (!control || !block.contains(control)) return
    const { beSportOption, beViewOption, beGroupOption, beCategoryOption } = control.dataset
    if (beSportOption) {
      const sport = sports.find(option => option === beSportOption)
      if (!sport || sport === state.sport) return
      update(() => {
        state.sport = sport
      })
    } else if (beViewOption) {
      const view = VIEWS.find(option => option.key === beViewOption)?.key
      if (view) selectView(view)
    } else if (beGroupOption || beCategoryOption) {
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

  render()
  return {
    element: block,
    mount: () => {
      mounted = true
      mountPickers()
      block.addEventListener('click', onClick)
      tablist.addEventListener('keydown', onTabKeydown)
      catbar.addEventListener('keydown', onCategoryKeydown)
      const linksCleanup = setupPowerCurveActivityLinks(block, context)
      mountChart()
      return () => {
        mounted = false
        block.removeEventListener('click', onClick)
        dropPickers()
        tablist.removeEventListener('keydown', onTabKeydown)
        catbar.removeEventListener('keydown', onCategoryKeydown)
        linksCleanup()
        chartCleanup?.()
        chartCleanup = null
      }
    },
  }
}
