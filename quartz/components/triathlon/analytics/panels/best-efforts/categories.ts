import type { BestEffortCategory, BestEffortGroup } from '../../../../../util/best-efforts'
import type { TriathlonContext } from '../../../runtime/context'
import { el } from '../../../runtime/dom'
import { nextMapMetricShortcutIndex } from '../../../shell/command-palette'
import { bestEffortCategoryLabel, bestEffortShortLabel } from '.././best-efforts-format'
import {
  BIKE_GROUP_COLOR,
  BIKE_GROUP_LABEL,
  BIKE_GROUP_SHORT,
  BestEffortsState,
  CategoryTabOption,
  CategoryTablist,
} from './model'

export function createCategoryControls({
  formatter,
  state,
  controls,
  tablist,
  runCategories,
  bikeGroups,
  groupCategories,
  selectCategoryTab,
}: {
  formatter: TriathlonContext['formatter']
  state: BestEffortsState
  controls: HTMLElement
  tablist: HTMLElement
  runCategories: BestEffortCategory[]
  bikeGroups: BestEffortGroup[]
  groupCategories: (group: BestEffortGroup) => BestEffortCategory[]
  selectCategoryTab: (tab: HTMLButtonElement, animate: boolean) => void
}) {
  const text = (key: string): string => formatter.text(key)

  // Category tabs follow the activity map's metric tabs: a short code that widens to the full
  // label when selected. Each list is kept so the expansion animates in place.
  const categoryTablist = (
    name: string,
    label: string,
    kind: 'category' | 'group',
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

  const catbar = el('div', 'tri-be-catbar')
  const categoryRow = el('div', 'tri-be-category-row')
  const bikeTabs = categoryLists.get('bike')
  if (bikeTabs) {
    controls.insertBefore(bikeTabs.element, tablist)
    bikeTabs.tabs.forEach((tab, i) => {
      tab.style.setProperty('--tri-be-tab-color', BIKE_GROUP_COLOR[bikeGroups[i]])
    })
  }
  // One row: the group's categories sit between the bike groups and the view tabs, and wrap
  // only when the panel is too narrow.
  controls.insertBefore(categoryRow, tablist)
  catbar.append(controls)

  const visibleCategoryLists = (): CategoryTablist[] => {
    const names =
      state.sport === 'run'
        ? ['run']
        : state.bikeGroup === 'distance' || state.bikeGroup === 'power'
          ? ['bike', state.bikeGroup]
          : ['bike']
    return names.flatMap(name => {
      const list = categoryLists.get(name)
      return list ? [list] : []
    })
  }

  const renderCategories = (category: BestEffortCategory, animate: boolean): void => {
    const visible = visibleCategoryLists()
    if (bikeTabs) bikeTabs.element.hidden = state.sport !== 'bike'
    const rows = visible.filter(list => list.name !== 'bike').map(list => list.element)
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
        list.values.indexOf(list.name === 'bike' ? state.bikeGroup : category.key),
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
  return { catbar, renderCategories, onCategoryKeydown }
}
