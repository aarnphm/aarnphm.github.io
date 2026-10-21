import type { ActivityKind } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { DEFAULT_TRI_ACTIVITY_LAYOUT, TRI_ACTIVITY_LAYOUTS } from '../triathlon-activity-layout'
import { triText } from '../triathlon-i18n'

// Pages hold `perPage` activities; the last page backs up so it stays full, matching the snap
// positions the track can reach.
export const dayTrackPages = (
  count: number,
  perPage: number,
): readonly { start: number; end: number }[] =>
  Array.from({ length: Math.ceil(count / perPage) }, (_, index) => {
    const start = Math.max(0, Math.min(index * perPage, count - perPage))
    return { start, end: Math.min(count, start + perPage) }
  })

export const buildDayPageTabs = <N>(
  f: TriNodeFactory<N>,
  sports: readonly ActivityKind[],
  perPage: number,
  selected: number,
): N[] =>
  dayTrackPages(sports.length, perPage).map(({ start, end }, index) => {
    const label = sports
      .slice(start, end)
      .map(sport => triText(f.presentation.locale, sport))
      .join(' · ')
    const shortcut = String(index + 1)
    const tab = f.el('button', 'tri-map-tab tri-day-page-tab', undefined, {
      type: 'button',
      role: 'tab',
      tabindex: index === selected ? '0' : '-1',
      'aria-label': label,
      'aria-selected': String(index === selected),
      'data-page': String(index),
      'data-shortcut': shortcut,
      style: `--tri-map-tab-shortcut-width:${shortcut.length}ch;--tri-map-tab-label-width:${label.length}ch`,
    })
    f.add(
      tab,
      f.el('span', 'tri-map-tab-shortcut', shortcut, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-map-tab-label', label, { 'aria-hidden': 'true' }),
    )
    return tab
  })

// Server output assumes the default two-up layout; the client rebuilds the page tabs and the
// pressed mode once it reads the stored layout and measures the track.
export const buildDayTrackPager = <N>(f: TriNodeFactory<N>, sports: readonly ActivityKind[]): N => {
  const locale = f.presentation.locale
  const pager = f.el('div', 'tri-day-pager', undefined, { 'data-paged': String(sports.length > 2) })
  const pages = f.el('div', 'tri-map-tablist tri-day-pages', undefined, {
    role: 'tablist',
    'aria-label': triText(locale, 'activities'),
    'data-i18n-aria-label': 'activities',
  })
  f.add(pages, ...buildDayPageTabs(f, sports, 2, 0))
  const modes = f.el('div', 'tri-map-tablist tri-day-modes', undefined, {
    role: 'group',
    'aria-label': triText(locale, 'activity layout'),
    'data-i18n-aria-label': 'activity layout',
  })
  for (const layout of TRI_ACTIVITY_LAYOUTS)
    f.add(
      modes,
      f.el(
        'button',
        'tri-map-tab tri-day-mode',
        layout === 'one' ? '1' : layout === 'two' ? '2' : triText(locale, 'all activities'),
        {
          type: 'button',
          'data-layout': layout,
          'aria-pressed': String(layout === DEFAULT_TRI_ACTIVITY_LAYOUT),
          ...(layout === 'all' ? { 'data-i18n': 'all activities' } : {}),
        },
      ),
    )
  f.add(pager, pages, modes)
  return pager
}
