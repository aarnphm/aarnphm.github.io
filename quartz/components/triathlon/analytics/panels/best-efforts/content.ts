import type { BestEffortCategory, BestEffortEntry } from '../../../../../util/best-efforts'
import type { TriathlonContext } from '../../../runtime/context'
import { compareStravaIds } from '../../../../../util/strava-id'
import { powerCurveActivityLinkAttributes } from '../../../../../util/triathlon-power-activity'
import { el } from '../../../runtime/dom'
import { bestEffortDisplay, buildBestEffortRank } from '.././best-efforts-format'
import { BestEffortsSort, BestEffortsState, byRank, entryYear } from './model'

export function createBestEffortsContent(formatter: TriathlonContext['formatter']) {
  const text = (key: string): string => formatter.text(key)

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

  const buildHero = (category: BestEffortCategory, label: string): HTMLElement => {
    const ranked = byRank(category.efforts)
    const best = ranked[0]
    if (!best) return el('div', 'tri-ana-empty', text('no efforts in the selected period'))
    const hero = el('section', 'tri-be-hero', undefined, {
      'aria-label': `${text('all-time')} · ${label}`,
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
    main.append(el('span', 'tri-be-cap', text('all-time')), record, meta)
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
    a.date.localeCompare(b.date) || compareStravaIds(a.id, b.id)

  const sortEfforts = (efforts: BestEffortEntry[], sort: BestEffortsSort): BestEffortEntry[] => {
    switch (sort) {
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

  const buildList = (
    category: BestEffortCategory,
    state: Pick<BestEffortsState, 'year' | 'sort'>,
  ): HTMLElement => {
    const list = el('div', 'tri-be-list')
    const box = el('div', 'tri-be-box')
    const scroller = el('div', 'tri-be-scroll')
    box.appendChild(scroller)
    list.appendChild(box)
    const efforts = sortEfforts(
      category.efforts.filter(entry => state.year == null || entryYear(entry) === state.year),
      state.sort,
    )
    if (efforts.length === 0) {
      scroller.appendChild(el('div', 'tri-ana-empty', text('no efforts in the selected year')))
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

  const buildTop = (category: BestEffortCategory, label: string): HTMLElement => {
    const top = byRank(category.efforts).slice(0, 10)
    const box = el('div', 'tri-be-box tri-be-top')
    const scroller = el('div', 'tri-be-scroll')
    const rows = el('ol', 'tri-be-rows', undefined, {
      'aria-label': `${text('top 10')} · ${label}`,
    })
    for (const entry of top) rows.appendChild(buildRow(category, entry))
    scroller.append(band(`${text('top 10')} · ${label}`, top.length), rows)
    if (top.length === 0)
      scroller.appendChild(el('div', 'tri-ana-empty', text('no efforts in the selected period')))
    box.appendChild(scroller)
    return box
  }
  return { buildHero, buildList, buildTop }
}
