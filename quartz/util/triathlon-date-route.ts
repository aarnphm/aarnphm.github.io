import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { FullSlug } from './path'

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/
const TRIATHLON_ON_SLUG = 'triathlon/on' as FullSlug
const TRIATHLON_YEAR_SLUG = /^triathlon\/on\/(\d{4})$/
const TRIATHLON_MONTH_SLUG = /^triathlon\/on\/(\d{4})\/(\d{2})$/
const TRIATHLON_DAY_SLUG = /^triathlon\/on\/(\d{4})\/(\d{2})\/(\d{2})$/
const SHORTCUT_DATE_PATH = /^\/(\d{4})(?:\/(\d{2})(?:\/(\d{2}))?)?\/?$/

export type TriathlonDateRoute =
  | { kind: 'index' }
  | { kind: 'year'; year: string }
  | { kind: 'month'; year: string; month: string }
  | { kind: 'day'; year: string; month: string; day: string; date: string }

export interface TriathlonFeedRoute {
  slug: FullSlug
  title: string
}

export interface TriathlonFeedScope {
  prefix: string
  title: string
}

const daysInMonth = (year: number, month: number): number => {
  if (month === 2) {
    const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
    return leap ? 29 : 28
  }
  return [4, 6, 9, 11].includes(month) ? 30 : 31
}

const validDateParts = (year: string, month: string, day: string): boolean => {
  const y = Number(year)
  const m = Number(month)
  const d = Number(day)
  return y > 0 && m >= 1 && m <= 12 && d >= 1 && d <= daysInMonth(y, m)
}

const validYear = (year: string): boolean => /^\d{4}$/.test(year) && Number(year) > 0

const validMonth = (year: string, month: string): boolean =>
  validYear(year) && /^\d{2}$/.test(month) && Number(month) >= 1 && Number(month) <= 12

export const triathlonOnSlug = (): FullSlug => TRIATHLON_ON_SLUG

export const triathlonYearSlug = (year: string): FullSlug | null =>
  validYear(year) ? (`${TRIATHLON_ON_SLUG}/${year}` as FullSlug) : null

export const triathlonMonthSlug = (year: string, month: string): FullSlug | null =>
  validMonth(year, month) ? (`${TRIATHLON_ON_SLUG}/${year}/${month}` as FullSlug) : null

export const triathlonDaySlug = (date: string): FullSlug | null => {
  const match = ISO_DATE.exec(date)
  if (!match || !validDateParts(match[1], match[2], match[3])) return null
  return `${TRIATHLON_ON_SLUG}/${match[1]}/${match[2]}/${match[3]}` as FullSlug
}

export const triathlonDateRouteFromSlug = (slug: string): TriathlonDateRoute | null => {
  if (slug === TRIATHLON_ON_SLUG) return { kind: 'index' }
  const day = TRIATHLON_DAY_SLUG.exec(slug)
  if (day && validDateParts(day[1], day[2], day[3])) {
    return {
      kind: 'day',
      year: day[1],
      month: day[2],
      day: day[3],
      date: `${day[1]}-${day[2]}-${day[3]}`,
    }
  }
  const month = TRIATHLON_MONTH_SLUG.exec(slug)
  if (month && validMonth(month[1], month[2])) {
    return { kind: 'month', year: month[1], month: month[2] }
  }
  const year = TRIATHLON_YEAR_SLUG.exec(slug)
  if (year && validYear(year[1])) return { kind: 'year', year: year[1] }
  return null
}

export const triathlonDateFromSlug = (slug: string): string | null => {
  const route = triathlonDateRouteFromSlug(slug)
  return route?.kind === 'day' ? route.date : null
}

export const triathlonFeedScopeFromSlug = (slug: string): TriathlonFeedScope | null => {
  const route = triathlonDateRouteFromSlug(slug)
  if (!route || route.kind === 'day') return null
  if (route.kind === 'index') return { prefix: '', title: 'feed' }
  if (route.kind === 'year') return { prefix: route.year, title: route.year }
  return { prefix: `${route.year}-${route.month}`, title: `${route.year} / ${route.month}` }
}

export const triathlonOnSlugFromShortcutPath = (pathname: string): FullSlug | null => {
  const match = SHORTCUT_DATE_PATH.exec(pathname)
  if (!match) return null
  if (match[3]) return triathlonDaySlug(`${match[1]}-${match[2]}-${match[3]}`)
  if (match[2]) return triathlonMonthSlug(match[1], match[2])
  return triathlonYearSlug(match[1])
}

export interface TriathlonTreeDay {
  date: string
  day: string
  slug: FullSlug
  count: number
  sports: string[]
  km: number
  timeS: number
}

export interface TriathlonTreeMonth {
  month: string
  slug: FullSlug
  count: number
  km: number
  timeS: number
  days: TriathlonTreeDay[]
}

export interface TriathlonTreeYear {
  year: string
  slug: FullSlug
  count: number
  km: number
  timeS: number
  months: TriathlonTreeMonth[]
}

const SPORT_ORDER = ['swim', 'bike', 'run', 'walk']

const sortSports = (sports: Iterable<string>): string[] =>
  [...sports].sort((left, right) => {
    const l = SPORT_ORDER.indexOf(left)
    const r = SPORT_ORDER.indexOf(right)
    if (l !== -1 || r !== -1)
      return (l === -1 ? SPORT_ORDER.length : l) - (r === -1 ? SPORT_ORDER.length : r)
    return left.localeCompare(right)
  })

export const triathlonDateTree = (
  details: Readonly<
    Record<string, Pick<StravaActivityDetail, 'date' | 'sport' | 'distanceKm' | 'movingTimeS'>>
  >,
  prefix = '',
): TriathlonTreeYear[] => {
  const byDate = new Map<
    string,
    { count: number; sports: Set<string>; km: number; timeS: number }
  >()
  for (const detail of Object.values(details)) {
    if (!detail.date.startsWith(prefix) || triathlonDaySlug(detail.date) === null) continue
    const entry = byDate.get(detail.date) ?? { count: 0, sports: new Set(), km: 0, timeS: 0 }
    entry.count += 1
    entry.sports.add(detail.sport)
    entry.km += detail.distanceKm
    entry.timeS += detail.movingTimeS
    byDate.set(detail.date, entry)
  }

  const years = new Map<string, TriathlonTreeYear>()
  for (const date of [...byDate.keys()].sort((left, right) => right.localeCompare(left))) {
    const entry = byDate.get(date)!
    const year = date.slice(0, 4)
    const month = date.slice(5, 7)
    let yearNode = years.get(year)
    if (!yearNode) {
      yearNode = { year, slug: triathlonYearSlug(year)!, count: 0, km: 0, timeS: 0, months: [] }
      years.set(year, yearNode)
    }
    let monthNode = yearNode.months.find(node => node.month === month)
    if (!monthNode) {
      monthNode = {
        month,
        slug: triathlonMonthSlug(year, month)!,
        count: 0,
        km: 0,
        timeS: 0,
        days: [],
      }
      yearNode.months.push(monthNode)
    }
    monthNode.days.push({
      date,
      day: date.slice(8, 10),
      slug: triathlonDaySlug(date)!,
      count: entry.count,
      sports: sortSports(entry.sports),
      km: entry.km,
      timeS: entry.timeS,
    })
    monthNode.count += entry.count
    monthNode.km += entry.km
    monthNode.timeS += entry.timeS
    yearNode.count += entry.count
    yearNode.km += entry.km
    yearNode.timeS += entry.timeS
  }
  return [...years.values()]
}

export const triathlonActivityDates = (
  details: Readonly<Record<string, Pick<StravaActivityDetail, 'date'>>>,
): string[] =>
  [...new Set(Object.values(details).map(detail => detail.date))]
    .filter(date => triathlonDaySlug(date) !== null)
    .sort()

export const triathlonActivityFeedRoutes = (
  details: Readonly<Record<string, Pick<StravaActivityDetail, 'date'>>>,
): TriathlonFeedRoute[] => {
  const routes = new Map<FullSlug, string>()
  for (const date of triathlonActivityDates(details)) {
    const year = date.slice(0, 4)
    const month = date.slice(5, 7)
    const yearSlug = triathlonYearSlug(year)
    const monthSlug = triathlonMonthSlug(year, month)
    if (yearSlug) routes.set(yearSlug, `triathlon · ${year}`)
    if (monthSlug) routes.set(monthSlug, `triathlon · ${year} / ${month}`)
  }
  return [...routes.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([slug, title]) => ({ slug, title }))
}
