import type { CardModel } from '../press-kit-card'
import { ACTIVITY_CARD_STATS, type ActivityCardData } from '../../../util/press-kit-activity'
import { triLabel, triText } from '../../../util/triathlon-i18n'
import { dayCardData, type TrainingCardBlock } from '../press-kit-day'
import { localizeActivity } from '../press-kit-i18n'
import {
  cardDate,
  runsFromNode,
  runsFromRange,
  runsFromText,
  runsToPlain,
  type Run,
} from '../press-kit-text'
import {
  arts,
  blockDisplays,
  selectedActivities,
  selectedBlocks,
  selectedMetrics,
  session,
  shareLocale,
  show,
  type PressKitTab,
  type Quote,
  type Subject,
} from './session-state'
import { selectedPropositions } from './tractatus'

const TABS: PressKitTab[] = ['note', 'quote', 'links']

const SOURCES = 'article.popover-hint, .stream-entry-body'

export function absolute(href: string): string {
  try {
    const url = new URL(href, session.page!.url)
    // A local preview still shares the canonical address.
    if (url.origin === location.origin)
      return `https://${session.page!.site}${url.pathname}${url.search}${url.hash}`
    return url.href
  } catch {
    return href
  }
}

export function headingText(heading: HTMLElement): string {
  const alias = heading.getAttribute('data-heading-alias')?.trim()
  if (alias) return alias
  return runsToPlain(runsFromNode(heading)).replace(/\s+/g, ' ').trim()
}

function excerptOf(root: Element | null): Run[] {
  if (!root) return []
  for (const p of Array.from(root.querySelectorAll(':scope > p, :scope > div > p'))) {
    if (p.closest('.callout, figure, table, .footnotes, details')) continue
    const runs = runsFromNode(p)
    if (runsToPlain(runs).length >= 60) return runs
  }
  return []
}

export function pageSubject(): Subject {
  const title = document.querySelector('[data-article-title] .article-title')
  const description = document.querySelector('[data-article-title] .description')
  const article = document.querySelector('article.popover-hint')
  return {
    kind: 'page',
    url: session.page!.url,
    title: title ? runsFromNode(title) : [{ text: document.title, style: 'roman' }],
    description: description ? runsFromNode(description) : [],
    excerpt: excerptOf(article),
    date: session.page!.date,
    iso: session.page!.iso,
    reading: session.page!.reading,
    tags: session.page!.tags,
    crumbs: session.page!.crumbs,
    headings: Array.from(
      document.querySelectorAll<HTMLElement>('.page-content :is(h2, h3, h4)[id]'),
    ).filter(heading => !heading.closest('.press-kit, .stream-entry') && headingText(heading)),
  }
}

// reading-time's default rate, which the build uses for a page's minutes.
export const readingTime = (text: string): string | undefined => {
  const words = text.split(/\s+/).filter(Boolean).length
  return words > 0 ? `${Math.ceil(words / 200)} min` : undefined
}

// A subject's breadcrumb is its address, as a page lists its slug; a query or hash is not a segment.
const crumbsOf = (url: string): string[] => [
  session.page!.site,
  ...new URL(url).pathname
    .split('/')
    .filter(Boolean)
    .map(part => part.toLowerCase()),
]

export function entrySubject(entry: HTMLElement, href: string | undefined): Subject {
  const title = entry.querySelector('.stream-entry-title')
  const body = entry.querySelector('.stream-entry-body')
  const content = document.createDocumentFragment()
  const collected = dayCardData(entry)
  // A day card's entry block keeps the description in its text; a plain entry sets it apart, as a
  // note sets its description under the title.
  const descriptionNode = collected
    ? null
    : body?.querySelector(':scope > .stream-entry-description')
  body
    ?.querySelectorAll(
      collected
        ? ':scope > .stream-entry-description, :scope > .stream-entry-content'
        : ':scope > .stream-entry-content',
    )
    .forEach(node => content.append(node.cloneNode(true)))
  const day = collected
    ? {
        blocks: collected.blocks.map(block => ({
          ...block,
          url: absolute(block.url),
          activities: block.activities.map(activity => ({
            ...activity,
            url: absolute(activity.url),
          })),
        })),
      }
    : undefined
  if (day)
    content
      .querySelectorAll(
        '.tri-day-embed, .tri-compare-embed, .tri-calc-embed, figure.audio-player, .transclude-src',
      )
      .forEach(node => node.remove())
  const time = entry.querySelector<HTMLTimeElement>(
    '.stream-entry-date time, time.stream-entry-date',
  )
  const url = absolute(href ?? `${location.pathname}#${entry.id}`)
  const excerpt = runsFromNode(content)
  const description = descriptionNode ? runsFromNode(descriptionNode) : []
  const activities =
    day?.blocks.filter(block => block.kind === 'activity').flatMap(block => block.activities) ?? []
  return {
    kind: day ? 'day' : 'entry',
    entryTitle: title?.textContent?.trim(),
    entryDate: time?.dateTime ? cardDate(time.dateTime, shareLocale()) : undefined,
    url,
    title: activities.length
      ? runsFromText(activities.map(activity => activity.title).join(' + '))
      : title
        ? runsFromNode(title)
        : [],
    description,
    excerpt,
    date: (() => {
      // A training day carries the day it trained, which can precede the entry's own day.
      const iso = day
        ? (entry.querySelector<HTMLElement>('.tri-day-embed')?.dataset.triathlonDate ??
          day.blocks.flatMap(block => block.activities)[0]?.date ??
          time?.dateTime)
        : time?.dateTime
      return iso ? cardDate(iso, shareLocale()) : time?.textContent?.trim() || undefined
    })(),
    iso: time?.dateTime || undefined,
    reading: day ? undefined : readingTime(`${runsToPlain(description)} ${runsToPlain(excerpt)}`),
    tags: Array.from(entry.querySelectorAll('.stream-entry-tag'), tag => tag.textContent!.trim())
      .filter(Boolean)
      .slice(0, 4),
    crumbs: crumbsOf(url),
    headings: [],
    day,
  }
}

export function activitySubject(activity: ActivityCardData): Subject {
  const url = absolute(activity.url)
  return {
    kind: 'activity',
    url,
    title: runsFromText(activity.title),
    description: [],
    excerpt: runsFromText(activity.stats.map(stat => `${stat.label}: ${stat.value}`).join('\n')),
    date: cardDate(activity.date, shareLocale()),
    iso: activity.date,
    tags: [activity.sport],
    crumbs: crumbsOf(url),
    headings: [],
    activity: { ...activity, url: absolute(activity.url) },
  }
}

export function citationAllowed(): boolean {
  const slug = session.page?.slug ?? ''
  return (
    session.subject?.kind !== 'activity' &&
    session.subject?.kind !== 'day' &&
    session.subject?.kind !== 'calendar' &&
    slug !== 'stream' &&
    !slug.startsWith('stream/on') &&
    slug !== 'triathlon' &&
    !slug.startsWith('triathlon/')
  )
}

export function availableTabs(): PressKitTab[] {
  return citationAllowed() ? TABS : ['note', 'links']
}

export function sourceOf(node: Node): Element | null {
  const el = node instanceof Element ? node : node.parentElement
  const source = el?.closest(SOURCES) ?? null
  return source && !source.closest('.press-kit') ? source : null
}

export function selectedRange(): Range | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  if (!sourceOf(range.commonAncestorContainer)) return null
  return selection.toString().trim().length >= 3 ? range.cloneRange() : null
}

function sectionBefore(node: Node, headings: HTMLElement[]): HTMLElement | undefined {
  let found: HTMLElement | undefined
  for (const heading of headings) {
    if (heading === node || heading.contains(node)) return heading
    if (heading.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING) found = heading
    else break
  }
  return found
}

/** Scroll-to-text directive; ranges through math skip it, since KaTeX text never matches. */
function fragmentFor(range: Range): string | undefined {
  const fragment = range.cloneContents()
  if (fragment.querySelector('.katex')) return undefined
  const words = range.toString().replace(/\s+/g, ' ').trim().split(' ').filter(Boolean)
  if (words.length === 0) return undefined
  const encode = (text: string) =>
    encodeURIComponent(text).replace(/-/g, '%2D').replace(/,/g, '%2C').replace(/&/g, '%26')
  if (words.length <= 8) return `text=${encode(words.join(' '))}`
  return `text=${encode(words.slice(0, 4).join(' '))},${encode(words.slice(-4).join(' '))}`
}

export function quoteFrom(range: Range | null): Quote {
  if (!range || !session.subject) return { runs: session.subject?.excerpt ?? [], chosen: false }
  const source = sourceOf(range.commonAncestorContainer)
  return {
    runs: runsFromRange(range, source),
    propositions: selectedPropositions(range, source),
    chosen: true,
    fragment: fragmentFor(range),
    section: sectionBefore(range.startContainer, session.subject.headings),
  }
}

export function sectionForCard(): HTMLElement | undefined {
  if (session.anchor.startsWith('#')) {
    return session.subject?.headings.find(heading => `#${heading.id}` === session.anchor)
  }
  return session.quote.section
}

export function shareUrl(): string {
  if (!session.subject) return session.page?.url ?? location.href
  if (session.anchor.startsWith('block:')) {
    const block = session.subject.day?.blocks.find(block => `block:${block.id}` === session.anchor)
    if (block) return absolute(block.url)
  }
  const base = session.subject.url.split('#')[0]
  if (session.anchor === 'passage' && session.quote.fragment) {
    const section = session.quote.section ? `#${session.quote.section.id}` : '#'
    return `${base}${section}:~:${session.quote.fragment}`
  }
  if (session.anchor.startsWith('#')) return `${base}${session.anchor}`
  return session.subject.url
}

export function model(): CardModel {
  const s = session.subject!
  const locale = shareLocale()
  const section = sectionForCard()
  const kind = session.tab === 'quote' ? 'quote' : s.kind === 'page' ? 'note' : s.kind
  const customTitle = session.editor?.title
  const attribution = kind === 'quote' ? session.quote.attribution?.trim() : undefined
  const day = s.day
    ? {
        blocks: activeDayBlocks().map(block => ({
          ...block,
          title:
            block.kind === 'memo'
              ? locale === 'fr'
                ? 'mémo audio'
                : 'audio memo'
              : block.kind === 'comparison'
                ? `${triText(locale, 'comparison')} · ${[...new Set(block.activities.map(activity => activity.date))].join(' / ')}`
                : block.title,
          rows: block.rows.map(row => ({ ...row, label: triLabel(locale, row.label) })),
          activities: block.activities.map(activity => localizeActivity(activity, locale)),
          display: blockDisplays.get(block.id),
        })),
      }
    : undefined
  const entry = runsToPlain(s.excerpt).trim()
  if (day && show.entry && entry)
    day.blocks.unshift({
      id: 'entry',
      kind: 'entry',
      title: s.entryTitle ?? 'entrée',
      url: s.url,
      activities: [],
      rows: [],
      text: entry,
      date: s.entryDate,
      reading: readingTime(entry),
      display: blockDisplays.get('entry'),
    })
  return {
    kind,
    locale,
    crumbs: s.crumbs,
    customTitle,
    title: customTitle
      ? [{ text: customTitle, style: 'roman' }]
      : attribution
        ? [{ text: attribution, style: 'roman' }]
        : s.title.length > 0
          ? s.title
          : [{ text: session.page!.site, style: 'roman' }],
    description: s.description,
    body: session.tab === 'quote' ? session.quote.runs : s.excerpt,
    propositions:
      session.tab === 'quote' && show.tractatus ? session.quote.propositions : undefined,
    date: s.date,
    reading: s.reading,
    tags: s.tags,
    section: section ? headingText(section) : undefined,
    art: session.theme === 'encre' ? arts.night : arts.day,
    // The night card's moon sits over the camp, so its crops centre higher.
    artFocus: session.theme === 'encre' ? [640, 280] : undefined,
    activity: s.activity
      ? { ...localizeActivity(s.activity, locale), title: customTitle ?? s.activity.title }
      : undefined,
    day,
    calendar: s.calendar,
    overlay: session.overlay,
    metricGap: session.editor?.metricGap ?? 0,
    railTextScale: session.editor?.railTextScale ?? 1,
    powerWindow: session.powerWindow,
    powerCurveScale: session.powerCurveScale,
    metricKeys: Object.fromEntries(Array.from(selectedMetrics, ([url, keys]) => [url, [...keys]])),
    show: {
      ...show,
      meta: kind === 'activity' || kind === 'day' || show.meta,
      excerpt: kind === 'entry' || show.excerpt,
    },
  }
}

export const titleText = () =>
  session.subject?.day?.blocks.find(block => `block:${block.id}` === session.anchor)?.title ??
  (runsToPlain(session.subject?.title ?? []) || session.page?.site || '')

export function subjectActivities(): ActivityCardData[] {
  const activities =
    session.subject?.day?.blocks.flatMap(block => block.activities) ??
    (session.subject?.activity ? [session.subject.activity] : [])
  const unique = new Map<string, ActivityCardData>()
  for (const activity of activities) {
    const previous = unique.get(activity.url)
    if (!previous || previous.stats.length < activity.stats.length)
      unique.set(activity.url, activity)
  }
  return [...unique.values()]
}

export function activeDayBlocks(): TrainingCardBlock[] {
  return (session.subject?.day?.blocks ?? [])
    .filter(block => selectedBlocks.has(block.id))
    .flatMap(block => {
      if (!block.activities.length) return [block]
      const activities = block.activities.filter(activity => selectedActivities.has(activity.url))
      if (!activities.length || (block.kind === 'comparison' && activities.length < 2)) return []
      if (activities.length === block.activities.length) return [block]
      return [{ ...block, activities, graphic: undefined, rows: [] }]
    })
}

export function activeActivities(): ActivityCardData[] {
  return session.subject?.day
    ? activeDayBlocks().flatMap(block => block.activities)
    : session.subject?.activity
      ? [session.subject.activity]
      : []
}

export function syncMetricIncludes() {
  for (const key of ACTIVITY_CARD_STATS)
    show[key] = subjectActivities().some(activity => selectedMetrics.get(activity.url)?.has(key))
}
