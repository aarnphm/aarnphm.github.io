import {
  documentPreviewUrl,
  isDocumentPreviewType,
  readDocumentPreview,
} from '../../../util/document-preview'
import { fetchCanonical } from '../../../util/fetch-canonical'
import {
  lessWrongPreviewApiUrl,
  type LessWrongTarget,
  readLessWrongPreview,
} from '../../../util/lesswrong'
import { normalizeRelativeURLs } from '../../../util/path'
import { readSepPreview, sepPreviewApiUrl, type SepTarget } from '../../../util/sep'
import { decodeCalcShare } from '../../../util/triathlon-calculator'
import {
  readTriathlonHubPreview,
  triathlonHubPreviewUrl,
} from '../../../util/triathlon-hub-preview'
import {
  readTriathlonCalendarPreview,
  readTriathlonPreview,
  triathlonCalendarPreviewUrl,
  triathlonPopoverDomId,
  type TriathlonPreview,
  type TriathlonPreviewActivity,
  triathlonPreviewDayHref,
  triathlonPreviewTarget,
  type TriathlonPreviewTarget,
  triathlonPreviewUrl,
} from '../../../util/triathlon-preview'
import { isStravaActivityIndex, STRAVA_ACTIVITY_INDEX_PATH } from '../../../util/triathlon-shortcut'
import {
  readWikipediaPreviewResponse,
  wikipediaActionApiUrl,
  type WikipediaTarget,
} from '../../../util/wikipedia'
import { coursePreviewTarget, renderCoursePreview } from '../course-preview-card'
import { curiusPagePreviewSource } from '../curius-page-preview'
import { curiusPopoverSource } from '../curius-popover'
import { renderDocumentPreview } from '../document-preview-card'
import {
  renderTriathlonCalculatorPreview,
  renderTriathlonCalendarPreview,
  renderTriathlonComparisonPreview,
  renderTriathlonPreview,
  triathlonPreviewPresentation,
} from '../triathlon-preview-card'
import { p } from './content-render'
import {
  renderLessWrongPreview,
  renderSepPreview,
  renderWikipediaPreview,
} from './external-preview-cards'
import { compareUrls } from './placement'
import { renderTriathlonHubPreview } from './triathlon-hub-preview-card'

function wikipediaTargetFromLink(link: HTMLAnchorElement): WikipediaTarget | undefined {
  const { wikipediaLang, wikipediaTitle } = link.dataset
  if (!wikipediaLang || !wikipediaTitle) return undefined
  return { lang: wikipediaLang, title: wikipediaTitle }
}

function lessWrongTargetFromLink(link: HTMLAnchorElement): LessWrongTarget | undefined {
  const { lesswrongPostId, lesswrongSlug } = link.dataset
  if (!lesswrongPostId) return undefined
  return { postId: lesswrongPostId, ...(lesswrongSlug ? { slug: lesswrongSlug } : {}) }
}

function sepTargetFromLink(link: HTMLAnchorElement): SepTarget | undefined {
  const { sepEntry, sepArchive } = link.dataset
  if (!sepEntry) return undefined
  return { entry: sepEntry, ...(sepArchive ? { archive: sepArchive } : {}) }
}

export interface SummaryPreviewSource {
  popoverId: string
  className: string
  loadingLabel?: string
  load(signal: AbortSignal): Promise<((popoverInner: HTMLDivElement) => void) | null>
}

function wikipediaPreviewSource(target: WikipediaTarget): SummaryPreviewSource {
  return {
    popoverId: `popover-wikipedia-${target.lang}-${encodeURIComponent(target.title)}`,
    className: 'wikipedia-popover',
    async load(signal) {
      const response = await fetch(wikipediaActionApiUrl(target), { signal })
      if (!response.ok) return null
      const preview = readWikipediaPreviewResponse(await response.json(), target)
      return preview ? popoverInner => renderWikipediaPreview(preview, popoverInner) : null
    },
  }
}

function lessWrongPreviewSource(target: LessWrongTarget): SummaryPreviewSource {
  return {
    popoverId: `popover-lesswrong-${target.postId}`,
    className: 'lesswrong-popover',
    async load(signal) {
      const url = lessWrongPreviewApiUrl(target, window.location.toString())
      const response = await fetch(url, { signal })
      if (!response.ok) return null
      const preview = readLessWrongPreview(await response.json())
      return preview ? popoverInner => renderLessWrongPreview(preview, popoverInner) : null
    },
  }
}

function sepPreviewSource(target: SepTarget): SummaryPreviewSource {
  return {
    popoverId: `popover-sep-${target.archive ?? 'current'}-${target.entry}`,
    className: 'sep-popover',
    async load(signal) {
      const response = await fetch(sepPreviewApiUrl(target, window.location.toString()), { signal })
      if (!response.ok) return null
      const preview = readSepPreview(await response.json())
      return preview ? popoverInner => renderSepPreview(preview, popoverInner) : null
    },
  }
}

function triathlonTargetFromLink(link: HTMLAnchorElement): TriathlonPreviewTarget | null {
  const date = link.dataset.triathlonDate
  if (date) {
    const activityId = link.dataset.powerActivityId
    return { kind: 'day', date, activityId: activityId ?? null }
  }
  // A same-page activity anchor scrolls to content already on screen.
  if (compareUrls(new URL(document.location.href), new URL(link.href))) return null
  return triathlonPreviewTarget(link.href)
}

let activityDatesRequest: Promise<Record<string, string> | null> | null = null

function activityDates(): Promise<Record<string, string> | null> {
  activityDatesRequest ??= fetch(new URL(STRAVA_ACTIVITY_INDEX_PATH, window.location.href))
    .then(async response => {
      if (!response.ok) return null
      const index = await response.json()
      return isStravaActivityIndex(index) ? index.activities : null
    })
    .catch(() => null)
  return activityDatesRequest.then(dates => {
    if (!dates) activityDatesRequest = null
    return dates
  })
}

async function fetchDayPreview(
  date: string,
  signal: AbortSignal,
): Promise<TriathlonPreview | null> {
  const response = await fetch(triathlonPreviewUrl(date, window.location.href), { signal })
  if (!response.ok) return null
  return readTriathlonPreview(await response.json(), date)
}

async function comparisonActivities(
  ids: readonly string[],
  signal: AbortSignal,
): Promise<TriathlonPreviewActivity[] | null> {
  const dates = await activityDates()
  if (!dates || signal.aborted) return null
  const byDate = new Map<string, string[]>()
  for (const id of ids) {
    const date = dates[id]
    if (!date) continue
    const group = byDate.get(date) ?? []
    group.push(id)
    byDate.set(date, group)
  }
  if (byDate.size === 0) return null
  const previews = await Promise.all([...byDate.keys()].map(date => fetchDayPreview(date, signal)))
  const activities = new Map<string, TriathlonPreviewActivity>()
  for (const preview of previews) {
    if (!preview) continue
    for (const activity of preview.activities) activities.set(String(activity.id), activity)
  }
  const matched = ids.flatMap(id => {
    const activity = activities.get(id)
    return activity ? [activity] : []
  })
  return matched.length > 0 ? matched : null
}

function triathlonPreviewSource(
  link: HTMLAnchorElement,
  target: TriathlonPreviewTarget,
): SummaryPreviewSource {
  const presentation = triathlonPreviewPresentation()
  const href = link.href
  return {
    popoverId: triathlonPopoverDomId(target, presentation.distance),
    className: 'triathlon-popover document-popover',
    async load(signal) {
      if (target.kind === 'calculator') {
        const share = decodeCalcShare(target.raw)
        return share
          ? popoverInner => renderTriathlonCalculatorPreview(share, href, popoverInner)
          : null
      }
      if (target.kind === 'calendar') {
        const response = await fetch(triathlonCalendarPreviewUrl(window.location.href), { signal })
        if (!response.ok) return null
        const preview = readTriathlonCalendarPreview(await response.json())
        return preview
          ? popoverInner => renderTriathlonCalendarPreview(preview, target, href, popoverInner)
          : null
      }
      if (target.kind === 'comparison') {
        const activities = await comparisonActivities(target.ids, signal)
        return activities
          ? popoverInner =>
              renderTriathlonComparisonPreview(activities, href, presentation, popoverInner)
          : null
      }
      if (target.kind === 'hub') {
        const response = await fetch(triathlonHubPreviewUrl(window.location.href), { signal })
        if (!response.ok) return null
        const preview = readTriathlonHubPreview(await response.json())
        return preview
          ? popoverInner => renderTriathlonHubPreview(preview, href, presentation, popoverInner)
          : null
      }
      const preview = await fetchDayPreview(target.date, signal)
      return preview
        ? popoverInner =>
            renderTriathlonPreview(
              preview,
              target,
              triathlonPreviewDayHref(target.date, href),
              presentation,
              popoverInner,
            )
        : null
    },
  }
}

function documentPreviewSource(link: HTMLAnchorElement): SummaryPreviewSource | null {
  const type = link.dataset.documentPreview
  const slug = link.dataset.slug
  if (!isDocumentPreviewType(type) || !slug) return null
  if (compareUrls(new URL(document.location.href), new URL(link.href))) return null
  const href = new URL(link.href)
  href.hash = ''
  return {
    popoverId: `popover-${type}-${slug}`,
    className: 'document-popover',
    async load(signal) {
      const response = await fetch(documentPreviewUrl(slug, window.location.href), { signal })
      if (!response.ok) return null
      const preview = readDocumentPreview(await response.json(), type, slug)
      return preview
        ? popoverInner => renderDocumentPreview(preview, href.toString(), popoverInner)
        : null
    },
  }
}

function coursePreviewSource(link: HTMLAnchorElement): SummaryPreviewSource | null {
  const href = coursePreviewTarget(new URL(link.href), new URL(document.location.href))
  if (!href) return null
  return {
    popoverId: `popover-course-${href.pathname.replace(/\/+$/, '')}`,
    className: 'course-popover',
    async load(signal) {
      const response = await fetchCanonical(href, { signal })
      if (!response.ok) return null
      const html = p.parseFromString(await response.text(), 'text/html')
      normalizeRelativeURLs(html, href)
      if (!html.querySelector('.course-home, [data-course-card], h1')) return null
      return popoverInner => renderCoursePreview(html, href, popoverInner)
    },
  }
}

/** Links that open a compact preview instead of the full page body. */
export function summaryPreviewSource(link: HTMLAnchorElement): SummaryPreviewSource | null {
  const curius = curiusPopoverSource(link)
  if (curius) return curius
  const curiusPage = curiusPagePreviewSource(link)
  if (curiusPage) return curiusPage
  const course = coursePreviewSource(link)
  if (course) return course
  const page = documentPreviewSource(link)
  if (page) return page
  const wikipedia = wikipediaTargetFromLink(link)
  if (wikipedia) return wikipediaPreviewSource(wikipedia)
  const lessWrong = lessWrongTargetFromLink(link)
  if (lessWrong) return lessWrongPreviewSource(lessWrong)
  const sep = sepTargetFromLink(link)
  if (sep) return sepPreviewSource(sep)
  const triathlon = triathlonTargetFromLink(link)
  return triathlon ? triathlonPreviewSource(link, triathlon) : null
}
