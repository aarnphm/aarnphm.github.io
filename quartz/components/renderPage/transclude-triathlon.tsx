import { Element, ElementContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { h } from 'hastscript'
import { render } from 'preact-render-to-string'
import { i18n } from '../../i18n'
import { ATHLETE } from '../../plugins/stores/analytics'
import { QuartzPluginData } from '../../plugins/vfile'
import { joinSegments, pathToRoot } from '../../util/path'
import { loadStravaPayloadSync } from '../../util/strava-payload'
import { buildTriathlonCalcCard, decodeCalcShare } from '../../util/triathlon-calculator'
import { parseTriathlonCalendar, parseTriathlonCalendars } from '../../util/triathlon-calendar'
import {
  buildActivityComparison,
  powerViewActivity,
  type DayCardExtras,
  type DetailCtx,
} from '../../util/triathlon-card'
import { decodeActivityComparisonAnchor } from '../../util/triathlon-comparison'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../util/triathlon-presentation'
import {
  resolveTriathlonEmbedDate,
  triathlonCardFactory,
  triathlonDayCard,
  triathlonDayExtras,
  triathlonDayProps,
  triathlonEmbedAnchorFromBlockRef,
  triathlonEmbedAnchorFromSource,
  triathlonEmbedDayHref,
} from '../triathlon-day-card'
import { CalendarPanel } from '../triathlon/calendar/Calendar'
import { wrapCollapsible, type TranscludeScope } from './transclude-scope'

const CALC_ANCHOR_RE = /#(calculator-[a-z0-9-]+)$/

export function calendarEmbedView(
  blockRef: string | undefined,
  anchorPath: string | undefined,
): { view: 'list' | 'year'; year?: number } | undefined {
  if (anchorPath) {
    let segments: unknown
    try {
      segments = JSON.parse(anchorPath)
    } catch {
      return undefined
    }
    if (!Array.isArray(segments) || segments[0] !== 'calendar') return undefined
    const display = segments.at(-1)
    const view =
      display === 'list' || display === 'display=list'
        ? 'list'
        : display === 'year' || display === 'display=year'
          ? 'year'
          : undefined
    if (segments.length === 2 && view) return { view }
    const year = segments[1]
    if (typeof year !== 'string' || !/^\d{4}$/.test(year) || Number(year) === 0) return undefined
    if (segments.length === 2) return { view: 'list', year: Number(year) }
    if (segments.length === 3 && view) return { view, year: Number(year) }
    return undefined
  }
  if (blockRef === '#calendar' || blockRef === '#calendar-list') return { view: 'list' }
  if (blockRef === '#calendar-year') return { view: 'year' }
  return undefined
}

export function transcludeCalendar(
  scope: TranscludeScope,
  node: Element,
  page: QuartzPluginData,
  calendarView: NonNullable<ReturnType<typeof calendarEmbedView>>,
  alias: string,
  transcludeMetadata: Record<string, unknown> | undefined,
): void {
  const { slug } = scope
  const { cfg, fileData } = scope.componentData
  const calendars = parseTriathlonCalendars(page.frontmatter?.calendar, page.frontmatter?.events)
  const calendar = parseTriathlonCalendar(
    page.frontmatter?.calendar,
    calendarView.year,
    page.frontmatter?.events,
  )
  const id = `tri-calendar-embed-${slug.replace(/[^a-zA-Z0-9_-]/g, '-')}-${++scope.calendarEmbedCount}`
  const children = fromHtml(
    render(
      <CalendarPanel
        calendar={calendar}
        calendars={calendars}
        year={calendarView.year}
        embedded
        id={id}
        view={calendarView.view}
      />,
    ),
    { fragment: true },
  ).children.filter((child): child is ElementContent => child.type !== 'doctype')
  if (fileData.frontmatter?.pageLayout !== 'reflection') {
    const href =
      joinSegments(pathToRoot(slug), 'triathlon/calendar') +
      (calendarView.year !== undefined
        ? `#${calendarView.year}${calendarView.view === 'year' ? '-year' : ''}`
        : calendarView.view === 'year'
          ? '#year'
          : '')
    children.push(
      h('a.internal.transclude-src', { href }, [
        {
          type: 'text',
          value:
            alias ||
            page.frontmatter?.title ||
            i18n(cfg.locale).components.transcludes.linkToOriginal,
        },
      ]),
    )
  }
  if (transcludeMetadata && 'collapsed' in transcludeMetadata) {
    wrapCollapsible(node, children, alias || 'calendar', Boolean(transcludeMetadata.collapsed))
  } else {
    node.children = children
  }
}

/** Day card, activity comparison or calculator embed of a triathlon page; false when the ref names none. */
export function transcludeTriathlonCard(
  scope: TranscludeScope,
  node: Element,
  page: QuartzPluginData,
  blockRef: string | undefined,
  anchorPath: string | undefined,
  inner: Element,
): boolean {
  const { slug, componentData } = scope
  const { cfg, fileData } = componentData
  let triathlonDate: string | undefined
  let triathlonEmbedExtras: Pick<
    DayCardExtras,
    'sport' | 'activityId' | 'excludedActivityIds' | 'settings' | 'analytics'
  > | null = null
  let triathlonPayload: ReturnType<typeof loadStravaPayloadSync> | null = null
  if (page.frontmatter?.layout === 'triathlon') {
    const anchor =
      triathlonEmbedAnchorFromBlockRef(blockRef) ??
      triathlonEmbedAnchorFromSource(anchorPath, fileData.rawMarkdownSource)
    if (anchor) {
      const since = page.frontmatter?.['strava']
      triathlonPayload = loadStravaPayloadSync(
        typeof since === 'string' ? since : undefined,
        page.tracking,
        {
          weights: page.tracking?.days,
          events: page.tracking?.races,
          dexa: page.frontmatter?.['dexa'],
          vo2labs: page.frontmatter?.['vo2max'],
        },
        componentData.ctx?.argv.directory,
      )
      triathlonDate = resolveTriathlonEmbedDate(anchor, triathlonPayload) ?? undefined
      triathlonEmbedExtras = 'date' in anchor ? anchor : { activityId: anchor.activityId }
    }
  }
  if (triathlonDate && triathlonPayload) {
    const dayHref = triathlonEmbedDayHref(
      pathToRoot(slug),
      triathlonDate,
      triathlonEmbedExtras?.activityId,
    )
    if (!dayHref) return true
    const extras: DayCardExtras = {
      ...triathlonDayExtras(page, triathlonDate),
      ...triathlonEmbedExtras,
      embedded: true,
    }
    const payload = triathlonPayload
    const children: ElementContent[] = [
      h(
        '.tri-day-embed',
        {
          ...triathlonDayProps(extras, triathlonDate),
          'data-detail-path': joinSegments(pathToRoot(slug), 'static/strava-detail.json'),
        },
        [
          triathlonDayCard(
            triathlonDate,
            payload.totalCount > 0 || extras.analytics ? payload : null,
            extras,
            {
              zones: payload.zones,
              curveRef: payload.powerCurveRef,
              curveYearRef: payload.powerCurveYearRef,
              runCurveRef: payload.runPowerCurveRef,
              runCurveYearRef: payload.runPowerCurveYearRef,
              curveYear: payload.powerCurveYear,
              criticalPower: payload.criticalPower,
              criticalPowerYear: payload.criticalPowerYear,
              ftp: ATHLETE.ftp,
              goalFtp: ATHLETE.goalFTP,
              vt1: null,
            } satisfies DetailCtx,
          ),
        ],
      ),
    ]
    if (fileData.frontmatter?.pageLayout !== 'reflection') {
      children.push(
        h('a', { href: dayHref, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }
    node.children = children
    return true
  }

  const comparisonAnchor =
    blockRef && page.frontmatter?.layout === 'triathlon'
      ? blockRef.startsWith('#')
        ? blockRef.slice(1)
        : blockRef
      : null
  const comparisonIds = comparisonAnchor ? decodeActivityComparisonAnchor(comparisonAnchor) : null
  if (comparisonIds && comparisonAnchor) {
    const since = page.frontmatter?.['strava']
    const payload = loadStravaPayloadSync(
      typeof since === 'string' ? since : undefined,
      page.tracking,
      {
        weights: page.tracking?.days,
        events: page.tracking?.races,
        dexa: page.frontmatter?.['dexa'],
        vo2labs: page.frontmatter?.['vo2max'],
      },
      componentData.ctx?.argv.directory,
    )
    const activities = comparisonIds.flatMap(activityId => {
      const activity = payload.details[activityId]
      return activity ? [powerViewActivity(DEFAULT_TRIATHLON_PRESENTATION, activity)] : []
    })
    const children: ElementContent[] = [
      h(
        '.tri-compare-embed',
        {
          'data-compare-anchor': comparisonAnchor,
          'data-detail-path': joinSegments(pathToRoot(slug), 'static/strava-detail.json'),
        },
        [
          buildActivityComparison(
            triathlonCardFactory,
            activities,
            {
              zones: payload.zones,
              curveRef: payload.powerCurveRef,
              curveYearRef: payload.powerCurveYearRef,
              runCurveRef: payload.runPowerCurveRef,
              runCurveYearRef: payload.runPowerCurveYearRef,
              curveYear: payload.powerCurveYear,
              criticalPower: payload.criticalPower,
              criticalPowerYear: payload.criticalPowerYear,
              ftp: ATHLETE.ftp,
              goalFtp: ATHLETE.goalFTP,
              vt1: null,
            } satisfies DetailCtx,
            { removable: false, health: payload.health },
          ),
        ],
      ),
    ]
    if (fileData.frontmatter?.pageLayout !== 'reflection') {
      children.push(
        h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }
    node.children = children
    return true
  }

  const calcAnchor =
    blockRef && page.frontmatter?.layout === 'triathlon'
      ? CALC_ANCHOR_RE.exec(blockRef)?.[1]
      : undefined
  const calcShare = calcAnchor ? decodeCalcShare(calcAnchor) : null
  if (calcAnchor && calcShare) {
    const children: ElementContent[] = [
      h('.tri-calc-embed', { 'data-calc-hash': calcAnchor }, [
        buildTriathlonCalcCard(triathlonCardFactory, calcShare),
      ]),
    ]
    if (fileData.frontmatter?.pageLayout !== 'reflection') {
      children.push(
        h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }
    node.children = children
    return true
  }
  return false
}
