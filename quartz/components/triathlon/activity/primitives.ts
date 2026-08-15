import type {
  ActivityHealth,
  ActivityKind,
  StravaActivityDetail,
} from '../../../plugins/stores/strava'
import type {
  ActivityTraceDomain,
  ActivityTraceReference,
  DetailCtx,
} from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { ActivityAnalysisRange } from './analysis'
import {
  buildElevation as buildElevationNode,
  buildHeartRateTrace as buildHeartRateTraceNode,
  buildHrZones as buildHrZonesNode,
  buildIcon as buildIconNode,
  buildPool as buildPoolNode,
  buildPowerCurve as buildPowerCurveNode,
  buildPowerHist as buildPowerHistNode,
  buildPowerZones as buildPowerZonesNode,
  buildRecovery as buildRecoveryNode,
  buildTrace as buildTraceNode,
  statRow as buildStatRowNode,
  zoneDuo as buildZoneDuoNode,
} from '../../../util/triathlon-card'
import { applyI18n, createDomFactory } from '../runtime/dom'

const htmlElement = (node: HTMLElement | SVGElement): HTMLElement => {
  if (node instanceof HTMLElement) return node
  throw new TypeError('Expected an HTML activity node')
}

const svgElement = (node: HTMLElement | SVGElement): SVGElement => {
  if (node instanceof SVGElement) return node
  throw new TypeError('Expected an SVG activity node')
}

export const buildIcon = (presentation: TriathlonPresentation, sport: ActivityKind): SVGElement =>
  svgElement(buildIconNode(createDomFactory(presentation), sport))

export const buildElevation = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  graphDomain?: ActivityAnalysisRange | null,
): HTMLElement =>
  htmlElement(buildElevationNode(createDomFactory(presentation), activity, null, graphDomain))

export const buildPool = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
): HTMLElement => htmlElement(buildPoolNode(createDomFactory(presentation), activity))

export const buildTrace = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  pick: (point: StravaActivityDetail['route'][number], index: number) => number | null,
  title: string,
  cap: (maximum: number) => string,
  tick: (value: number) => string,
  graphDomain?: ActivityAnalysisRange | null,
  domain?: ActivityTraceDomain,
  reference?: ActivityTraceReference | null,
): HTMLElement =>
  htmlElement(
    buildTraceNode(
      createDomFactory(presentation),
      activity,
      pick,
      title,
      cap,
      tick,
      domain,
      null,
      graphDomain,
      undefined,
      reference,
    ),
  )

export const buildHeartRateTrace = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  graphDomain?: ActivityAnalysisRange | null,
): HTMLElement =>
  htmlElement(buildHeartRateTraceNode(createDomFactory(presentation), activity, null, graphDomain))

export const zoneDuo = (
  presentation: TriathlonPresentation,
  first: HTMLElement | null,
  second: HTMLElement | null,
): HTMLElement | null => {
  const node = buildZoneDuoNode(createDomFactory(presentation), first, second)
  return node ? htmlElement(node) : null
}

export const statRow = (
  presentation: TriathlonPresentation,
  label: string,
  value: string,
  attrs?: Record<string, string>,
): HTMLElement => htmlElement(buildStatRowNode(createDomFactory(presentation), label, value, attrs))

export const buildRecovery = (
  presentation: TriathlonPresentation,
  health: ActivityHealth,
): HTMLElement | null => {
  const node = buildRecoveryNode(createDomFactory(presentation), health)
  return node ? htmlElement(node) : null
}

const translated = (
  presentation: TriathlonPresentation,
  node: HTMLElement | SVGElement | null,
): HTMLElement | null => {
  if (!node) return null
  applyI18n(node, presentation)
  return htmlElement(node)
}

export const buildHrZones = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  detailContext: DetailCtx,
): HTMLElement | null =>
  translated(
    presentation,
    buildHrZonesNode(createDomFactory(presentation), activity, detailContext),
  )

export const buildPowerZones = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  detailContext: DetailCtx,
): HTMLElement | null =>
  translated(
    presentation,
    buildPowerZonesNode(createDomFactory(presentation), activity, detailContext),
  )

export const buildPowerHist = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
): HTMLElement | null =>
  translated(presentation, buildPowerHistNode(createDomFactory(presentation), activity))

export const buildPowerCurve = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  detailContext: DetailCtx,
): HTMLElement | null =>
  translated(
    presentation,
    buildPowerCurveNode(createDomFactory(presentation), activity, detailContext),
  )
