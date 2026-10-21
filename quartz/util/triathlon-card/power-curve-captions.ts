import type { CriticalPowerEstimate } from '../../plugins/stores/critical-power'
import type { TriNodeFactory } from './construction-contracts'
import {
  criticalPowerEvidenceText,
  criticalPowerSummaryParts,
  criticalPowerSummaryText,
  triText,
} from '../triathlon-i18n'
import { powerCurveActivityLinkAttributes } from '../triathlon-power-activity'
import { dlabel } from './metric-text'
import { powerCurveValueText, type PowerCurveRange } from './power-curve-data'

export const powerCurveValueNode = <N>(f: TriNodeFactory<N>, watts: number, compact = true): N =>
  f.el(
    'span',
    'tri-curve-power-value',
    powerCurveValueText(watts, null, f.presentation.locale, compact),
    { 'data-curve-watts': String(watts), 'data-curve-compact': String(compact) },
  )

export const buildCriticalPowerAnchorLinks = <N>(
  f: TriNodeFactory<N>,
  estimate: CriticalPowerEstimate,
  range: PowerCurveRange,
  selected: boolean,
  excludeActivityId?: number | string,
): N => {
  const links = f.el('span', 'tri-critical-power-anchors', undefined, {
    'data-critical-power-range': range,
    ...(selected ? {} : { hidden: '' }),
  })
  for (const anchor of estimate.anchors) {
    const powerPoint = {
      s: anchor.durationS,
      w: anchor.meanPowerWatts,
      activityId: anchor.activityId,
      activityDate: anchor.activityDate,
    }
    const link = f.el(
      'a',
      'tri-critical-power-anchor',
      undefined,
      powerCurveActivityLinkAttributes(powerPoint, excludeActivityId),
    )
    f.add(
      link,
      f.el('span', 'tri-critical-power-anchor-duration', dlabel(anchor.durationS)),
      f.el('span', 'tri-critical-power-anchor-separator', '·', { 'aria-hidden': 'true' }),
      f.el('span', 'tri-critical-power-anchor-date', anchor.activityDate),
      f.el('span', 'tri-critical-power-anchor-separator', '·', { 'aria-hidden': 'true' }),
      f.el(
        'span',
        'tri-critical-power-anchor-power',
        `${anchor.meanPowerWatts.toLocaleString(f.presentation.locale === 'fr' ? 'fr-CA' : 'en-US', { maximumFractionDigits: 1 })}W`,
      ),
    )
    if (anchor.mechanics)
      f.add(
        link,
        f.el(
          'span',
          'tri-critical-power-mechanics',
          `${anchor.mechanics.averageCadenceRpm.toFixed(0)} rpm · ${anchor.mechanics.averageTorqueNm.toFixed(1)} N·m · ${(anchor.mechanics.coverage * 100).toFixed(0)}% ${triText(f.presentation.locale, 'coverage')}`,
          {
            title: `Wahoo · ${(anchor.mechanics.coverage * 100).toFixed(0)}% ${triText(f.presentation.locale, 'coverage')}`,
          },
        ),
      )
    f.add(links, link)
  }
  return links
}

export const addPowerCurveThresholdCaption = <N>(
  f: TriNodeFactory<N>,
  caption: N,
  estimates: ReadonlyArray<readonly [PowerCurveRange, CriticalPowerEstimate | null]>,
  selectedRange: PowerCurveRange,
  ftp: number | null,
  goalFtp: number | null,
  excludeActivityId?: number | string,
): void => {
  const thresholds = f.el('span', 'tri-curve-thresholds')
  const anchorRows: N[] = []
  let hasThreshold = false
  for (const [range, estimate] of estimates) {
    if (!estimate) continue
    f.add(
      thresholds,
      f.el(
        'span',
        'tri-ana-k tri-curve-cp-k',
        criticalPowerSummaryText(f.presentation.locale, estimate),
        {
          'data-critical-power-range': range,
          'data-gloss': '',
          'data-gloss-def': criticalPowerEvidenceText(f.presentation.locale, estimate),
          tabindex: '0',
          ...(selectedRange === range ? {} : { hidden: '' }),
        },
      ),
    )
    anchorRows.push(
      buildCriticalPowerAnchorLinks(f, estimate, range, selectedRange === range, excludeActivityId),
    )
    hasThreshold = true
  }
  if (ftp != null) {
    f.add(thresholds, f.el('span', 'tri-ana-k tri-curve-ftp-k', `FTP ${ftp}W`))
    hasThreshold = true
  }
  if (goalFtp != null) {
    f.add(thresholds, f.el('span', 'tri-ana-k tri-curve-goal-k', `goal ${goalFtp}W`))
    hasThreshold = true
  }
  if (hasThreshold) f.add(caption, thresholds, ...anchorRows)
}

export const addActivityCriticalPowerCaption = <N>(
  f: TriNodeFactory<N>,
  caption: N,
  estimate: CriticalPowerEstimate,
): N => {
  const thresholds = f.el('span', 'tri-curve-thresholds')
  const summary = f.el('span', 'tri-ana-k tri-curve-cp-k tri-curve-cp-k--ride', undefined, {
    'data-gloss': '',
    'data-gloss-def': criticalPowerEvidenceText(f.presentation.locale, estimate),
    tabindex: '0',
  })
  f.add(
    summary,
    f.el('span', undefined, triText(f.presentation.locale, 'this ride'), {
      'data-i18n': 'this ride',
    }),
    f.el('span', undefined, ' · eCP '),
    powerCurveValueNode(f, estimate.criticalPowerWatts, false),
    f.el(
      'span',
      undefined,
      ` · ${criticalPowerSummaryParts(f.presentation.locale, estimate).wPrime}`,
    ),
  )
  f.add(thresholds, summary)
  f.add(caption, thresholds)
  return summary
}

export const buildPowerCurveRanges = <N>(
  f: TriNodeFactory<N>,
  selected: PowerCurveRange,
  year: number | null,
  sixWeeksAvailable: boolean,
  yearAvailable: boolean,
): N | null => {
  if (year == null || !yearAvailable) return null
  const ranges = f.el('div', 'tri-curve-ranges', undefined, {
    role: 'group',
    'aria-label': 'comparison range',
    'data-i18n-aria-label': 'comparison range',
  })
  const sixWeekAttrs: Record<string, string> = {
    type: 'button',
    'data-curve-range': 'six-weeks',
    'aria-pressed': String(selected === 'six-weeks'),
    'data-i18n': '6 weeks',
  }
  if (!sixWeeksAvailable) sixWeekAttrs.disabled = ''
  const yearButton = f.el('button', 'tri-curve-range', undefined, {
    type: 'button',
    'data-curve-range': 'year',
    'aria-pressed': String(selected === 'year'),
  })
  f.add(
    yearButton,
    f.el('span', undefined, 'all of', { 'data-i18n': 'all of' }),
    f.el('span', undefined, ` ${year}`),
  )
  f.add(ranges, f.el('button', 'tri-curve-range', '6 weeks', sixWeekAttrs), yearButton)
  return ranges
}
