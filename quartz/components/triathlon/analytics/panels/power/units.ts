import type { PowerCurveBlock } from '../../../../../plugins/stores/analytics'
import type { TriathlonContext } from '../../../runtime/context'
import {
  buildPowerCurveGrid,
  powerCurveAxisTicks,
  powerCurveValueText,
} from '../../../../../util/triathlon-card'
import { criticalPowerSummaryParts } from '../../../../../util/triathlon-i18n'
import { powerCurveActivityLinkAttributes } from '../../../../../util/triathlon-power-activity'
import { createDomFactory, el } from '../../../runtime/dom'
import {
  bestPowerSeries,
  bestPowerSeriesLabel,
  criticalPowerForSeries,
  estimatedFtpForSeries,
} from './series'

const massKg = (power: PowerCurveBlock): number | null => {
  const kg = power.ranking.massKg
  return kg != null && Number.isFinite(kg) && kg > 0 ? kg : null
}

export const buildPowerUnits = (power: PowerCurveBlock, context: TriathlonContext): HTMLElement => {
  const group = el('div', 'tri-best-power-units', undefined, {
    role: 'group',
    'aria-label': context.formatter.text('power curve units'),
  })
  for (const [unit, label] of [
    ['watts', 'W'],
    ['watts-per-kg', 'W/kg'],
  ]) {
    const unavailable = unit === 'watts-per-kg' && massKg(power) == null
    group.appendChild(
      el('button', 'tri-best-power-unit', label, {
        type: 'button',
        'data-power-unit': unit,
        'aria-pressed': String(unit === 'watts'),
        ...(unavailable
          ? {
              disabled: '',
              title: context.formatter.text('W/kg unavailable: no body weight recorded'),
            }
          : {}),
      }),
    )
  }
  return group
}

export const buildPowerCaptions = (
  power: PowerCurveBlock,
  context: TriathlonContext,
  weightKg: number | null = null,
): HTMLElement => {
  const cap = el('div', 'tri-best-power-cap')
  const text = (watts: number): string =>
    powerCurveValueText(watts, weightKg, context.presentation.locale)
  if (power.ftp != null)
    cap.appendChild(el('span', 'tri-best-power-cap-ftp', `FTP ${text(power.ftp)}`))
  let hasCriticalPower = false
  let hasEstimatedFtp = false
  for (const { key, curve } of bestPowerSeries(power)) {
    const estimate = criticalPowerForSeries(power, key)
    if (estimate) {
      const summary = criticalPowerSummaryParts(context.presentation.locale, estimate)
      cap.appendChild(
        el(
          'span',
          'tri-best-power-cap-cp',
          `eCP ${text(estimate.criticalPowerWatts)} · ${summary.wPrime}`,
          { 'data-power-cap-series': key, ...(hasCriticalPower ? { hidden: '' } : {}) },
        ),
      )
      hasCriticalPower = true
    }
    const ftp = curve.length >= 2 ? estimatedFtpForSeries(power, key) : null
    if (!ftp) continue
    const source = ftp.anchor
    const period = bestPowerSeriesLabel(context.formatter, power, key)
    const date = source.activityDate ? ` · ${context.formatter.shortDate(source.activityDate)}` : ''
    const label = `eFTP ${text(ftp.watts)}`
    cap.appendChild(
      el('a', 'tri-best-power-cap-eftp', label, {
        'data-power-eftp-caption': key,
        'aria-label': `${label} · ${period}`,
        title: `${period} · 20m ${text(source.w)}${date}`,
        ...powerCurveActivityLinkAttributes(source),
        ...(hasEstimatedFtp ? { hidden: '' } : {}),
      }),
    )
    hasEstimatedFtp = true
  }
  if (power.goalFtp != null)
    cap.appendChild(
      el(
        'span',
        'tri-best-power-cap-goal',
        `${context.formatter.text('goal')} ${text(power.goalFtp)}`,
      ),
    )
  if (weightKg != null) {
    const { massDate, massSource } = power.ranking
    const source =
      massSource === 'garmin'
        ? 'Garmin'
        : massSource === 'apple'
          ? 'Apple Health'
          : massSource === 'tracking'
            ? context.formatter.text('tracking')
            : ''
    cap.appendChild(
      el(
        'span',
        'tri-best-power-weight',
        [`${context.formatter.number(weightKg, 2)} kg`, source, massDate ?? '']
          .filter(Boolean)
          .join(' · '),
      ),
    )
  }
  return cap
}

export const createPowerUnitView = (
  block: HTMLElement,
  graph: SVGSVGElement,
  power: PowerCurveBlock,
  context: TriathlonContext,
): { text: (watts: number) => string; select: (unit: string) => void } => {
  const mass = massKg(power)
  let weightKg: number | null = null
  const domainMax = Number(graph.dataset.powerDomainMax)
  const height = graph.viewBox.baseVal.height
  return {
    text: watts => powerCurveValueText(watts, weightKg, context.presentation.locale),
    select: unit => {
      weightKg = unit === 'watts-per-kg' ? mass : null
      block.dataset.powerUnit = weightKg == null ? 'watts' : 'watts-per-kg'
      for (const button of block.querySelectorAll<HTMLButtonElement>('.tri-best-power-unit'))
        button.setAttribute(
          'aria-pressed',
          String(button.dataset.powerUnit === block.dataset.powerUnit),
        )
      block
        .querySelector('.tri-best-power-cap')
        ?.replaceWith(buildPowerCaptions(power, context, weightKg))
      const axis = block.querySelector<HTMLElement>('.tri-best-power-curve-pane .tri-cax-yax')
      if (!axis || !Number.isFinite(domainMax) || domainMax <= 0 || height <= 0) return
      // One mass divides every curve and reference, so watt-domain geometry stays valid.
      const ticks = powerCurveAxisTicks(domainMax, weightKg, context.presentation.locale)
      const y = (watts: number): number => height - (watts / domainMax) * (height - 1)
      axis.replaceChildren(
        ...ticks.map(tick => {
          const label = el('span', 'tri-cax-yt', tick.label)
          label.style.top = `${(y(tick.watts) / height) * 100}%`
          return label
        }),
      )
      graph.querySelector('.tri-power-curve-grid--power')?.replaceWith(
        buildPowerCurveGrid(
          createDomFactory(context.presentation),
          'power',
          ticks.map(tick => y(tick.watts)),
          height,
        ),
      )
    },
  }
}
