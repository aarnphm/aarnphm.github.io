import type { Analytics } from '../../../../../plugins/stores/analytics'
import type { PowerCurvePoint } from '../../../../../plugins/stores/strava'
import type { TriathlonContext } from '../../../runtime/context'
import { criticalPowerAtDuration } from '../../../../../plugins/stores/critical-power'
import {
  nearestPowerCurvePoint,
  powerCurveFraction,
  powerCurveHoverAt,
  zoneClock,
} from '../../../../../util/triathlon-card'
import { syncPowerCurveActivityLink } from '../../../activity/power-links'
import { setText } from '../../../runtime/dom'
import { setupPowerCurveTicks } from '../../../shell/power-curve-ticks'
import { rafEvent, rectOf } from '../../scrub-primitives'
import { clampN } from '../../shared'
import { mountPowerRank } from './rank-interaction'
import {
  bestPowerSeries,
  bestPowerSeriesLabel,
  criticalPowerForSeries,
  estimatedFtpForSeries,
  powerSkillAtSeconds,
  type BestPowerSeriesKey,
} from './series'
import { createPowerUnitView } from './units'

export const mountPowerCurve = (
  panel: HTMLElement,
  data: Analytics,
  context: TriathlonContext,
): (() => void) => {
  const cleanups: (() => void)[] = []
  const powerBlock = panel.querySelector<HTMLElement>('.tri-best-power')
  const powerSvg = powerBlock?.querySelector<SVGSVGElement>('.tri-best-power-svg')
  const power = data.powerCurve
  const powerSeries = bestPowerSeries(power)
  const activePowerSeries = new Set<BestPowerSeriesKey>(
    powerSeries.filter(({ curve }) => curve.length >= 2).map(({ key }) => key),
  )
  if (powerBlock && powerSvg && activePowerSeries.size > 0) {
    cleanups.push(setupPowerCurveTicks(powerBlock))
    const units = createPowerUnitView(powerBlock, powerSvg, power, context)
    const cursor = powerSvg.querySelector<SVGLineElement>('.tri-best-power-cursor')
    const duration = powerBlock.querySelector<HTMLElement>('.tri-best-power-duration')
    const axisTicks = Array.from(
      powerBlock.querySelectorAll<HTMLButtonElement>('.tri-best-power-tick'),
    )
    const powerDom = new Map(
      powerSeries.map(({ key }) => {
        const point = powerBlock.querySelector<HTMLElement>(
          `.tri-best-power-point[data-power-series="${key}"]`,
        )
        const row = powerBlock.querySelector<HTMLAnchorElement>(
          `.tri-best-power-readout-row[data-power-series="${key}"]`,
        )
        const modelRow = powerBlock.querySelector<HTMLElement>(
          `.tri-best-power-readout-row--model[data-power-model-series="${key}"]`,
        )
        return [
          key,
          {
            point,
            row,
            value: row?.querySelector<HTMLElement>('.tri-best-power-value') ?? null,
            swatch: row?.querySelector<HTMLElement>('.tri-best-power-swatch') ?? null,
            modelRow,
            modelValue: modelRow?.querySelector<HTMLElement>('.tri-best-power-value') ?? null,
          },
        ] as const
      }),
    )
    const minSeconds = Number(powerSvg.getAttribute('aria-valuemin'))
    const maxSeconds = Number(powerSvg.getAttribute('aria-valuemax'))
    const domainMax = Number(powerSvg.dataset.powerDomainMax)
    const height = powerSvg.viewBox.baseVal.height
    const ftpToggle = powerBlock.querySelector<HTMLButtonElement>('.tri-best-power-eftp-toggle')
    let showEstimatedFtp = ftpToggle?.getAttribute('aria-pressed') === 'true'
    const syncEstimatedFtp = (): void => {
      const selected = powerSeries.find(
        ({ key }) => activePowerSeries.has(key) && estimatedFtpForSeries(power, key) != null,
      )?.key
      for (const { key } of powerSeries) {
        const enabled = activePowerSeries.has(key) && estimatedFtpForSeries(power, key) != null
        for (const element of powerBlock.querySelectorAll<SVGElement>(
          `[data-power-eftp-series="${key}"]`,
        ))
          element.toggleAttribute('hidden', !showEstimatedFtp || !enabled)
      }
      for (const caption of powerBlock.querySelectorAll<HTMLElement>('[data-power-eftp-caption]'))
        caption.hidden = !showEstimatedFtp || caption.dataset.powerEftpCaption !== selected
    }
    let selectedSeconds = Number(powerSvg.dataset.powerSelectedSeconds)
    let shownSeconds = -1
    const curveFor = (key: BestPowerSeriesKey): PowerCurvePoint[] =>
      key === 'six-weeks' ? power.sixWeeks : power.year
    const activeAnchor = (): PowerCurvePoint[] => {
      const key = activePowerSeries.has('six-weeks') ? 'six-weeks' : 'year'
      return curveFor(key)
    }
    const syncPowerCaption = (): void => {
      const captions = Array.from(
        powerBlock.querySelectorAll<HTMLElement>('[data-power-cap-series]'),
      )
      const selected = (['six-weeks', 'year'] as const).find(
        key =>
          activePowerSeries.has(key) &&
          captions.some(caption => caption.dataset.powerCapSeries === key),
      )
      for (const caption of captions)
        caption.toggleAttribute('hidden', caption.dataset.powerCapSeries !== selected)
    }
    const showSeconds = (requestedSeconds: number, commit: boolean): number | null => {
      const anchor = activeAnchor()
      const selected = powerCurveHoverAt(
        anchor,
        [],
        powerCurveFraction(requestedSeconds, anchor[0].s, anchor[anchor.length - 1].s),
      )
      if (!selected) return null
      const seconds = selected.durationS
      if (commit) {
        selectedSeconds = seconds
        powerSvg.dataset.powerSelectedSeconds = String(seconds)
        for (const tick of axisTicks)
          tick.setAttribute(
            'aria-pressed',
            String(Number(tick.dataset.powerSeconds) === selectedSeconds),
          )
      }
      if (!commit && seconds === shownSeconds) return seconds
      shownSeconds = seconds
      const xPct = powerCurveFraction(seconds, minSeconds, maxSeconds) * 100
      const powerSkill = powerSkillAtSeconds(seconds)
      cursor?.setAttribute('x1', xPct.toFixed(2))
      cursor?.setAttribute('x2', xPct.toFixed(2))
      if (duration) setText(duration, zoneClock(seconds))
      const valueText: string[] = []
      let firstVisibleModelRow = true
      for (const { key, curve } of powerSeries) {
        const enabled = activePowerSeries.has(key)
        const selectedPoint = enabled ? nearestPowerCurvePoint(curve, seconds) : null
        const watts = selectedPoint?.w ?? null
        const dom = powerDom.get(key)
        if (dom?.point) {
          dom.point.style.display = watts == null ? 'none' : ''
          if (dom.point.hidden) dom.point.hidden = false
          dom.point.dataset.powerSkill = powerSkill
          if (watts != null && domainMax > 0 && height > 0) {
            const y = height - (Math.min(domainMax, Math.max(0, watts)) / domainMax) * (height - 1)
            dom.point.style.left = `${xPct.toFixed(2)}%`
            dom.point.style.top = `${((y / height) * 100).toFixed(2)}%`
          }
        }
        if (dom?.row) {
          if (dom.row.hidden !== !enabled) dom.row.hidden = !enabled
          syncPowerCurveActivityLink(dom.row, selectedPoint)
          if (dom.value) setText(dom.value, watts == null ? '—' : units.text(watts))
          if (dom.swatch) dom.swatch.dataset.powerSkill = powerSkill
        }
        if (watts != null)
          valueText.push(
            `${bestPowerSeriesLabel(context.formatter, power, key)} ${units.text(watts)}`,
          )
        const estimate = enabled ? criticalPowerForSeries(power, key) : null
        const modelWatts =
          estimate && seconds >= 180 && seconds <= 720
            ? Math.round(criticalPowerAtDuration(estimate, seconds))
            : null
        if (dom?.modelRow) {
          if (dom.modelRow.hidden) dom.modelRow.hidden = false
          dom.modelRow.style.display = modelWatts == null ? 'none' : ''
          if (modelWatts != null) {
            dom.modelRow.style.marginTop = firstVisibleModelRow ? '0.28rem' : '0px'
            firstVisibleModelRow = false
          }
          if (dom.modelValue)
            setText(dom.modelValue, modelWatts == null ? '—' : units.text(modelWatts))
        }
        if (modelWatts != null)
          valueText.push(
            `${bestPowerSeriesLabel(context.formatter, power, key)} ${context.formatter.text('critical power model')} ${units.text(modelWatts)}`,
          )
      }
      powerSvg.setAttribute('aria-valuenow', String(seconds))
      powerSvg.setAttribute('aria-valuetext', `${zoneClock(seconds)}; ${valueText.join('; ')}`)
      rank.sync(seconds)
      return seconds
    }
    const showFraction = (fraction: number, commit: boolean): number | null => {
      const seconds = Math.exp(
        Math.log(minSeconds) +
          clampN(fraction, 0, 1) * (Math.log(maxSeconds) - Math.log(minSeconds)),
      )
      return showSeconds(seconds, commit)
    }
    const rafPowerMove = rafEvent((event: PointerEvent): void => {
      const rect = rectOf(powerSvg)
      if (rect.width <= 0) return
      showFraction((event.clientX - rect.left) / rect.width, false)
    })
    const onPowerDown = (event: PointerEvent): void => {
      const rect = rectOf(powerSvg)
      if (rect.width <= 0) return
      showFraction((event.clientX - rect.left) / rect.width, true)
      powerSvg.focus({ preventScroll: true })
    }
    const onPowerLeave = (): void => {
      rafPowerMove.cancel()
      showSeconds(selectedSeconds, false)
    }
    const onPowerKey = (event: KeyboardEvent): void => {
      const anchor = activeAnchor()
      const selected = powerCurveHoverAt(
        anchor,
        [],
        powerCurveFraction(selectedSeconds, anchor[0].s, anchor[anchor.length - 1].s),
      )
      if (!selected) return
      let nextIndex: number | null = null
      if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') nextIndex = selected.index - 1
      else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') nextIndex = selected.index + 1
      else if (event.key === 'Home') nextIndex = 0
      else if (event.key === 'End') nextIndex = anchor.length - 1
      else if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        powerSvg.blur()
        return
      }
      if (nextIndex == null) return
      event.preventDefault()
      event.stopPropagation()
      showSeconds(anchor[clampN(nextIndex, 0, anchor.length - 1)].s, true)
    }
    const onPowerClick = (event: MouseEvent): void => {
      if (!(event.target instanceof Element)) return
      const unitButton = event.target.closest<HTMLButtonElement>('.tri-best-power-unit')
      if (unitButton && !unitButton.disabled) {
        units.select(unitButton.dataset.powerUnit ?? 'watts')
        syncPowerCaption()
        syncEstimatedFtp()
        shownSeconds = -1
        showSeconds(selectedSeconds, false)
        return
      }
      const tick = event.target.closest<HTMLButtonElement>('.tri-best-power-tick')
      if (tick) {
        const seconds = Number(tick.dataset.powerSeconds)
        if (seconds > 0) {
          showSeconds(seconds, true)
          powerSvg.focus({ preventScroll: true })
        }
        return
      }
      const axis = event.target.closest<HTMLElement>('.tri-cax-xax')
      if (axis && powerBlock.contains(axis)) {
        const rect = rectOf(axis)
        if (rect.width > 0) {
          showFraction((event.clientX - rect.left) / rect.width, true)
          powerSvg.focus({ preventScroll: true })
        }
        return
      }
      if (
        event.target.closest('.tri-best-power-eftp-toggle') === ftpToggle &&
        ftpToggle &&
        !ftpToggle.disabled
      ) {
        showEstimatedFtp = !showEstimatedFtp
        ftpToggle.setAttribute('aria-pressed', String(showEstimatedFtp))
        syncEstimatedFtp()
        return
      }
      const button = event.target.closest<HTMLButtonElement>(
        '.tri-best-power-toggle[data-power-series]',
      )
      if (!button || button.disabled) return
      const key: BestPowerSeriesKey = button.dataset.powerSeries === 'year' ? 'year' : 'six-weeks'
      const enabled = activePowerSeries.has(key)
      if (enabled && activePowerSeries.size === 1) return
      if (enabled) activePowerSeries.delete(key)
      else activePowerSeries.add(key)
      button.setAttribute('aria-pressed', String(!enabled))
      const lines = powerSvg.querySelectorAll<SVGElement>(
        `.tri-best-power-line[data-power-series="${key}"]`,
      )
      for (const line of lines) line.toggleAttribute('hidden', enabled)
      for (const element of powerBlock.querySelectorAll<HTMLElement | SVGElement>(
        `[data-power-model-series="${key}"]`,
      ))
        element.toggleAttribute('hidden', enabled)
      for (const element of powerBlock.querySelectorAll<HTMLElement | SVGElement>(
        `[data-power-radar-series="${key}"]`,
      ))
        element.toggleAttribute('hidden', enabled)
      shownSeconds = -1
      rank.reset()
      syncPowerCaption()
      syncEstimatedFtp()
      showSeconds(selectedSeconds, false)
    }
    powerSvg.addEventListener('pointermove', rafPowerMove.onEvent)
    powerSvg.addEventListener('pointerdown', onPowerDown)
    powerSvg.addEventListener('pointerleave', onPowerLeave)
    powerSvg.addEventListener('pointercancel', onPowerLeave)
    powerSvg.addEventListener('keydown', onPowerKey)
    powerBlock.addEventListener('click', onPowerClick)
    const rank = mountPowerRank(powerBlock, power, context, activePowerSeries, showSeconds, () =>
      showSeconds(selectedSeconds, false),
    )
    syncPowerCaption()
    syncEstimatedFtp()
    showSeconds(selectedSeconds, true)
    cleanups.push(() => {
      rafPowerMove.cancel()
      rank.cleanup()
      powerSvg.removeEventListener('pointermove', rafPowerMove.onEvent)
      powerSvg.removeEventListener('pointerdown', onPowerDown)
      powerSvg.removeEventListener('pointerleave', onPowerLeave)
      powerSvg.removeEventListener('pointercancel', onPowerLeave)
      powerSvg.removeEventListener('keydown', onPowerKey)
      powerBlock.removeEventListener('click', onPowerClick)
    })
  }

  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}
