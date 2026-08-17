import type { Analytics } from '../../../plugins/stores/analytics'
import type { PowerRankInterval } from '../../../plugins/stores/power-rank'
import type { PowerCurvePoint } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import type { WkKind } from './panels/performance'
import type { BestPowerSeriesKey } from './panels/power'
import { criticalPowerAtDuration } from '../../../plugins/stores/critical-power'
import { nearestPowerCurvePoint } from '../../../util/triathlon-card'
import { powerCurveFraction } from '../../../util/triathlon-card'
import { powerCurveHoverAt } from '../../../util/triathlon-card'
import { zoneClock } from '../../../util/triathlon-card'
import { weeklyChartIndex } from '../../../util/weekly-target-range'
import { weeklyChartX } from '../../../util/weekly-target-range'
import { syncPowerCurveActivityLink } from '../activity/power-links'
import { el } from '../runtime/dom'
import { setMath } from '../runtime/dom'
import { groupBodyByDay } from './panels/body'
import { renderWkDetail } from './panels/performance'
import { wkTrendRows } from './panels/performance'
import { bestPowerSeries } from './panels/power'
import { bestPowerSeriesLabel } from './panels/power'
import { criticalPowerForSeries } from './panels/power'
import { powerRankEffortLabel } from './panels/power'
import { powerRankRangeRows } from './panels/power'
import { powerRankProgressLabel } from './panels/power'
import { powerRankProgressNextLabel } from './panels/power'
import { powerSkillAtSeconds } from './panels/power'
import { scrubBind } from './scrub-primitives'
import { ANA_W } from './shared'
import { clampN } from './shared'
import { hms } from './shared'
import { signed } from './shared'
import { weightUnitLabel } from './shared'
import { wFmt } from './shared'
import { wNum } from './shared'
import { wSigned } from './shared'

export type PrimaryScrubPanel =
  | 'body'
  | 'dexa'
  | 'recovery'
  | 'sleep'
  | 'vo2max'
  | 'power'
  | 'weekly'
  | 'effort'
  | 'heat'

const powerRangeTipContent = (
  interval: PowerRankInterval,
  key: BestPowerSeriesKey,
): HTMLElement[] => {
  const effort = interval.efforts[key]
  const head = el('div', 'tri-power-range-tip-head')
  head.append(
    el('strong', 'tri-power-range-tip-title', `${zoneClock(interval.durationS)} power`),
    el(
      'span',
      'tri-power-range-tip-best',
      effort == null
        ? 'no data'
        : `${effort.watts.toLocaleString()} W · ${effort.wattsPerKg.toFixed(2)} W/kg`,
    ),
  )
  const table = el('table', 'tri-power-range-table')
  const tableHead = el('thead')
  const headerRow = el('tr')
  for (const label of ['levels', '%', 'power', 'your best'])
    headerRow.appendChild(el('th', undefined, label, { scope: 'col' }))
  tableHead.appendChild(headerRow)
  const tableBody = el('tbody')
  for (const row of powerRankRangeRows(interval, key)) {
    const tableRow = el(
      'tr',
      row.current ? 'tri-power-range-row tri-power-range-row--current' : 'tri-power-range-row',
      undefined,
      row.current ? { 'aria-current': 'true' } : undefined,
    )
    tableRow.append(
      el('th', 'tri-power-range-level', `${row.level} ${row.label}`, { scope: 'row' }),
      el('td', undefined, `${row.percentile}%`),
      el('td', undefined, `${row.thresholdWatts.toLocaleString()} W`),
      el(
        'td',
        'tri-power-range-result',
        row.bestWatts != null
          ? `${row.bestWatts.toLocaleString()} W`
          : row.gapWatts != null
            ? `+${row.gapWatts.toLocaleString()} W`
            : '',
      ),
    )
    tableBody.appendChild(tableRow)
  }
  table.append(tableHead, tableBody)
  return [head, table]
}

export const mountPrimaryPanel = (
  kind: PrimaryScrubPanel,
  panel: HTMLElement,
  data: Analytics,
  context: TriathlonContext,
): (() => void) => {
  const cleanups: (() => void)[] = []
  const bind = (
    blockSel: string,
    svgSel: string,
    count: number,
    vbW: number,
    textOf: (i: number) => string,
  ) => {
    const block = panel.querySelector<HTMLElement>(blockSel)
    const svgEl = block?.querySelector<SVGElement>(svgSel)
    const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
    const readout = block?.querySelector<HTMLElement>('.tri-chart-readout')
    if (block && svgEl && cursor && readout)
      cleanups.push(scrubBind(block, svgEl, cursor, readout, count, vbW, textOf))
  }

  if (kind === 'recovery') {
    const rec = data.recovery.series
    bind('.tri-ana-recovery', '.tri-rec-svg', rec.length, ANA_W, i => {
      const d = rec[i]
      const z = d.hrvZ != null ? ` $${signed(d.hrvZ)}\\sigma$` : ''
      return `${d.date} · HRV ${d.hrv ?? '—'}${z} · RHR ${d.rhr ?? '—'} · rdy ${d.readiness ?? '—'}`
    })
  }

  if (kind === 'sleep') {
    const sleepView = data.recovery.series
    bind('.tri-ana-sleep', '.tri-sleep-svg', sleepView.length, sleepView.length, i => {
      const d = sleepView[i]
      const debt = d.sleepDebtS != null ? `${(d.sleepDebtS / 3600).toFixed(1)}h` : '—'
      return `${d.date} · ${d.sleepS != null ? hms(d.sleepS) : '—'} · score ${d.sleepScore ?? '—'} · debt ${debt}`
    })
  }

  if (kind === 'vo2max') {
    const trend = data.engine.vo2max.trend
    bind('.tri-engine-vo2', '.tri-engine-vo2-spark', trend.length, ANA_W, i => {
      const p = trend[i]
      const src = p.method === 'bike' ? `bike (${context.formatter.text('projected')})` : p.method
      return `${p.weekStart} · ${p.vo2max.toFixed(1)} ml/kg/min · ${src}`
    })
  }

  const powerBlock = kind === 'power' ? panel.querySelector<HTMLElement>('.tri-best-power') : null
  const powerSvg = powerBlock?.querySelector<SVGSVGElement>('.tri-best-power-svg')
  const power = data.powerCurve
  const powerSeries = bestPowerSeries(power)
  const activePowerSeries = new Set<BestPowerSeriesKey>(
    powerSeries.filter(({ curve }) => curve.length >= 2).map(({ key }) => key),
  )
  if (powerBlock && powerSvg && activePowerSeries.size > 0) {
    const cursor = powerSvg.querySelector<SVGLineElement>('.tri-best-power-cursor')
    const duration = powerBlock.querySelector<HTMLElement>('.tri-best-power-duration')
    const rankBlock = powerBlock.querySelector<HTMLElement>('.tri-power-radar')
    const rankSvg = rankBlock?.querySelector<SVGSVGElement>('.tri-power-radar-svg')
    const rankIntervals = power.ranking.intervals
    const rankDuration = rankBlock?.querySelector<HTMLElement>('.tri-power-radar-duration')
    const rankLabels = rankBlock
      ? Array.from(rankBlock.querySelectorAll<SVGElement>('.tri-power-radar-label'))
      : []
    const axisTicks = Array.from(
      powerBlock.querySelectorAll<HTMLButtonElement>('.tri-best-power-tick'),
    )
    const progressTrack = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-track]')
    const progressFill = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-fill]')
    const minSeconds = Number(powerSvg.getAttribute('aria-valuemin'))
    const maxSeconds = Number(powerSvg.getAttribute('aria-valuemax'))
    const domainMax = Number(powerSvg.dataset.powerDomainMax)
    const height = powerSvg.viewBox.baseVal.height
    document.body.querySelector('.tri-power-range-tip')?.remove()
    const powerRangeTip =
      rankIntervals.length === 0
        ? null
        : el('div', 'tri-gloss tri-power-range-tip', undefined, {
            role: 'tooltip',
            'aria-hidden': 'true',
          })
    if (powerRangeTip) document.body.appendChild(powerRangeTip)
    let selectedSeconds = Number(powerSvg.dataset.powerSelectedSeconds)
    const curveFor = (key: BestPowerSeriesKey): PowerCurvePoint[] =>
      key === 'six-weeks' ? power.sixWeeks : power.year
    const activeAnchor = (): PowerCurvePoint[] => {
      const key = activePowerSeries.has('six-weeks') ? 'six-weeks' : 'year'
      return curveFor(key)
    }
    const rankIndexAtSeconds = (seconds: number): number =>
      rankIntervals.reduce(
        (best, interval, index) =>
          Math.abs(Math.log(interval.durationS) - Math.log(seconds)) <
          Math.abs(Math.log(rankIntervals[best].durationS) - Math.log(seconds))
            ? index
            : best,
        0,
      )
    const syncPowerRank = (seconds: number): void => {
      if (!rankSvg || rankIntervals.length === 0) return
      const index = rankIndexAtSeconds(seconds)
      const interval = rankIntervals[index]
      rankSvg.dataset.powerRankIndex = String(index)
      rankSvg.setAttribute('aria-valuenow', String(interval.durationS))
      for (const label of rankLabels)
        label.classList.toggle(
          'tri-power-radar-label--active',
          Number(label.dataset.powerRankIndex) === index,
        )
      if (rankDuration)
        rankDuration.textContent = `${zoneClock(interval.durationS)} · ${interval.skill}`
      const valueText: string[] = []
      for (const { key } of powerSeries) {
        const row = rankBlock?.querySelector<HTMLElement>(
          `.tri-power-radar-readout-row[data-power-rank-series="${key}"]`,
        )
        const effort = activePowerSeries.has(key) ? interval.efforts[key] : null
        if (row) {
          row.hidden = !activePowerSeries.has(key)
          const value = row.querySelector<HTMLElement>('.tri-power-radar-readout-value')
          if (value) value.textContent = powerRankEffortLabel(effort)
          const swatch = row.querySelector<HTMLElement>('.tri-best-power-swatch')
          if (swatch) swatch.dataset.powerSkill = interval.skill
        }
        if (effort)
          valueText.push(
            `${bestPowerSeriesLabel(context.formatter, power, key)} ${powerRankEffortLabel(effort)}`,
          )
      }
      const primaryKey: BestPowerSeriesKey = activePowerSeries.has('six-weeks')
        ? 'six-weeks'
        : 'year'
      const primaryEffort = interval.efforts[primaryKey]
      const progressLabel = powerBlock.querySelector<HTMLElement>(
        '[data-power-rank-progress-label]',
      )
      const progressNext = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-next]')
      if (progressTrack) progressTrack.dataset.powerSkill = interval.skill
      if (progressFill) {
        progressFill.style.setProperty(
          '--tri-power-rank-progress',
          `${primaryEffort?.percentile ?? 0}%`,
        )
        progressFill.setAttribute('aria-valuenow', String(primaryEffort?.percentile ?? 0))
        progressFill.setAttribute('aria-valuetext', powerRankProgressLabel(primaryEffort))
        progressFill.setAttribute(
          'aria-label',
          `${bestPowerSeriesLabel(context.formatter, power, primaryKey)} power rank`,
        )
      }
      if (progressLabel) progressLabel.textContent = powerRankProgressLabel(primaryEffort)
      if (progressNext) progressNext.textContent = powerRankProgressNextLabel(primaryEffort)
      if (powerRangeTip) {
        powerRangeTip.dataset.powerSkill = interval.skill
        if (
          powerRangeTip.dataset.powerRankIndex !== String(index) ||
          powerRangeTip.dataset.powerSeries !== primaryKey
        ) {
          powerRangeTip.dataset.powerRankIndex = String(index)
          powerRangeTip.dataset.powerSeries = primaryKey
          powerRangeTip.replaceChildren(...powerRangeTipContent(interval, primaryKey))
        }
      }
      rankSvg.setAttribute(
        'aria-valuetext',
        `${zoneClock(interval.durationS)} ${interval.skill}; ${valueText.join('; ')}`,
      )
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
      const xPct = powerCurveFraction(seconds, minSeconds, maxSeconds) * 100
      const powerSkill = powerSkillAtSeconds(seconds)
      cursor?.setAttribute('x1', xPct.toFixed(2))
      cursor?.setAttribute('x2', xPct.toFixed(2))
      if (duration) duration.textContent = zoneClock(seconds)
      const valueText: string[] = []
      for (const { key, curve } of powerSeries) {
        const enabled = activePowerSeries.has(key)
        const selectedPoint = enabled ? nearestPowerCurvePoint(curve, seconds) : null
        const watts = selectedPoint?.w ?? null
        const point = powerBlock.querySelector<HTMLElement>(
          `.tri-best-power-point[data-power-series="${key}"]`,
        )
        const row = powerBlock.querySelector<HTMLAnchorElement>(
          `.tri-best-power-readout-row[data-power-series="${key}"]`,
        )
        if (point) {
          point.hidden = watts == null
          point.dataset.powerSkill = powerSkill
          if (watts != null && domainMax > 0 && height > 0) {
            const y = height - (Math.min(domainMax, Math.max(0, watts)) / domainMax) * (height - 1)
            point.style.left = `${xPct.toFixed(2)}%`
            point.style.top = `${((y / height) * 100).toFixed(2)}%`
          }
        }
        if (row) {
          row.hidden = !enabled
          syncPowerCurveActivityLink(row, selectedPoint)
          const value = row.querySelector<HTMLElement>('.tri-best-power-value')
          if (value) value.textContent = watts == null ? '—' : `${watts.toLocaleString()} W`
          const swatch = row.querySelector<HTMLElement>('.tri-best-power-swatch')
          if (swatch) swatch.dataset.powerSkill = powerSkill
        }
        if (watts != null)
          valueText.push(
            `${bestPowerSeriesLabel(context.formatter, power, key)} ${watts.toLocaleString()} W`,
          )
        const estimate = enabled ? criticalPowerForSeries(power, key) : null
        const modelWatts =
          estimate && seconds >= 180 && seconds <= 720
            ? Math.round(criticalPowerAtDuration(estimate, seconds))
            : null
        const modelRow = powerBlock.querySelector<HTMLElement>(
          `.tri-best-power-readout-row--model[data-power-model-series="${key}"]`,
        )
        if (modelRow) {
          modelRow.hidden = modelWatts == null
          const value = modelRow.querySelector<HTMLElement>('.tri-best-power-value')
          if (value)
            value.textContent = modelWatts == null ? '—' : `${modelWatts.toLocaleString()} W`
        }
        if (modelWatts != null)
          valueText.push(
            `${bestPowerSeriesLabel(context.formatter, power, key)} ${context.formatter.text('critical power model')} ${modelWatts.toLocaleString()} W`,
          )
      }
      powerSvg.setAttribute('aria-valuenow', String(seconds))
      powerSvg.setAttribute('aria-valuetext', `${zoneClock(seconds)}; ${valueText.join('; ')}`)
      syncPowerRank(seconds)
      return seconds
    }
    const showFraction = (fraction: number, commit: boolean): number | null => {
      const seconds = Math.exp(
        Math.log(minSeconds) +
          clampN(fraction, 0, 1) * (Math.log(maxSeconds) - Math.log(minSeconds)),
      )
      return showSeconds(seconds, commit)
    }
    const hidePowerRangeTip = (): void => {
      powerRangeTip?.classList.remove('tri-gloss--on')
      powerRangeTip?.setAttribute('aria-hidden', 'true')
    }
    const placePowerRangeTip = (clientX: number, clientY: number): void => {
      if (!powerRangeTip) return
      const rect = powerRangeTip.getBoundingClientRect()
      const gap = 14
      const margin = 8
      const preferredLeft = clientX + gap
      const left =
        preferredLeft + rect.width <= window.innerWidth - margin
          ? preferredLeft
          : clientX - gap - rect.width
      const maxTop = Math.max(margin, window.innerHeight - rect.height - margin)
      const top = clampN(clientY - rect.height / 2, margin, maxTop)
      powerRangeTip.style.left = `${Math.max(margin, left).toFixed(0)}px`
      powerRangeTip.style.top = `${top.toFixed(0)}px`
    }
    const onPowerMove = (event: PointerEvent): void => {
      const rect = powerSvg.getBoundingClientRect()
      if (rect.width <= 0) return
      showFraction((event.clientX - rect.left) / rect.width, false)
    }
    const showPowerRangeTip = (clientX: number, clientY: number): void => {
      if (!powerRangeTip) return
      powerRangeTip.classList.add('tri-gloss--on')
      powerRangeTip.setAttribute('aria-hidden', 'false')
      placePowerRangeTip(clientX, clientY)
    }
    const onPowerDown = (event: PointerEvent): void => {
      const rect = powerSvg.getBoundingClientRect()
      if (rect.width <= 0) return
      showFraction((event.clientX - rect.left) / rect.width, true)
      powerSvg.focus({ preventScroll: true })
    }
    const onPowerLeave = (): void => {
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
        const rect = axis.getBoundingClientRect()
        if (rect.width > 0) {
          showFraction((event.clientX - rect.left) / rect.width, true)
          powerSvg.focus({ preventScroll: true })
        }
        return
      }
      const button = event.target.closest<HTMLButtonElement>('.tri-best-power-toggle')
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
      syncPowerCaption()
      showSeconds(selectedSeconds, false)
    }
    powerSvg.addEventListener('pointermove', onPowerMove)
    powerSvg.addEventListener('pointerdown', onPowerDown)
    powerSvg.addEventListener('pointerleave', onPowerLeave)
    powerSvg.addEventListener('pointercancel', onPowerLeave)
    powerSvg.addEventListener('keydown', onPowerKey)
    powerBlock.addEventListener('click', onPowerClick)
    let onRankMove: ((event: MouseEvent) => void) | null = null
    let onRankLeave: (() => void) | null = null
    let onRankDown: ((event: PointerEvent) => void) | null = null
    let onRankKey: ((event: KeyboardEvent) => void) | null = null
    if (rankSvg && rankIntervals.length > 0) {
      const indexAt = (clientX: number, clientY: number): number => {
        const rect = rankSvg.getBoundingClientRect()
        const angle =
          (Math.atan2(
            clientY - (rect.top + rect.height / 2),
            clientX - (rect.left + rect.width / 2),
          ) *
            180) /
          Math.PI
        return (
          ((Math.round(((angle + 90) / 360) * rankIntervals.length) % rankIntervals.length) +
            rankIntervals.length) %
          rankIntervals.length
        )
      }
      onRankMove = event => {
        const index = indexAt(event.clientX, event.clientY)
        const interval = rankIntervals[index]
        showSeconds(interval.durationS, false)
        showPowerRangeTip(event.clientX, event.clientY)
      }
      onRankLeave = () => {
        hidePowerRangeTip()
        showSeconds(selectedSeconds, false)
      }
      onRankDown = event => {
        const index = indexAt(event.clientX, event.clientY)
        showSeconds(rankIntervals[index].durationS, true)
      }
      onRankKey = event => {
        let index = Number(rankSvg.dataset.powerRankIndex)
        if (!Number.isInteger(index)) index = 0
        if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') index--
        else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') index++
        else if (event.key === 'Home') index = 0
        else if (event.key === 'End') index = rankIntervals.length - 1
        else if (event.key === 'Escape') {
          event.preventDefault()
          event.stopPropagation()
          rankSvg.blur()
          return
        } else return
        event.preventDefault()
        event.stopPropagation()
        showSeconds(rankIntervals[clampN(index, 0, rankIntervals.length - 1)].durationS, true)
      }
      rankSvg.addEventListener('mousemove', onRankMove)
      rankSvg.addEventListener('mouseleave', onRankLeave)
      rankSvg.addEventListener('pointerdown', onRankDown)
      rankSvg.addEventListener('keydown', onRankKey)
    }
    syncPowerCaption()
    showSeconds(selectedSeconds, true)
    cleanups.push(() => {
      powerSvg.removeEventListener('pointermove', onPowerMove)
      powerSvg.removeEventListener('pointerdown', onPowerDown)
      powerSvg.removeEventListener('pointerleave', onPowerLeave)
      powerSvg.removeEventListener('pointercancel', onPowerLeave)
      powerSvg.removeEventListener('keydown', onPowerKey)
      powerBlock.removeEventListener('click', onPowerClick)
      if (rankSvg && onRankMove) rankSvg.removeEventListener('mousemove', onRankMove)
      if (rankSvg && onRankLeave) rankSvg.removeEventListener('mouseleave', onRankLeave)
      if (rankSvg && onRankDown) rankSvg.removeEventListener('pointerdown', onRankDown)
      if (rankSvg && onRankKey) rankSvg.removeEventListener('keydown', onRankKey)
      powerRangeTip?.remove()
    })
  }

  const bindWkTrend = (blockSel: string, kind: WkKind): void => {
    const rows = wkTrendRows(data, kind)
    const block = panel.querySelector<HTMLElement>(blockSel)
    const svgEl = block?.querySelector<SVGElement>('.tri-wkt-svg')
    const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
    const current = svgEl?.querySelector<SVGElement>('.tri-wkt-current')
    const wrap = block?.querySelector<HTMLElement>('.tri-wkdetail-wrap')
    const detail = block?.querySelector<HTMLElement>('.tri-wkdetail')
    if (!block || !svgEl || !cursor || !wrap || !detail || !rows.length) return
    const pts = Array.from(svgEl.querySelectorAll<SVGElement>('.tri-wkt-pt'))
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const mark = (cls: string, idx: number | null): void => {
      for (const p of pts) p.classList.toggle(cls, p.dataset.week === String(idx))
    }
    const idxAt = (event: MouseEvent): number => {
      const r = svgEl.getBoundingClientRect()
      const f = clampN((event.clientX - r.left) / r.width, 0, 1)
      return weeklyChartIndex(f, rows.length)
    }
    let selected = rows.length - 1
    let shown = selected
    let detailAnimation: Animation | null = null
    const show = (i: number): void => {
      if (i === shown) return
      shown = i
      detailAnimation?.cancel()
      renderWkDetail(block, data, kind, i, context)
      if (reduce) return
      detailAnimation = detail.animate([{ opacity: 0.35 }, { opacity: 1 }], {
        duration: 140,
        easing: 'ease-out',
      })
    }
    const onMove = (event: MouseEvent): void => {
      const i = idxAt(event)
      const cx = (weeklyChartX(i, rows.length) * ANA_W).toFixed(2)
      cursor.setAttribute('x1', cx)
      cursor.setAttribute('x2', cx)
      show(i)
      mark('tri-wkt-pt--hot', i)
      block.classList.toggle('tri-wkt--current-hover', current != null && i === rows.length - 1)
      block.classList.add('tri-chart--hover')
    }
    const onLeave = (): void => {
      block.classList.remove('tri-chart--hover')
      block.classList.remove('tri-wkt--current-hover')
      mark('tri-wkt-pt--hot', null)
      show(selected)
    }
    const applySelected = (i: number): void => {
      selected = i
      show(i)
      mark('tri-wkt-pt--sel', i)
      wrap.classList.add('tri-wkdetail-wrap--open')
    }
    const unsubscribe = context.events.subscribe('analyticsWeek', event => {
      if (event.source !== kind) applySelected(event.index)
    })
    const onClick = (event: MouseEvent): void => {
      const i = idxAt(event)
      applySelected(i)
      context.events.dispatch('analyticsWeek', { source: kind, index: i })
    }
    svgEl.addEventListener('mousemove', onMove)
    svgEl.addEventListener('mouseleave', onLeave)
    svgEl.addEventListener('click', onClick)
    cleanups.push(() => {
      unsubscribe()
      svgEl.removeEventListener('mousemove', onMove)
      svgEl.removeEventListener('mouseleave', onLeave)
      svgEl.removeEventListener('click', onClick)
      detailAnimation?.cancel()
    })
  }
  if (kind === 'weekly') bindWkTrend('.tri-ana-weekly', 'load')
  if (kind === 'effort') bindWkTrend('.tri-ana-effort', 'effort')

  const heatActivities = data.heat.activities
  const heatDays = data.heat.series
  const heatBlock = kind === 'heat' ? panel.querySelector<HTMLElement>('.tri-ana-accl') : null
  const heatSvg = heatBlock?.querySelector<SVGElement>('.tri-accl-svg')
  const heatCursor = heatSvg?.querySelector<SVGElement>('.tri-ana-cursor')
  const heatReadout = heatBlock?.querySelector<HTMLElement>('.tri-chart-readout')
  if (
    heatBlock &&
    heatSvg &&
    heatCursor &&
    heatReadout &&
    heatActivities.length &&
    heatDays.length
  ) {
    const fromMs = Date.parse(`${heatDays[0].date}T00:00:00Z`)
    const toMs = Date.parse(`${heatDays[heatDays.length - 1].date}T23:59:59Z`)
    const positioned = heatActivities.map(activity => ({
      activity,
      fraction: clampN((Date.parse(activity.startedAt) - fromMs) / (toMs - fromMs), 0, 1),
    }))
    const dayByDate = new Map(heatDays.map(day => [day.date, day]))
    const onMove = (event: MouseEvent): void => {
      const rect = heatSvg.getBoundingClientRect()
      const fraction = clampN((event.clientX - rect.left) / rect.width, 0, 1)
      let nearest = positioned[0]
      for (const point of positioned)
        if (Math.abs(point.fraction - fraction) < Math.abs(nearest.fraction - fraction))
          nearest = point
      const { activity } = nearest
      const day = dayByDate.get(activity.date)
      const temperatureDigits = activity.source === 'core' ? 2 : 0
      const temperature =
        context.presentation.distance === 'imperial'
          ? `${((activity.temperatureC * 9) / 5 + 32).toFixed(temperatureDigits)}°F`
          : `${activity.temperatureC.toFixed(temperatureDigits)}°C`
      const heatStrain =
        activity.heatStrainIndex == null ? '' : ` · HSI ${activity.heatStrainIndex.toFixed(1)}`
      const source =
        activity.source === 'core'
          ? activity.coreOrigin === 'app'
            ? 'CORE app'
            : 'CORE FIT'
          : activity.source === 'weatherkit'
            ? 'WeatherKit'
            : 'Strava'
      const cx = (nearest.fraction * ANA_W).toFixed(2)
      heatCursor.setAttribute('x1', cx)
      heatCursor.setAttribute('x2', cx)
      heatReadout.textContent = `${context.formatter.shortDate(activity.date)} · ${activity.name} · ${temperature}${heatStrain} · ${activity.hotMinutes} ${context.formatter.text('hot min')} · ${context.formatter.text('proxy')} ${day?.acclimatisationPct.toFixed(0) ?? '—'}% · ${source}`
      heatBlock.classList.add('tri-chart--hover')
    }
    const onLeave = (): void => heatBlock.classList.remove('tri-chart--hover')
    heatSvg.addEventListener('mousemove', onMove)
    heatSvg.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      heatSvg.removeEventListener('mousemove', onMove)
      heatSvg.removeEventListener('mouseleave', onLeave)
    })
  }

  const bodySeries = data.body.series
  const bodyBlock = kind === 'body' ? panel.querySelector<HTMLElement>('.tri-ana-bodywt') : null
  const bodyPlot = bodyBlock?.querySelector<HTMLElement>('.tri-bodywt-plot')
  const bodyCursor = bodyPlot?.querySelector<SVGElement>('.tri-ana-cursor')
  const bodyReadout = bodyBlock?.querySelector<HTMLElement>('.tri-chart-readout')
  if (bodyBlock && bodyPlot && bodyCursor && bodyReadout && bodySeries.length >= 2) {
    const bdays = groupBodyByDay(bodySeries)
    const bmrByDay = new Map<string, number>()
    for (const p of Array.isArray(data.body.bmrSeries) ? data.body.bmrSeries : [])
      bmrByDay.set(p.date, p.bmr)
    const ffmiByDay = new Map<string, number>()
    for (const p of Array.isArray(data.body.ffmiSeries) ? data.body.ffmiSeries : [])
      ffmiByDay.set(p.date, p.ffmi)
    const bt0 = bdays[0].ts
    const bt1 = bdays[bdays.length - 1].ts
    const bx = (ts: number): number => (bt1 > bt0 ? ((ts - bt0) / (bt1 - bt0)) * 100 : 50)
    const ranges = Array.from(bodyPlot.querySelectorAll<SVGLineElement>('.tri-bodywt-range'))
    const onMove = (event: MouseEvent) => {
      const r = bodyPlot.getBoundingClientRect()
      const fx = clampN((event.clientX - r.left) / r.width, 0, 1) * 100
      let best = bdays[0]
      let bestD = Infinity
      for (const d of bdays) {
        const dd = Math.abs(bx(d.ts) - fx)
        if (dd < bestD) {
          bestD = dd
          best = d
        }
      }
      const cx = bx(best.ts).toFixed(2)
      bodyCursor.setAttribute('x1', cx)
      bodyCursor.setAttribute('x2', cx)
      const bmrV = bmrByDay.get(best.date)
      const bmrTxt = bmrV != null ? ` · BMR ${bmrV} kcal` : ''
      const ffmiV = ffmiByDay.get(best.date)
      const ffmiTxt = ffmiV != null ? ` · FFMI ${context.formatter.number(ffmiV, 1, 1)}` : ''
      if (best.samples.length > 1) {
        const delta = best.last - best.first
        setMath(
          bodyReadout,
          `${context.formatter.shortDate(best.date)} · $${best.samples.length}\\times$ · ${wNum(context.formatter, best.min)}–${wNum(context.formatter, best.max)} ${weightUnitLabel(context.formatter)} · $\\Delta${wSigned(context.formatter, delta, 1)}$${bmrTxt}${ffmiTxt}`,
        )
      } else {
        setMath(
          bodyReadout,
          `${context.formatter.shortDate(best.date)} · ${wFmt(context.formatter, best.last)}${bmrTxt}${ffmiTxt}`,
        )
      }
      for (const ln of ranges)
        ln.classList.toggle('tri-bodywt-range--active', ln.dataset.day === best.date)
      bodyBlock.classList.add('tri-chart--hover')
    }
    const onLeave = () => {
      bodyBlock.classList.remove('tri-chart--hover')
      for (const ln of ranges) ln.classList.remove('tri-bodywt-range--active')
    }
    bodyPlot.addEventListener('mousemove', onMove)
    bodyPlot.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      bodyPlot.removeEventListener('mousemove', onMove)
      bodyPlot.removeEventListener('mouseleave', onLeave)
    })
  }

  const dexaColumns =
    kind === 'dexa' ? Array.from(panel.querySelectorAll<HTMLElement>('.tri-dexa-column')) : []
  if (dexaColumns.length > 0) {
    document.body.querySelector('.tri-dexa-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-dexa-tip')
    tip.setAttribute('role', 'tooltip')
    document.body.appendChild(tip)
    const show = (column: HTMLElement): void => {
      const head = el('span', 'tri-dexa-tip-head')
      head.append(
        el('span', 'tri-gloss-h', column.dataset.dexaRegion ?? ''),
        el('span', 'tri-dexa-tip-total', column.dataset.dexaTotal ?? ''),
      )
      const rows = [
        ['lean', 'is-lean', column.dataset.dexaLean, column.dataset.dexaLeanPct],
        ['fat', 'is-fat', column.dataset.dexaFat, column.dataset.dexaFatPct],
        ['bone', 'is-bone', column.dataset.dexaBone, column.dataset.dexaBonePct],
      ] as const
      tip.replaceChildren(
        head,
        ...rows.map(([label, cls, value, pct]) => {
          const row = el('span', 'tri-dexa-tip-row')
          row.append(
            el('span', `tri-dexa-dot ${cls}`, undefined, { 'aria-hidden': 'true' }),
            el('span', 'tri-dexa-tip-label', context.formatter.text(label)),
            el('span', 'tri-dexa-tip-value', value ?? ''),
            el('span', 'tri-dexa-tip-pct', pct ?? ''),
          )
          return row
        }),
      )
      tip.classList.add('tri-gloss--on')
    }
    const placeAtPointer = (event: MouseEvent): void => {
      const rect = tip.getBoundingClientRect()
      const left =
        event.clientX + 14 + rect.width > window.innerWidth - 8
          ? event.clientX - 14 - rect.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + rect.height > window.innerHeight - 8
          ? event.clientY - 14 - rect.height
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const placeAtColumn = (column: HTMLElement): void => {
      const columnRect = column.getBoundingClientRect()
      const tipRect = tip.getBoundingClientRect()
      const left = clampN(
        columnRect.left + columnRect.width / 2 - tipRect.width / 2,
        8,
        window.innerWidth - tipRect.width - 8,
      )
      const above = columnRect.top - tipRect.height - 8
      const top = above >= 8 ? above : columnRect.bottom + 8
      tip.style.left = `${left.toFixed(0)}px`
      tip.style.top = `${Math.min(top, window.innerHeight - tipRect.height - 8).toFixed(0)}px`
    }
    const bound: Array<{
      column: HTMLElement
      enter: (event: MouseEvent) => void
      move: (event: MouseEvent) => void
      leave: () => void
      focus: () => void
      blur: () => void
    }> = []
    for (const column of dexaColumns) {
      const enter = (event: MouseEvent): void => {
        show(column)
        placeAtPointer(event)
      }
      const move = (event: MouseEvent): void => placeAtPointer(event)
      const leave = (): void => {
        if (!column.matches(':focus-visible')) tip.classList.remove('tri-gloss--on')
      }
      const focus = (): void => {
        if (!column.matches(':focus-visible')) return
        show(column)
        placeAtColumn(column)
      }
      const blur = (): void => tip.classList.remove('tri-gloss--on')
      column.addEventListener('mouseenter', enter)
      column.addEventListener('mousemove', move)
      column.addEventListener('mouseleave', leave)
      column.addEventListener('focus', focus)
      column.addEventListener('blur', blur)
      bound.push({ column, enter, move, leave, focus, blur })
    }
    cleanups.push(() => {
      for (const { column, enter, move, leave, focus, blur } of bound) {
        column.removeEventListener('mouseenter', enter)
        column.removeEventListener('mousemove', move)
        column.removeEventListener('mouseleave', leave)
        column.removeEventListener('focus', focus)
        column.removeEventListener('blur', blur)
      }
      tip.remove()
    })
  }

  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}
