import type { Analytics } from '../../../../../plugins/stores/analytics'
import type { PowerRankInterval } from '../../../../../plugins/stores/power-rank'
import type { TriathlonContext } from '../../../runtime/context'
import { zoneClock } from '../../../../../util/triathlon-card'
import { el, setText } from '../../../runtime/dom'
import { rafEvent, rectOf } from '../../scrub-primitives'
import { clampN } from '../../shared'
import {
  powerRankEffortLabel,
  powerRankProgressLabel,
  powerRankProgressNextLabel,
  powerRankRangeRows,
} from './rank'
import { bestPowerSeries, bestPowerSeriesLabel, type BestPowerSeriesKey } from './series'

interface PowerRangeTipSlots {
  title: HTMLElement
  best: HTMLElement
  body: HTMLElement
  rows: Array<{
    tr: HTMLElement
    level: HTMLElement
    pct: HTMLElement
    pwr: HTMLElement
    res: HTMLElement
  }>
}

const powerRangeTipSlots = new WeakMap<HTMLElement, PowerRangeTipSlots>()

const updatePowerRangeTip = (
  tip: HTMLElement,
  interval: PowerRankInterval,
  key: BestPowerSeriesKey,
): void => {
  let slots = powerRangeTipSlots.get(tip)
  const dataRows = powerRankRangeRows(interval, key)
  if (!slots || slots.rows.length !== dataRows.length) {
    const head = el('div', 'tri-power-range-tip-head')
    const title = el('strong', 'tri-power-range-tip-title')
    const best = el('span', 'tri-power-range-tip-best')
    head.append(title, best)
    const table = el('table', 'tri-power-range-table')
    const tableHead = el('thead')
    const headerRow = el('tr')
    for (const label of ['levels', '%', 'power', 'your best'])
      headerRow.appendChild(el('th', undefined, label, { scope: 'col' }))
    tableHead.appendChild(headerRow)
    const tableBody = el('tbody')
    const rows = dataRows.map(() => {
      const tr = el('tr', 'tri-power-range-row')
      const level = el('th', 'tri-power-range-level', '', { scope: 'row' })
      const pct = el('td')
      const pwr = el('td')
      const res = el('td', 'tri-power-range-result')
      tr.append(level, pct, pwr, res)
      tableBody.appendChild(tr)
      return { tr, level, pct, pwr, res }
    })
    table.append(tableHead, tableBody)
    tip.replaceChildren(head, table)
    slots = { title, best, body: tableBody, rows }
    powerRangeTipSlots.set(tip, slots)
  }
  const effort = interval.efforts[key]
  setText(slots.title, `${zoneClock(interval.durationS)} power`)
  setText(
    slots.best,
    effort == null
      ? 'no data'
      : `${effort.watts.toLocaleString()} W · ${effort.wattsPerKg.toFixed(2)} W/kg`,
  )
  for (let i = 0; i < dataRows.length; i++) {
    const row = dataRows[i]
    const slot = slots.rows[i]
    slot.tr.classList.toggle('tri-power-range-row--current', Boolean(row.current))
    if (row.current) slot.tr.setAttribute('aria-current', 'true')
    else slot.tr.removeAttribute('aria-current')
    setText(slot.level, `${row.level} ${row.label}`)
    setText(slot.pct, `${row.percentile}%`)
    setText(slot.pwr, `${row.thresholdWatts.toLocaleString()} W`)
    setText(
      slot.res,
      row.bestWatts != null
        ? `${row.bestWatts.toLocaleString()} W`
        : row.gapWatts != null
          ? `+${row.gapWatts.toLocaleString()} W`
          : '',
    )
  }
}

export const mountPowerRank = (
  powerBlock: HTMLElement,
  power: Analytics['powerCurve'],
  context: TriathlonContext,
  activePowerSeries: ReadonlySet<BestPowerSeriesKey>,
  showSeconds: (seconds: number, commit: boolean) => number | null,
  restoreSelection: () => void,
): { sync: (seconds: number) => void; reset: () => void; cleanup: () => void } => {
  const powerSeries = bestPowerSeries(power)
  const rankBlock = powerBlock.querySelector<HTMLElement>('.tri-power-radar')
  const rankSvg = rankBlock?.querySelector<SVGSVGElement>('.tri-power-radar-svg')
  const rankIntervals = power.ranking.intervals
  const rankDuration = rankBlock?.querySelector<HTMLElement>('.tri-power-radar-duration')
  const rankLabels = rankBlock
    ? Array.from(rankBlock.querySelectorAll<SVGElement>('.tri-power-radar-label'))
    : []
  const progressTrack = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-track]')
  const progressFill = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-fill]')
  const progressLabel = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-label]')
  const progressNext = powerBlock.querySelector<HTMLElement>('[data-power-rank-progress-next]')
  const rankDom = new Map(
    powerSeries.map(({ key }) => {
      const row = rankBlock?.querySelector<HTMLElement>(
        `.tri-power-radar-readout-row[data-power-rank-series="${key}"]`,
      )
      return [
        key,
        {
          row,
          value: row?.querySelector<HTMLElement>('.tri-power-radar-readout-value') ?? null,
          swatch: row?.querySelector<HTMLElement>('.tri-best-power-swatch') ?? null,
        },
      ] as const
    }),
  )
  document.body.querySelector('.tri-power-range-tip')?.remove()
  const powerRangeTip =
    rankIntervals.length === 0
      ? null
      : el('div', 'tri-gloss tri-power-range-tip', undefined, {
          role: 'tooltip',
          'aria-hidden': 'true',
        })
  if (powerRangeTip) document.body.appendChild(powerRangeTip)
  let shownRankKey = ''
  let tipWidth = 0
  let tipHeight = 0
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
    const primaryKey: BestPowerSeriesKey = activePowerSeries.has('six-weeks') ? 'six-weeks' : 'year'
    const rankKey = `${index}:${primaryKey}:${activePowerSeries.size}`
    if (rankKey === shownRankKey) return
    shownRankKey = rankKey
    const interval = rankIntervals[index]
    rankSvg.dataset.powerRankIndex = String(index)
    rankSvg.setAttribute('aria-valuenow', String(interval.durationS))
    for (const label of rankLabels)
      label.classList.toggle(
        'tri-power-radar-label--active',
        Number(label.dataset.powerRankIndex) === index,
      )
    if (rankDuration) setText(rankDuration, `${zoneClock(interval.durationS)} · ${interval.skill}`)
    const valueText: string[] = []
    for (const { key } of powerSeries) {
      const dom = rankDom.get(key)
      const effort = activePowerSeries.has(key) ? interval.efforts[key] : null
      if (dom?.row) {
        const hideRow = !activePowerSeries.has(key)
        if (dom.row.hidden !== hideRow) dom.row.hidden = hideRow
        if (dom.value) setText(dom.value, powerRankEffortLabel(effort))
        if (dom.swatch) dom.swatch.dataset.powerSkill = interval.skill
      }
      if (effort)
        valueText.push(
          `${bestPowerSeriesLabel(context.formatter, power, key)} ${powerRankEffortLabel(effort)}`,
        )
    }
    const primaryEffort = interval.efforts[primaryKey]
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
    if (progressLabel) setText(progressLabel, powerRankProgressLabel(primaryEffort))
    if (progressNext) setText(progressNext, powerRankProgressNextLabel(primaryEffort))
    if (powerRangeTip) {
      powerRangeTip.dataset.powerSkill = interval.skill
      if (
        powerRangeTip.dataset.powerRankIndex !== String(index) ||
        powerRangeTip.dataset.powerSeries !== primaryKey
      ) {
        powerRangeTip.dataset.powerRankIndex = String(index)
        powerRangeTip.dataset.powerSeries = primaryKey
        updatePowerRangeTip(powerRangeTip, interval, primaryKey)
      }
    }
    rankSvg.setAttribute(
      'aria-valuetext',
      `${zoneClock(interval.durationS)} ${interval.skill}; ${valueText.join('; ')}`,
    )
  }
  const hidePowerRangeTip = (): void => {
    powerRangeTip?.classList.remove('tri-gloss--on')
    powerRangeTip?.setAttribute('aria-hidden', 'true')
  }
  const placePowerRangeTip = (clientX: number, clientY: number): void => {
    if (!powerRangeTip) return
    if (tipWidth <= 0 || tipHeight <= 0) {
      tipWidth = powerRangeTip.offsetWidth
      tipHeight = powerRangeTip.offsetHeight
    }
    const gap = 14
    const margin = 8
    const preferredLeft = clientX + gap
    const left =
      preferredLeft + tipWidth <= window.innerWidth - margin
        ? preferredLeft
        : clientX - gap - tipWidth
    const maxTop = Math.max(margin, window.innerHeight - tipHeight - margin)
    const top = clampN(clientY - tipHeight / 2, margin, maxTop)
    powerRangeTip.style.left = `${Math.max(margin, left).toFixed(0)}px`
    powerRangeTip.style.top = `${top.toFixed(0)}px`
  }
  const showPowerRangeTip = (clientX: number, clientY: number): void => {
    if (!powerRangeTip) return
    powerRangeTip.classList.add('tri-gloss--on')
    powerRangeTip.setAttribute('aria-hidden', 'false')
    placePowerRangeTip(clientX, clientY)
  }
  let onRankMove: ((event: MouseEvent) => void) | null = null
  let onRankLeave: (() => void) | null = null
  let onRankDown: ((event: PointerEvent) => void) | null = null
  let onRankKey: ((event: KeyboardEvent) => void) | null = null
  let cancelRankMove: (() => void) | null = null
  if (rankSvg && rankIntervals.length > 0) {
    const indexAt = (clientX: number, clientY: number): number => {
      const rect = rectOf(rankSvg)
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
    const rafRank = rafEvent((event: MouseEvent) => {
      const index = indexAt(event.clientX, event.clientY)
      const interval = rankIntervals[index]
      showSeconds(interval.durationS, false)
      showPowerRangeTip(event.clientX, event.clientY)
    })
    onRankMove = rafRank.onEvent
    cancelRankMove = rafRank.cancel
    onRankLeave = () => {
      rafRank.cancel()
      hidePowerRangeTip()
      restoreSelection()
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
  return {
    sync: syncPowerRank,
    reset: () => {
      shownRankKey = ''
    },
    cleanup: () => {
      cancelRankMove?.()
      if (rankSvg && onRankMove) rankSvg.removeEventListener('mousemove', onRankMove)
      if (rankSvg && onRankLeave) rankSvg.removeEventListener('mouseleave', onRankLeave)
      if (rankSvg && onRankDown) rankSvg.removeEventListener('pointerdown', onRankDown)
      if (rankSvg && onRankKey) rankSvg.removeEventListener('keydown', onRankKey)
      powerRangeTip?.remove()
    },
  }
}
