import type { Analytics } from '../../../plugins/stores/analytics'
import type { TriathlonContext } from '../runtime/context'
import type { WkKind } from './panels/performance'
import { weeklyChartIndex } from '../../../util/weekly-target-range'
import { weeklyChartX } from '../../../util/weekly-target-range'
import { daySleepBarIndex } from '../activity/day-sleep'
import { el, setText } from '../runtime/dom'
import { groupBodyByDay } from './panels/body'
import { renderWkDetail, wkTrendRows } from './panels/performance'
import { mountPowerCurve } from './panels/power'
import { rafEvent, rectOf, scrubBind, setFastMath } from './scrub-primitives'
import { ANA_W, clampN, hms, signed, weightUnitLabel, wFmt, wNum, wSigned } from './shared'

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
    indexAt?: (fraction: number) => number,
  ) => {
    const block = panel.querySelector<HTMLElement>(blockSel)
    const svgEl = block?.querySelector<SVGElement>(svgSel)
    const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
    const readout = block?.querySelector<HTMLElement>('.tri-chart-readout')
    if (block && svgEl && cursor && readout)
      cleanups.push(scrubBind(block, svgEl, cursor, readout, count, vbW, textOf, indexAt))
  }

  if (kind === 'recovery') {
    const rec = data.recovery.series
    bind('.tri-ana-recovery', '.tri-rec-svg', rec.length, ANA_W, i => {
      const d = rec[i]
      const z = d.hrvZ != null ? ` ${signed(d.hrvZ)}$\\sigma$` : ''
      return `${d.date} · HRV ${d.hrv ?? '—'}${z} · RHR ${d.rhr ?? '—'} · rdy ${d.readiness ?? '—'}`
    })
  }

  if (kind === 'sleep') {
    const sleepView = data.recovery.series
    const indexAt = (fraction: number): number => daySleepBarIndex(fraction, sleepView.length)
    bind(
      '.tri-sleep-overview--night',
      '.tri-sleep-svg',
      sleepView.length,
      sleepView.length,
      i => {
        const d = sleepView[i]
        const debt = d.sleepDebtS != null ? `${(d.sleepDebtS / 3600).toFixed(1)}h` : '—'
        return `${d.date} · ${d.sleepS != null ? hms(d.sleepS) : '—'} · score ${d.sleepScore ?? '—'} · debt ${debt}`
      },
      indexAt,
    )
  }

  if (kind === 'vo2max') {
    const trend = data.engine.vo2max.trend
    bind('.tri-engine-vo2', '.tri-engine-vo2-spark', trend.length, ANA_W, i => {
      const p = trend[i]
      const src = p.method === 'bike' ? `bike (${context.formatter.text('projected')})` : p.method
      return `${p.weekStart} · ${p.vo2max.toFixed(1)} ml/kg/min · ${src}`
    })
  }

  if (kind === 'power') cleanups.push(mountPowerCurve(panel, data, context))

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
      const r = rectOf(svgEl)
      const f = clampN((event.clientX - r.left) / r.width, 0, 1)
      return weeklyChartIndex(f, rows.length)
    }
    let selected = rows.length - 1
    let shown = selected
    let hotIdx: number | null = null
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
    const rafMove = rafEvent((event: MouseEvent): void => {
      const i = idxAt(event)
      if (i !== hotIdx) {
        hotIdx = i
        const cx = (weeklyChartX(i, rows.length) * ANA_W).toFixed(2)
        cursor.setAttribute('x1', cx)
        cursor.setAttribute('x2', cx)
        show(i)
        mark('tri-wkt-pt--hot', i)
        block.classList.toggle('tri-wkt--current-hover', current != null && i === rows.length - 1)
      }
      block.classList.add('tri-chart--hover')
    })
    const onLeave = (): void => {
      rafMove.cancel()
      hotIdx = null
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
    svgEl.addEventListener('mousemove', rafMove.onEvent)
    svgEl.addEventListener('mouseleave', onLeave)
    svgEl.addEventListener('click', onClick)
    cleanups.push(() => {
      rafMove.cancel()
      unsubscribe()
      svgEl.removeEventListener('mousemove', rafMove.onEvent)
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
    let lastNearest: (typeof positioned)[0] | null = null
    const rafMove = rafEvent((event: MouseEvent): void => {
      const rect = rectOf(heatSvg)
      if (rect.width <= 0) return
      const fraction = clampN((event.clientX - rect.left) / rect.width, 0, 1)
      let nearest = positioned[0]
      for (const point of positioned)
        if (Math.abs(point.fraction - fraction) < Math.abs(nearest.fraction - fraction))
          nearest = point
      if (nearest !== lastNearest) {
        lastNearest = nearest
        const { activity } = nearest
        const day = dayByDate.get(activity.date)
        const temperatureDigits = activity.source === 'core' ? 2 : 0
        const temperatureC = activity.sauna?.temperatureC ?? activity.temperatureC
        const temperature =
          temperatureC == null
            ? '—'
            : context.presentation.distance === 'imperial'
              ? `${((temperatureC * 9) / 5 + 32).toFixed(temperatureDigits)}°F`
              : `${temperatureC.toFixed(temperatureDigits)}°C`
        const heatStrain =
          activity.heatStrainIndex == null ? '' : ` · HSI ${activity.heatStrainIndex.toFixed(1)}`
        const source =
          activity.source === 'core'
            ? activity.coreOrigin === 'app'
              ? 'CORE app'
              : 'CORE FIT'
            : activity.source === 'weatherkit'
              ? 'WeatherKit'
              : activity.source === 'manual-sauna'
                ? context.formatter.text('manual sauna')
                : 'Strava'
        const exposure = activity.sauna
          ? `${context.formatter.text('room temperature')} ${temperature} · ${activity.sauna.humidityPct}% ${context.formatter.text('humidity')} · ${activity.sauna.durationMinutes.toFixed(0)} ${context.formatter.text('sauna min')} · HTL ${activity.sauna.heatTrainingLoad?.toFixed(1) ?? '—'} · ${context.formatter.text('cooldown')}: ${context.formatter.text(activity.sauna.cooldown)}`
          : `${temperature}${heatStrain} · ${activity.hotMinutes} ${context.formatter.text('hot min')}`
        const cx = (nearest.fraction * ANA_W).toFixed(2)
        heatCursor.setAttribute('x1', cx)
        heatCursor.setAttribute('x2', cx)
        setText(
          heatReadout,
          `${context.formatter.shortDate(activity.date)} · ${activity.name} · ${exposure} · ${context.formatter.text('proxy')} ${day?.acclimatisationPct.toFixed(0) ?? '—'}% · ${source}`,
        )
      }
      heatBlock.classList.add('tri-chart--hover')
    })
    const onLeave = (): void => {
      rafMove.cancel()
      heatBlock.classList.remove('tri-chart--hover')
    }
    heatSvg.addEventListener('mousemove', rafMove.onEvent)
    heatSvg.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      rafMove.cancel()
      heatSvg.removeEventListener('mousemove', rafMove.onEvent)
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
    let lastBestDate = ''
    const rafMove = rafEvent((event: MouseEvent) => {
      const r = rectOf(bodyPlot)
      if (r.width <= 0) return
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
      if (best.date !== lastBestDate) {
        lastBestDate = best.date
        const cx = bx(best.ts).toFixed(2)
        bodyCursor.setAttribute('x1', cx)
        bodyCursor.setAttribute('x2', cx)
        const bmrV = bmrByDay.get(best.date)
        const bmrTxt = bmrV != null ? ` · BMR ${bmrV} kcal` : ''
        const ffmiV = ffmiByDay.get(best.date)
        const ffmiTxt = ffmiV != null ? ` · FFMI ${context.formatter.number(ffmiV, 1, 1)}` : ''
        if (best.samples.length > 1) {
          const delta = best.last - best.first
          setFastMath(
            bodyReadout,
            `${context.formatter.shortDate(best.date)} · ${best.samples.length}$\\times$ · ${wNum(context.formatter, best.min)}–${wNum(context.formatter, best.max)} ${weightUnitLabel(context.formatter)} · $\\Delta$${wSigned(context.formatter, delta, 1)}${bmrTxt}${ffmiTxt}`,
          )
        } else {
          setFastMath(
            bodyReadout,
            `${context.formatter.shortDate(best.date)} · ${wFmt(context.formatter, best.last)}${bmrTxt}${ffmiTxt}`,
          )
        }
        for (const ln of ranges)
          ln.classList.toggle('tri-bodywt-range--active', ln.dataset.day === best.date)
      }
      bodyBlock.classList.add('tri-chart--hover')
    })
    const onLeave = () => {
      rafMove.cancel()
      lastBestDate = ''
      bodyBlock.classList.remove('tri-chart--hover')
      for (const ln of ranges) ln.classList.remove('tri-bodywt-range--active')
    }
    bodyPlot.addEventListener('mousemove', rafMove.onEvent)
    bodyPlot.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      rafMove.cancel()
      bodyPlot.removeEventListener('mousemove', rafMove.onEvent)
      bodyPlot.removeEventListener('mouseleave', onLeave)
    })
  }

  const dexaColumns =
    kind === 'dexa' ? Array.from(panel.querySelectorAll<HTMLElement>('.tri-dexa-column')) : []
  if (dexaColumns.length > 0) {
    document.body.querySelector('.tri-dexa-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-dexa-tip')
    tip.setAttribute('role', 'tooltip')
    const head = el('span', 'tri-dexa-tip-head')
    const regionEl = el('span', 'tri-gloss-h')
    const totalEl = el('span', 'tri-dexa-tip-total')
    head.append(regionEl, totalEl)
    const rowSpecs = [
      ['lean', 'is-lean'],
      ['fat', 'is-fat'],
      ['bone', 'is-bone'],
    ] as const
    const rowSlots = rowSpecs.map(([label, cls]) => {
      const row = el('span', 'tri-dexa-tip-row')
      const valEl = el('span', 'tri-dexa-tip-value')
      const pctEl = el('span', 'tri-dexa-tip-pct')
      row.append(
        el('span', `tri-dexa-dot ${cls}`, undefined, { 'aria-hidden': 'true' }),
        el('span', 'tri-dexa-tip-label', context.formatter.text(label)),
        valEl,
        pctEl,
      )
      return { row, valEl, pctEl }
    })
    tip.append(head, ...rowSlots.map(s => s.row))
    document.body.appendChild(tip)
    let tipWidth = 0
    let tipHeight = 0
    const show = (column: HTMLElement): void => {
      setText(regionEl, column.dataset.dexaRegion ?? '')
      setText(totalEl, column.dataset.dexaTotal ?? '')
      const vals = [
        [column.dataset.dexaLean, column.dataset.dexaLeanPct],
        [column.dataset.dexaFat, column.dataset.dexaFatPct],
        [column.dataset.dexaBone, column.dataset.dexaBonePct],
      ] as const
      for (let i = 0; i < 3; i++) {
        setText(rowSlots[i].valEl, vals[i][0] ?? '')
        setText(rowSlots[i].pctEl, vals[i][1] ?? '')
      }
      tip.classList.add('tri-gloss--on')
      if (tipWidth <= 0 || tipHeight <= 0) {
        tipWidth = tip.offsetWidth
        tipHeight = tip.offsetHeight
      }
    }
    const placeAtPointer = (event: MouseEvent): void => {
      const left =
        event.clientX + 14 + tipWidth > window.innerWidth - 8
          ? event.clientX - 14 - tipWidth
          : event.clientX + 14
      const top =
        event.clientY + 14 + tipHeight > window.innerHeight - 8
          ? event.clientY - 14 - tipHeight
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const placeAtColumn = (column: HTMLElement): void => {
      const columnRect = rectOf(column)
      const left = clampN(
        columnRect.left + columnRect.width / 2 - tipWidth / 2,
        8,
        window.innerWidth - tipWidth - 8,
      )
      const above = columnRect.top - tipHeight - 8
      const top = above >= 8 ? above : columnRect.bottom + 8
      tip.style.left = `${left.toFixed(0)}px`
      tip.style.top = `${Math.min(top, window.innerHeight - tipHeight - 8).toFixed(0)}px`
    }
    const rafPlace = rafEvent(placeAtPointer)
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
      const move = rafPlace.onEvent
      const leave = (): void => {
        rafPlace.cancel()
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
      rafPlace.cancel()
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
