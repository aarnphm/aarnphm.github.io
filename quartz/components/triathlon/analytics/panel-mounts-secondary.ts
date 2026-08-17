import type { Analytics } from '../../../plugins/stores/analytics'
import type { CardioKey } from '../../../plugins/stores/analytics'
import type { Sport } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import type { AbilityAxis } from './panels/abilities'
import type { SportAbility } from './panels/abilities'
import { calculateFtpHypothesis } from '../../../util/ftp-hypothesis'
import { clock } from '../../../util/triathlon-card'
import { formatStrideLength } from '../../../util/triathlon-card'
import { formatVerticalOscillation } from '../../../util/triathlon-card'
import { M_TO_FT } from '../../../util/triathlon-card'
import { el } from '../runtime/dom'
import { renderGlossDef } from '../shell/glossary'
import { radarAxisDefinition } from './panels/abilities'
import { radarAxisLabel } from './panels/abilities'
import { radarNotationDefinition } from './panels/abilities'
import { radarPaceHint } from './panels/abilities'
import { cardioSeriesOf } from './panels/cardio'
import { cardioValueText } from './panels/cardio'
import { fmtTrendShort } from './panels/thresholds'
import { fmtTrendVal } from './panels/thresholds'
import { lactateThresholdSamples } from './panels/thresholds'
import { sampleTrend } from './panels/thresholds'
import { trendSamples } from './panels/thresholds'
import { scrubGroup, type ScrubItem } from './scrub-primitives'
import { ANA_W } from './shared'
import { bySport } from './shared'
import { clampN } from './shared'
import { KG_PER_LB } from './shared'
import { weightUnitLabel } from './shared'
import { wNum } from './shared'

export type SecondaryPanelKind =
  | 'cardio'
  | 'trend'
  | 'lactate'
  | 'abilities'
  | 'vo2max'
  | 'dexa'
  | 'ftp'

export const mountSecondaryPanel = (
  kind: SecondaryPanelKind,
  panel: HTMLElement,
  data: Analytics,
  context: TriathlonContext,
): (() => void) => {
  const cleanups: (() => void)[] = []
  const cardioBlock =
    kind === 'cardio' ? panel.querySelector<HTMLElement>('.tri-engine-cardio') : null
  if (cardioBlock) {
    const seriesByMetric = new Map(
      data.engine.cardio.metrics.map(metric => [
        metric.key,
        cardioSeriesOf(data.engine.cardio, metric.key),
      ]),
    )
    const dates = [
      ...new Set([...seriesByMetric.values()].flatMap(points => points.map(p => p.date))),
    ].sort()
    const valuesByMetric = new Map(
      [...seriesByMetric].map(([key, points]) => [
        key,
        new Map(points.map(point => [point.date, point.value])),
      ]),
    )
    const dateReadout = cardioBlock.querySelector<HTMLElement>('.tri-cardio-readout-date')
    const readout = cardioBlock.querySelector<HTMLElement>('.tri-cardio-readout')
    const valueCells = new Map<CardioKey, HTMLElement>()
    const items: Array<{ svgEl: SVGElement; cursor: SVGElement }> = []
    for (const metric of data.engine.cardio.metrics) {
      const row = cardioBlock.querySelector<HTMLElement>(
        `.tri-engine-row[data-metric="${metric.key}"]`,
      )
      const svgEl = row?.querySelector<SVGElement>('.tri-engine-spark')
      const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
      if (svgEl && cursor) items.push({ svgEl, cursor })
      const valueCell = cardioBlock.querySelector<HTMLElement>(
        `.tri-cardio-readout-value[data-cardio-metric="${metric.key}"]`,
      )
      if (valueCell) valueCells.set(metric.key, valueCell)
    }
    if (dates.length > 0 && dateReadout && readout && items.length > 0) {
      const startMs = Date.parse(`${dates[0]}T00:00:00Z`)
      const endMs = Date.parse(`${dates[dates.length - 1]}T00:00:00Z`)
      const spanMs = Math.max(1, endMs - startMs)
      const dateIndexAt = (fraction: number): number => {
        const targetMs = startMs + clampN(fraction, 0, 1) * spanMs
        let nearestIndex = 0
        for (let index = 1; index < dates.length; index++)
          if (
            Math.abs(Date.parse(`${dates[index]}T00:00:00Z`) - targetMs) <
            Math.abs(Date.parse(`${dates[nearestIndex]}T00:00:00Z`) - targetMs)
          )
            nearestIndex = index
        return nearestIndex
      }
      let activeIndex = dates.length - 1
      const positionReadout = (
        item: { svgEl: SVGElement },
        fraction: number,
        pointerY?: number,
      ): void => {
        const blockRect = cardioBlock.getBoundingClientRect()
        const graphRect = item.svgEl.getBoundingClientRect()
        const anchorX = graphRect.left - blockRect.left + clampN(fraction, 0, 1) * graphRect.width
        const anchorY =
          pointerY != null
            ? pointerY - blockRect.top
            : graphRect.top - blockRect.top + graphRect.height / 2
        const readoutWidth = readout.offsetWidth
        const readoutHeight = readout.offsetHeight
        const gap = 10
        const inset = 4
        const rightSpace = blockRect.width - anchorX
        const leftSpace = anchorX
        const opensRight = rightSpace >= readoutWidth + gap || rightSpace >= leftSpace
        const left = clampN(
          opensRight ? anchorX + gap : anchorX - readoutWidth - gap,
          inset,
          Math.max(inset, blockRect.width - readoutWidth - inset),
        )
        const top = clampN(
          anchorY,
          readoutHeight / 2 + inset,
          Math.max(readoutHeight / 2 + inset, blockRect.height - readoutHeight / 2 - inset),
        )
        readout.dataset.side = left < anchorX ? 'left' : 'right'
        readout.style.setProperty('--tri-cardio-readout-x', `${left}px`)
        readout.style.setProperty('--tri-cardio-readout-y', `${top}px`)
      }
      const showIndex = (
        requestedIndex: number,
        item: { svgEl: SVGElement },
        pointerY?: number,
      ): void => {
        const nextIndex = clampN(requestedIndex, 0, dates.length - 1)
        activeIndex = nextIndex
        const date = dates[activeIndex]
        const dateMs = Date.parse(`${date}T00:00:00Z`)
        const cursorFraction = clampN((dateMs - startMs) / spanMs, 0, 1)
        const cursorX = (cursorFraction * 100).toFixed(2)
        dateReadout.textContent = context.formatter.shortDate(date)
        const ariaValues = [date]
        for (const metric of data.engine.cardio.metrics) {
          const value = valuesByMetric.get(metric.key)?.get(date)
          const valueText = value != null ? cardioValueText(value, metric.unit) : '—'
          const valueCell = valueCells.get(metric.key)
          if (valueCell) valueCell.textContent = valueText
          ariaValues.push(`${context.formatter.text(metric.label)} ${valueText}`)
        }
        const ariaValueText = ariaValues.join(' · ')
        for (const item of items) {
          item.cursor.setAttribute('x1', cursorX)
          item.cursor.setAttribute('x2', cursorX)
          item.svgEl.setAttribute('aria-valuenow', String(activeIndex + 1))
          item.svgEl.setAttribute('aria-valuetext', ariaValueText)
        }
        cardioBlock.classList.add('tri-chart--hover')
        positionReadout(item, cursorFraction, pointerY)
      }
      const hide = (): void => cardioBlock.classList.remove('tri-chart--hover')
      const bound: Array<{
        svgEl: SVGElement
        move: (event: MouseEvent) => void
        leave: () => void
        focus: () => void
        blur: () => void
        keydown: (event: KeyboardEvent) => void
      }> = []
      for (const item of items) {
        const move = (event: MouseEvent): void => {
          const rect = item.svgEl.getBoundingClientRect()
          showIndex(dateIndexAt((event.clientX - rect.left) / rect.width), item, event.clientY)
        }
        const leave = (): void => {
          if (!item.svgEl.matches(':focus-visible')) hide()
        }
        const focus = (): void => showIndex(activeIndex, item)
        const blur = (): void => {
          window.queueMicrotask(() => {
            if (!cardioBlock.contains(document.activeElement)) hide()
          })
        }
        const keydown = (event: KeyboardEvent): void => {
          const nextIndex =
            event.key === 'ArrowLeft'
              ? activeIndex - 1
              : event.key === 'ArrowRight'
                ? activeIndex + 1
                : event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? dates.length - 1
                    : null
          if (nextIndex == null) return
          event.preventDefault()
          showIndex(nextIndex, item)
        }
        item.svgEl.addEventListener('mousemove', move)
        item.svgEl.addEventListener('mouseleave', leave)
        item.svgEl.addEventListener('focus', focus)
        item.svgEl.addEventListener('blur', blur)
        item.svgEl.addEventListener('keydown', keydown)
        bound.push({ svgEl: item.svgEl, move, leave, focus, blur, keydown })
      }
      cleanups.push(() => {
        for (const item of bound) {
          item.svgEl.removeEventListener('mousemove', item.move)
          item.svgEl.removeEventListener('mouseleave', item.leave)
          item.svgEl.removeEventListener('focus', item.focus)
          item.svgEl.removeEventListener('blur', item.blur)
          item.svgEl.removeEventListener('keydown', item.keydown)
        }
      })
    }
  }

  const trendBlock = kind === 'trend' ? panel.querySelector<HTMLElement>('.tri-ana-trend') : null
  if (trendBlock) {
    const items: ScrubItem[] = []
    for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
      const wrap = trendBlock.querySelector<HTMLElement>(`.tri-trend-panel[data-sport="${sport}"]`)
      const svgEl = wrap?.querySelector<SVGElement>('.tri-trend-svg')
      const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
      const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
      const tr = bySport(data.trends, sport)
      const samples = tr ? trendSamples(tr) : null
      if (!wrap || !svgEl || !cursor || !readout || !tr || !samples) continue
      items.push({
        svgEl,
        cursor,
        readout,
        hover: wrap,
        textOf: f => {
          const at = sampleTrend(samples, f)
          const band = `${fmtTrendShort(context.formatter, sport, Math.min(at.lo, at.hi))}–${fmtTrendShort(context.formatter, sport, Math.max(at.lo, at.hi))}`
          return `+${(at.days / 7).toFixed(1)} wk · ${fmtTrendVal(context.formatter, sport, at.value)} · ${band}`
        },
      })
    }
    cleanups.push(scrubGroup(items, f => f * ANA_W))
  }

  const lactateBlock =
    kind === 'lactate' ? panel.querySelector<HTMLElement>('.tri-ana-lactate') : null
  if (lactateBlock) {
    const items: ScrubItem[] = []
    for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
      const wrap = lactateBlock.querySelector<HTMLElement>(`.tri-lt-panel[data-sport="${sport}"]`)
      const svgEl = wrap?.querySelector<SVGElement>('.tri-trend-svg')
      const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
      const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
      const projection = bySport(data.engine.lactateThreshold.sports, sport)
      const samples = projection ? lactateThresholdSamples(projection) : null
      if (!wrap || !svgEl || !cursor || !readout || !projection || !samples) continue
      items.push({
        svgEl,
        cursor,
        readout,
        hover: wrap,
        textOf: f => {
          const at = sampleTrend(samples, f)
          const band = `${fmtTrendShort(context.formatter, sport, Math.min(at.lo, at.hi))}–${fmtTrendShort(context.formatter, sport, Math.max(at.lo, at.hi))}`
          return `+${(at.days / 7).toFixed(1)} wk · LT2 ${fmtTrendVal(context.formatter, sport, at.value)} · ${band}`
        },
      })
    }
    cleanups.push(scrubGroup(items, f => f * ANA_W))
  }

  const radarBlock =
    kind === 'abilities' ? panel.querySelector<HTMLElement>('.tri-engine-radar') : null
  const radarSvg = radarBlock?.querySelector<SVGElement>('.tri-radar-svg')
  if (radarBlock && radarSvg) {
    document.body.querySelector('.tri-radar-tip')?.remove()
    const radarTip = el('div', 'tri-gloss tri-radar-tip')
    radarTip.setAttribute('role', 'tooltip')
    document.body.appendChild(radarTip)
    const abilities = data.engine.abilities.sports
    const rawOf = (sp: SportAbility, a: AbilityAxis): string => {
      const pace = radarPaceHint(context.presentation, sp.sport, a)
      if (a.rawValue == null) return context.formatter.text('no data')
      if (a.rawUnit === 'm') return formatStrideLength(context.presentation, a.rawValue)
      if (a.rawUnit === 'cm') return formatVerticalOscillation(context.presentation, a.rawValue)
      if (a.rawUnit === 's/100m') return `${clock(a.rawValue)} /100m`
      const vamFt = a.rawUnit === 'm/h' && context.presentation.distance === 'imperial'
      const value = vamFt ? Math.round(a.rawValue * M_TO_FT) : a.rawValue
      return `${value} ${vamFt ? 'ft/h' : a.rawUnit}${pace ? ` (${pace})` : ''}`
    }
    const projTxtOf = (a: AbilityAxis): string =>
      a.proj != null && a.proj !== a.score ? ` → ${a.proj}/100` : ''
    const onMove = (event: MouseEvent) => {
      const n = abilities[0]?.axes.length ?? 0
      if (!n) return
      const rect = radarSvg.getBoundingClientRect()
      const dx = event.clientX - (rect.left + rect.width / 2)
      const dy = event.clientY - (rect.top + rect.height / 2)
      const deg = (Math.atan2(dy, dx) * 180) / Math.PI
      const idx = ((Math.round(((deg + 90) / 360) * n) % n) + n) % n
      const pressedSports = (radarBlock.dataset.pressed ?? '').split(',').filter(Boolean)
      if (!pressedSports.length) {
        radarTip.classList.remove('tri-gloss--on')
        return
      }
      const single =
        pressedSports.length === 1 ? abilities.find(sp => sp.sport === pressedSports[0]) : undefined
      if (single) {
        const a = single.axes[idx]
        radarTip.replaceChildren(
          el(
            'span',
            'tri-gloss-h',
            `${context.formatter.text(single.sport)} · ${context.formatter.text(a.label)}`,
          ),
          el(
            'span',
            'tri-gloss-def',
            `${rawOf(single, a)} · ${a.score != null ? `${a.score}/100` : '—'}${projTxtOf(a)}`,
          ),
          renderGlossDef(radarAxisDefinition(context.presentation, single.sport, a)),
          renderGlossDef(radarNotationDefinition(context.presentation, a)),
        )
      } else {
        const activeAbilities = abilities.filter(sp => pressedSports.includes(sp.sport))
        const rows: HTMLElement[] = [
          el(
            'span',
            'tri-gloss-h',
            radarAxisLabel(context.presentation, activeAbilities, idx) ||
              context.formatter.text(abilities[0].axes[idx].label),
          ),
        ]
        if (radarBlock.dataset.sport === 'avg') {
          const xs = abilities
            .filter(sp => pressedSports.includes(sp.sport))
            .map(sp => sp.axes[idx].score)
            .filter((v): v is number => v != null)
          if (xs.length)
            rows.push(
              el(
                'span',
                'tri-gloss-def',
                `${context.formatter.text('average')}: ${Math.round(xs.reduce((acc, v) => acc + v, 0) / xs.length)}/100`,
              ),
            )
        }
        for (const sp of abilities) {
          if (!pressedSports.includes(sp.sport)) continue
          const a = sp.axes[idx]
          rows.push(
            el(
              'span',
              'tri-gloss-def',
              `${context.formatter.text(sp.sport)}: ${a.score != null ? `${a.score}/100` : '—'}${projTxtOf(a)} · ${rawOf(sp, a)}`,
            ),
          )
        }
        radarTip.replaceChildren(...rows)
      }
      radarTip.classList.add('tri-gloss--on')
      const pr = radarTip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      radarTip.style.left = `${Math.max(8, left).toFixed(0)}px`
      radarTip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const onLeave = () => radarTip.classList.remove('tri-gloss--on')
    radarSvg.addEventListener('mousemove', onMove)
    radarSvg.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      radarSvg.removeEventListener('mousemove', onMove)
      radarSvg.removeEventListener('mouseleave', onLeave)
      radarTip.remove()
    })
  }

  const vo2TestTargets = Array.from(panel.querySelectorAll<HTMLElement>('.tri-vo2t')).flatMap(
    test =>
      Array.from(test.querySelectorAll<HTMLElement>('[data-tip-h]')).filter(
        target => !target.closest('.tri-vo2p'),
      ),
  )
  if (vo2TestTargets.length > 0) {
    document.body.querySelector('.tri-vo2t-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-vo2t-tip')
    tip.setAttribute('role', 'tooltip')
    document.body.appendChild(tip)
    const place = (event: MouseEvent): void => {
      const pr = tip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const bound: Array<[HTMLElement, (e: MouseEvent) => void, () => void]> = []
    for (const t of vo2TestTargets) {
      const move = (e: MouseEvent): void => {
        tip.replaceChildren(
          el('span', 'tri-gloss-h', t.dataset.tipH ?? ''),
          el('span', 'tri-gloss-def', t.dataset.tipD ?? ''),
        )
        tip.classList.add('tri-gloss--on')
        place(e)
      }
      const leave = (): void => tip.classList.remove('tri-gloss--on')
      t.addEventListener('mousemove', move)
      t.addEventListener('mouseleave', leave)
      bound.push([t, move, leave])
    }
    cleanups.push(() => {
      for (const [t, move, leave] of bound) {
        t.removeEventListener('mousemove', move)
        t.removeEventListener('mouseleave', leave)
      }
      tip.remove()
    })
  }

  const vo2Profiles = Array.from(panel.querySelectorAll<HTMLElement>('.tri-vo2p'))
  if (vo2Profiles.length > 0) {
    document.body.querySelector('.tri-vo2p-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-vo2p-tip')
    tip.setAttribute('role', 'tooltip')
    document.body.appendChild(tip)
    const place = (event: MouseEvent): void => {
      const pr = tip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const bound: Array<[SVGElement, (e: MouseEvent) => void, () => void]> = []
    for (const t of vo2Profiles.flatMap(profile =>
      Array.from(profile.querySelectorAll<SVGElement>('[data-tip-h]')),
    )) {
      const move = (e: MouseEvent): void => {
        tip.replaceChildren(
          el('span', 'tri-gloss-h', t.getAttribute('data-tip-h') ?? ''),
          el('span', 'tri-gloss-def', t.getAttribute('data-tip-d') ?? ''),
        )
        tip.classList.add('tri-gloss--on')
        place(e)
      }
      const leave = (): void => tip.classList.remove('tri-gloss--on')
      t.addEventListener('mousemove', move)
      t.addEventListener('mouseleave', leave)
      bound.push([t, move, leave])
    }
    cleanups.push(() => {
      for (const [t, move, leave] of bound) {
        t.removeEventListener('mousemove', move)
        t.removeEventListener('mouseleave', leave)
      }
      tip.remove()
    })
  }

  const ftpBlock = kind === 'ftp' ? panel.querySelector<HTMLElement>('.tri-ftp') : null
  if (ftpBlock) {
    const inputs = Array.from(ftpBlock.querySelectorAll<HTMLInputElement>('[data-ftp-param]'))
    const out = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-out="${key}"]`)
    const valOut = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-val="${key}"]`)
    const bar = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-bar="${key}"]`)
    const inputFor = (key: string): HTMLInputElement | undefined =>
      inputs.find(input => input.dataset.ftpParam === key)
    const num = (key: string): number => Number(inputFor(key)?.value ?? 0)
    const setText = (key: string, value: string): void => {
      const node = out(key)
      if (node) node.textContent = value
    }
    const setBar = (key: string, value: number): void => {
      const node = bar(key)
      if (node) node.style.width = `${clampN((value / 350) * 100, 4, 100)}%`
    }
    const setNum = (key: string, display: string, unit: string): void => {
      const numIn = ftpBlock.querySelector<HTMLInputElement>(`[data-ftp-num="${key}"]`)
      if (numIn && document.activeElement !== numIn) numIn.value = display
      const unitNode = ftpBlock.querySelector<HTMLElement>(`[data-ftp-unit="${key}"]`)
      if (unitNode) unitNode.textContent = unit
    }
    const renderFtp = (): void => {
      const mass = num('mass')
      const vo2 = num('vo2')
      const discountPct = num('discount')
      const thresholdPct = num('threshold')
      const efficiencyPct = num('efficiency')
      const calculation = calculateFtpHypothesis(vo2, mass, {
        crossModalDiscountPct: discountPct,
        thresholdPct,
        grossEfficiencyPct: efficiencyPct,
      })
      if (!calculation) return
      const writeVal = (key: string, value: string): void => {
        const node = valOut(key)
        if (node) node.textContent = value
      }
      setNum('mass', wNum(context.formatter, mass, 1, 0), ` ${weightUnitLabel(context.formatter)}`)
      setNum('vo2', vo2.toFixed(1), '')
      writeVal('discount', `${discountPct}%`)
      writeVal('threshold', `${thresholdPct}%`)
      writeVal('efficiency', `${efficiencyPct}%`)
      setText('headline', String(calculation.ftp))
      setText('band', `${calculation.low}-${calculation.high} W`)
      setText('wkg', `${calculation.wattsPerKg.toFixed(2)} W/kg`)
      setText('efficiencyFtp', `${Math.round(calculation.efficiencyFtp)} W`)
      setText('acsmFtp', `${Math.round(calculation.acsmFtp)} W`)
      setText('absoluteVo2', `${calculation.absoluteVo2.toFixed(2)} L/min`)
      setText('cyclingVo2max', `${calculation.cyclingVo2max.toFixed(2)} L/min`)
      setText('thresholdVo2', `${calculation.thresholdVo2.toFixed(2)} L/min`)
      setText('metabolicWatts', `${Math.round(calculation.metabolicWatts)} W`)
      setText('acsmMapWatts', `${Math.round(calculation.acsmMapWatts)} W`)
      setBar('efficiencyFtp', calculation.efficiencyFtp)
      setBar('acsmFtp', calculation.acsmFtp)
    }
    const onInput = () => renderFtp()
    for (const input of inputs) input.addEventListener('input', onInput)
    const numInputs = Array.from(ftpBlock.querySelectorAll<HTMLInputElement>('[data-ftp-num]'))
    const onNum = (e: Event): void => {
      const numIn = e.target as HTMLInputElement
      const key = numIn.dataset.ftpNum
      const range = key ? inputFor(key) : undefined
      if (!key || !range) return
      let v = Number(numIn.value)
      if (!Number.isFinite(v)) return renderFtp()
      if (key === 'mass' && context.presentation.distance === 'imperial') v *= KG_PER_LB
      range.value = String(clampN(v, Number(range.min), Number(range.max)))
      renderFtp()
    }
    const onNumKey = (e: KeyboardEvent): void => {
      if (e.key === 'Enter') {
        e.preventDefault()
        ;(e.target as HTMLInputElement).blur()
      }
    }
    const onNumFocus = (e: Event): void => (e.target as HTMLInputElement).select()
    for (const numIn of numInputs) {
      numIn.addEventListener('change', onNum)
      numIn.addEventListener('keydown', onNumKey)
      numIn.addEventListener('focus', onNumFocus)
    }
    const reset = ftpBlock.querySelector<HTMLButtonElement>('.tri-ftp-reset')
    const onReset = (): void => {
      for (const input of inputs)
        if (input.dataset.ftpDefault) input.value = input.dataset.ftpDefault
      renderFtp()
    }
    reset?.addEventListener('click', onReset)
    renderFtp()
    cleanups.push(() => {
      for (const input of inputs) input.removeEventListener('input', onInput)
      for (const numIn of numInputs) {
        numIn.removeEventListener('change', onNum)
        numIn.removeEventListener('keydown', onNumKey)
        numIn.removeEventListener('focus', onNumFocus)
      }
      reset?.removeEventListener('click', onReset)
    })
  }

  return () => {
    for (const c of cleanups) c()
  }
}
