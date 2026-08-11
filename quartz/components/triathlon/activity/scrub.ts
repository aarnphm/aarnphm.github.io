import type { PowerCurvePoint } from '../../../plugins/stores/strava'
import type { SwimTrendChartPoint } from '../../../util/triathlon-card'
import type { SwimTrendMode } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { clock } from '../../../util/triathlon-card'
import { decodePowerCurve } from '../../../util/triathlon-card'
import { powerCurveFraction } from '../../../util/triathlon-card'
import { powerCurveHoverAt } from '../../../util/triathlon-card'
import { swimTrendHoverAt } from '../../../util/triathlon-card'
import { zoneClock } from '../../../util/triathlon-card'
import { powerCurveReferenceLabel } from '../../../util/triathlon-i18n'
import { swimActivityComparisonText } from '../../../util/triathlon-i18n'
import { swimActivityDisplayValue } from '../../../util/triathlon-i18n'
import { swimActivityDistanceText } from '../../../util/triathlon-i18n'
import { swimActivityHeaderValue } from '../../../util/triathlon-i18n'
import { swimActivityPointText } from '../../../util/triathlon-i18n'
import { swimActivityValueText } from '../../../util/triathlon-i18n'
import { triText } from '../../../util/triathlon-i18n'

export const setupChartScrub = (
  scope: HTMLElement,
  presentation: () => TriathlonPresentation,
): (() => void) => {
  const text = (key: string): string => triText(presentation().locale, key)
  type CurveRange = 'six-weeks' | 'year'
  let activeWrap: HTMLElement | null = null
  let activeBar: Element | null = null
  let focusedSvg: SVGSVGElement | null = null
  const curveCache = new WeakMap<
    SVGSVGElement,
    { curve: PowerCurvePoint[]; sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] }
  >()
  const swimCache = new WeakMap<
    SVGSVGElement,
    { lengths: SwimTrendChartPoint[]; '100m': SwimTrendChartPoint[] }
  >()
  const swimAnimations = new Map<SVGGElement, Animation>()
  const curveData = (
    svg: SVGSVGElement,
  ): { curve: PowerCurvePoint[]; sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] } => {
    const cached = curveCache.get(svg)
    if (cached) return cached
    const value = {
      curve: decodePowerCurve(svg.dataset.curve),
      sixWeeks: decodePowerCurve(svg.dataset.curveRefSixWeeks),
      year: decodePowerCurve(svg.dataset.curveRefYear),
    }
    curveCache.set(svg, value)
    return value
  }
  const isSwimTrendPoint = (value: unknown): value is SwimTrendChartPoint =>
    typeof value === 'object' &&
    value !== null &&
    'elapsedS' in value &&
    typeof value.elapsedS === 'number' &&
    Number.isFinite(value.elapsedS) &&
    'cumulativeDistanceM' in value &&
    typeof value.cumulativeDistanceM === 'number' &&
    Number.isFinite(value.cumulativeDistanceM) &&
    'value' in value &&
    typeof value.value === 'number' &&
    Number.isFinite(value.value) &&
    'xPct' in value &&
    typeof value.xPct === 'number' &&
    Number.isFinite(value.xPct) &&
    'yPct' in value &&
    typeof value.yPct === 'number' &&
    Number.isFinite(value.yPct) &&
    (!('windowStartDistanceM' in value) ||
      (typeof value.windowStartDistanceM === 'number' &&
        Number.isFinite(value.windowStartDistanceM)))
  const swimMode = (svg: SVGSVGElement): SwimTrendMode =>
    svg.dataset.swimMode === '100m' ? '100m' : 'lengths'
  const decodeSwimData = (value: string | undefined): SwimTrendChartPoint[] => {
    const parsed: unknown = JSON.parse(value ?? '[]')
    return Array.isArray(parsed) ? parsed.filter(isSwimTrendPoint) : []
  }
  const swimData = (
    svg: SVGSVGElement,
    mode: SwimTrendMode = swimMode(svg),
  ): SwimTrendChartPoint[] => {
    const cached = swimCache.get(svg)
    if (cached) return cached[mode]
    const value = {
      lengths: decodeSwimData(svg.dataset.swimSeriesLengths),
      '100m': decodeSwimData(svg.dataset.swimSeriesHundred),
    }
    swimCache.set(svg, value)
    return value[mode]
  }
  const curveRange = (svg: SVGSVGElement): CurveRange =>
    svg.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
  const curveReference = (
    svg: SVGSVGElement,
    data: { sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] },
  ): PowerCurvePoint[] => (curveRange(svg) === 'year' ? data.year : data.sixWeeks)
  const curveReferenceYear = (svg: SVGSVGElement): number | null => {
    if (curveRange(svg) !== 'year') return null
    const year = Number(svg.dataset.curveYear)
    return Number.isInteger(year) ? year : null
  }
  const selectedCurveIndex = (svg: SVGSVGElement): number => {
    const value = Number(svg.dataset.curveSelectedIndex ?? 0)
    return Number.isInteger(value) ? value : 0
  }
  const selectedSwimIndex = (svg: SVGSVGElement): number => {
    const value = Number(svg.dataset.swimIndex ?? 0)
    return Number.isInteger(value) ? value : 0
  }
  const curveValueText = (
    svg: SVGSVGElement,
    point: PowerCurvePoint,
    referenceWatts: number | null,
  ): string =>
    `${zoneClock(point.s)}, ${text('this ride')} ${point.w.toLocaleString()} watts${referenceWatts == null ? '' : `, ${powerCurveReferenceLabel(presentation().locale, curveReferenceYear(svg))} ${referenceWatts.toLocaleString()} watts`}`
  const swimKind = (svg: SVGSVGElement): 'pace' | 'stroke' =>
    svg.dataset.swimKind === 'stroke' ? 'stroke' : 'pace'
  const swimAriaLabel = (svg: SVGSVGElement): string =>
    `${text('swim')} ${text(swimKind(svg) === 'pace' ? 'pace' : 'stroke rate')} · ${text(swimMode(svg) === '100m' ? '100 m' : 'lengths')}`
  const swimDisplayValue = (kind: 'pace' | 'stroke', value: number): string =>
    swimActivityDisplayValue(presentation().locale, kind, value, clock(value))
  const swimTextPoint = (point: SwimTrendChartPoint) => ({
    elapsed: clock(point.elapsedS),
    cumulativeDistanceM: point.cumulativeDistanceM,
    ...(point.windowStartDistanceM == null
      ? {}
      : { windowStartDistanceM: point.windowStartDistanceM }),
  })
  const swimValueText = (kind: 'pace' | 'stroke', point: SwimTrendChartPoint): string =>
    swimActivityValueText(
      presentation().locale,
      kind,
      swimTextPoint(point),
      point.value,
      clock(point.value),
    )
  const deactivate = (wrap: HTMLElement): void => {
    wrap.classList.remove('tri-chart--hover')
    const point = wrap.querySelector<HTMLElement>('.tri-swim-trend-hover')
    if (point) point.hidden = true
  }
  const activate = (wrap: HTMLElement): void => {
    if (activeWrap && activeWrap !== wrap) deactivate(activeWrap)
    activeWrap = wrap
    wrap.classList.add('tri-chart--hover')
  }
  const clear = (): void => {
    if (activeWrap) deactivate(activeWrap)
    activeBar?.classList.remove('tri-hist-bar--on')
    activeWrap = null
    activeBar = null
  }
  const placeCurvePoint = (
    point: HTMLElement | null,
    x: number,
    watts: number | null,
    maxWatts: number,
    height: number,
  ): void => {
    if (!point) return
    point.hidden = watts == null
    if (watts == null) return
    const y = height - (Math.min(maxWatts, Math.max(0, watts)) / maxWatts) * (height - 1)
    point.style.left = `${x.toFixed(2)}%`
    point.style.top = `${((y / height) * 100).toFixed(2)}%`
  }
  const showCurve = (
    svg: SVGSVGElement,
    fraction: number,
    activateChart = true,
    commit = false,
  ): void => {
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const readout = wrap?.querySelector<HTMLElement>('.tri-curve-readout')
    if (!wrap || !readout) return
    const data = curveData(svg)
    const curve = data.curve
    const reference = curveReference(svg, data)
    const hover = powerCurveHoverAt(curve, reference, fraction)
    if (!hover) return
    if (commit) {
      svg.dataset.curveSelectedIndex = String(hover.index)
      for (const tick of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-tick'))
        tick.setAttribute(
          'aria-pressed',
          String(Number(tick.dataset.curveSeconds) === hover.durationS),
        )
    }
    if (svg.dataset.curveIndex !== String(hover.index)) {
      cursor?.setAttribute('x1', hover.xPct.toFixed(2))
      cursor?.setAttribute('x2', hover.xPct.toFixed(2))
      const duration = readout.querySelector<HTMLElement>('.tri-curve-readout-duration')
      const ride = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ride')
      const referenceRow = readout.querySelector<HTMLElement>('.tri-curve-readout-row--ref')
      const referenceValue = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ref')
      const referenceLabel = readout.querySelector<HTMLElement>('.tri-curve-readout-label--ref')
      const maxWatts = Number(svg.dataset.curveDomainMax)
      const height = svg.viewBox.baseVal.height
      if (Number.isFinite(maxWatts) && maxWatts > 0 && height > 0) {
        placeCurvePoint(
          wrap.querySelector<HTMLElement>('.tri-curve-point--ride'),
          hover.xPct,
          hover.watts,
          maxWatts,
          height,
        )
        placeCurvePoint(
          wrap.querySelector<HTMLElement>('.tri-curve-point--ref'),
          hover.xPct,
          hover.referenceWatts,
          maxWatts,
          height,
        )
      }
      if (duration) duration.textContent = zoneClock(hover.durationS)
      if (ride) ride.textContent = `${hover.watts.toLocaleString()} W`
      if (referenceRow) referenceRow.hidden = hover.referenceWatts == null
      if (referenceValue && hover.referenceWatts != null)
        referenceValue.textContent = `${hover.referenceWatts.toLocaleString()} W`
      if (referenceLabel)
        referenceLabel.textContent = powerCurveReferenceLabel(
          presentation().locale,
          curveReferenceYear(svg),
        )
      svg.dataset.curveIndex = String(hover.index)
      svg.setAttribute('aria-valuenow', String(hover.durationS))
      svg.setAttribute(
        'aria-valuetext',
        curveValueText(svg, { s: hover.durationS, w: hover.watts }, hover.referenceWatts),
      )
    }
    if (activateChart) {
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = null
      activate(wrap)
    }
  }
  const showCurveIndex = (
    svg: SVGSVGElement,
    requestedIndex: number,
    activateChart = true,
    commit = false,
  ): void => {
    const { curve } = curveData(svg)
    if (curve.length < 2) return
    const index = Math.min(curve.length - 1, Math.max(0, requestedIndex))
    showCurve(
      svg,
      powerCurveFraction(curve[index].s, curve[0].s, curve[curve.length - 1].s),
      activateChart,
      commit,
    )
  }
  const showSwim = (svg: SVGSVGElement, fraction: number, activateChart = true): void => {
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const point = wrap?.querySelector<HTMLElement>('.tri-swim-trend-hover')
    const readoutPosition = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-position')
    const readoutValue = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-value')
    if (!wrap || !point || !readoutPosition || !readoutValue) return
    const kind = swimKind(svg)
    const hover = swimTrendHoverAt(swimData(svg), fraction)
    if (!hover) return
    cursor?.setAttribute('x1', hover.xPct.toFixed(2))
    cursor?.setAttribute('x2', hover.xPct.toFixed(2))
    point.style.left = `${hover.xPct.toFixed(2)}%`
    point.style.top = `${hover.yPct.toFixed(2)}%`
    point.hidden = !activateChart
    readoutPosition.textContent = swimActivityPointText(presentation().locale, swimTextPoint(hover))
    readoutValue.textContent = swimDisplayValue(kind, hover.value)
    svg.dataset.swimIndex = String(hover.index)
    svg.setAttribute('aria-label', swimAriaLabel(svg))
    svg.setAttribute('aria-valuenow', String(Math.round(hover.cumulativeDistanceM)))
    svg.setAttribute('aria-valuetext', swimValueText(kind, hover))
    if (activateChart) {
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = null
      activate(wrap)
    }
  }
  const showSwimIndex = (
    svg: SVGSVGElement,
    requestedIndex: number,
    activateChart = true,
  ): void => {
    const points = swimData(svg)
    if (points.length === 0) return
    const index = Math.min(points.length - 1, Math.max(0, requestedIndex))
    showSwim(svg, points[index].xPct / 100, activateChart)
  }
  const onSwimRestore = (event: Event): void => {
    if (
      !(event.target instanceof SVGSVGElement) ||
      !event.target.classList.contains('tri-swim-trend-svg')
    )
      return
    const svg = event.target
    const distanceM = Number(svg.dataset.swimRestoreDistance)
    const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
    const activateChart = svg.dataset.swimRestoreActive === 'true'
    delete svg.dataset.swimRestoreDistance
    delete svg.dataset.swimRestoreActive
    if (!Number.isFinite(distanceM) || !Number.isFinite(totalDistanceM) || totalDistanceM <= 0)
      return
    showSwim(svg, distanceM / totalDistanceM, activateChart)
  }
  const showFocused = (): void => {
    if (!focusedSvg) {
      clear()
      return
    }
    if (focusedSvg.classList.contains('tri-curve-svg'))
      showCurveIndex(focusedSvg, selectedCurveIndex(focusedSvg))
    else showSwimIndex(focusedSvg, selectedSwimIndex(focusedSvg))
  }
  const onPointer = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>(
      '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg',
    )
    if (!svg) {
      if (activeWrap) showFocused()
      return
    }
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
    const r = svg.getBoundingClientRect()
    const frac = r.width > 0 ? Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) : 0
    if (svg.classList.contains('tri-curve-svg')) {
      const commit = event.type === 'pointerdown'
      showCurve(svg, frac, true, commit)
      if (commit) {
        focusedSvg = svg
        svg.focus({ preventScroll: true })
      }
      return
    }
    if (svg.classList.contains('tri-swim-trend-svg')) {
      showSwim(svg, frac)
      return
    } else {
      const hist = JSON.parse(svg.dataset.hist ?? '[]') as number[]
      const n = hist.length
      if (n < 2) return
      const total = hist.reduce((a, b) => a + b, 0) || 1
      const bin = Math.max(0, Math.min(n - 1, Math.floor(frac * n)))
      cursor?.setAttribute('x1', `${bin + 0.5}`)
      cursor?.setAttribute('x2', `${bin + 0.5}`)
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = svg.querySelector(`.tri-hist-bar[data-bin="${bin}"]`)
      activeBar?.classList.add('tri-hist-bar--on')
      if (readout)
        readout.textContent = `${bin * 25}–${bin * 25 + 24} W · ${zoneClock(hist[bin])} (${((hist[bin] / total) * 100).toFixed(1)}%)`
    }
    if (wrap) activate(wrap)
  }
  const onFocus = (event: FocusEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    focusedSvg = svg
    if (svg.classList.contains('tri-curve-svg')) showCurveIndex(svg, selectedCurveIndex(svg))
    else showSwimIndex(svg, selectedSwimIndex(svg))
  }
  const onBlur = (event: FocusEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    if (focusedSvg === svg) focusedSvg = null
    clear()
  }
  const onKey = (event: KeyboardEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    const isCurve = svg.classList.contains('tri-curve-svg')
    const length = isCurve ? curveData(svg).curve.length : swimData(svg).length
    if (length < 2) return
    const current = isCurve ? selectedCurveIndex(svg) : selectedSwimIndex(svg)
    let next: number | null = null
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = current - 1
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = current + 1
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = length - 1
    else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      focusedSvg = null
      svg.blur()
      clear()
      return
    }
    if (next == null) return
    event.preventDefault()
    focusedSvg = svg
    if (isCurve) showCurveIndex(svg, next, true, true)
    else showSwimIndex(svg, next)
  }
  const setSwimLayer = (svg: SVGSVGElement, mode: SwimTrendMode, animate: boolean): void => {
    const previousMode = swimMode(svg)
    if (previousMode === mode) return
    const previous = svg.querySelector<SVGGElement>(
      `.tri-swim-series[data-swim-mode="${previousMode}"]`,
    )
    const next = svg.querySelector<SVGGElement>(`.tri-swim-series[data-swim-mode="${mode}"]`)
    if (!previous || !next) return
    const previousOpacity = getComputedStyle(previous).opacity
    const nextOpacity = getComputedStyle(next).opacity
    swimAnimations.get(previous)?.cancel()
    swimAnimations.get(next)?.cancel()
    swimAnimations.delete(previous)
    swimAnimations.delete(next)
    previous.classList.remove('tri-swim-series--active')
    previous.setAttribute('aria-hidden', 'true')
    next.classList.add('tri-swim-series--active')
    next.setAttribute('aria-hidden', 'false')
    svg.dataset.swimMode = mode
    if (!animate || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timing: KeyframeAnimationOptions = {
      duration: 180,
      easing: 'cubic-bezier(0.77, 0, 0.175, 1)',
    }
    swimAnimations.set(
      previous,
      previous.animate([{ opacity: previousOpacity }, { opacity: 0 }], timing),
    )
    swimAnimations.set(next, next.animate([{ opacity: nextOpacity }, { opacity: 1 }], timing))
  }
  const setSwimMode = (section: HTMLElement, mode: SwimTrendMode, animate: boolean): void => {
    const toggle = section.querySelector<HTMLElement>('.tri-swim-mode-toggle')
    if (!toggle || toggle.dataset.swimMode === mode) return
    toggle.dataset.swimMode = mode
    for (const option of toggle.querySelectorAll<HTMLButtonElement>('.tri-swim-mode'))
      option.setAttribute('aria-pressed', String(option.dataset.swimMode === mode))
    for (const svg of section.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
      const previous = swimData(svg)
      const selected = Math.min(previous.length - 1, Math.max(0, selectedSwimIndex(svg)))
      const fraction = previous[selected]?.xPct != null ? previous[selected].xPct / 100 : 1
      const wrap = svg.closest<HTMLElement>('.tri-zone')
      const wasActive = wrap?.classList.contains('tri-chart--hover') ?? false
      setSwimLayer(svg, mode, animate)
      delete svg.dataset.swimIndex
      showSwim(svg, fraction, wasActive)
    }
  }
  const onChartClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const swimButton = event.target.closest<HTMLButtonElement>('.tri-swim-mode')
    const swimSection = swimButton?.closest<HTMLElement>('.tri-swim-trends')
    if (swimButton && swimSection) {
      const mode: SwimTrendMode = swimButton.dataset.swimMode === '100m' ? '100m' : 'lengths'
      setSwimMode(swimSection, mode, event.detail > 0)
      return
    }
    const curveTick = event.target.closest<HTMLButtonElement>('.tri-curve-tick')
    const curveAxis = event.target.closest<HTMLElement>('.tri-curve-chart .tri-cax-xax')
    const curveWrap = (curveTick ?? curveAxis)?.closest<HTMLElement>('.tri-curve-chart')
    const curveSvg = curveWrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
    if (curveWrap && curveSvg) {
      if (curveTick) {
        const seconds = Number(curveTick.dataset.curveSeconds)
        const data = curveData(curveSvg).curve
        if (seconds > 0 && data.length >= 2)
          showCurve(
            curveSvg,
            powerCurveFraction(seconds, data[0].s, data[data.length - 1].s),
            true,
            true,
          )
      } else if (curveAxis) {
        const rect = curveAxis.getBoundingClientRect()
        if (rect.width > 0)
          showCurve(curveSvg, (event.clientX - rect.left) / rect.width, true, true)
      }
      focusedSvg = curveSvg
      curveSvg.focus({ preventScroll: true })
      return
    }
    const button = event.target.closest<HTMLButtonElement>('.tri-curve-range')
    const wrap = button?.closest<HTMLElement>('.tri-zone')
    const svg = wrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
    if (!button || button.disabled || !wrap || !svg) return
    const range: CurveRange = button.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
    const data = curveData(svg)
    const reference = range === 'year' ? data.year : data.sixWeeks
    if (reference.length === 0) return
    const index = selectedCurveIndex(svg)
    const wasActive = wrap.classList.contains('tri-chart--hover')
    svg.dataset.curveRange = range
    for (const option of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-range'))
      option.setAttribute('aria-pressed', String(option.dataset.curveRange === range))
    for (const path of svg.querySelectorAll<SVGElement>('.tri-curve-ref[data-curve-range]'))
      path.toggleAttribute('hidden', path.dataset.curveRange !== range)
    delete svg.dataset.curveIndex
    showCurveIndex(svg, index, wasActive)
  }
  const onLocale = (): void => {
    for (const delta of scope.querySelectorAll<HTMLElement>('.tri-swim-trend-delta')) {
      const kind = delta.dataset.swimComparisonKind === 'stroke' ? 'stroke' : 'pace'
      const rawDelta = delta.dataset.swimComparisonDelta
      const rawPrior = delta.dataset.swimComparisonPrior
      const comparisonDelta = rawDelta == null ? null : Number(rawDelta)
      const comparisonPrior = rawPrior == null ? null : Number(rawPrior)
      delta.textContent = swimActivityComparisonText(
        presentation().locale,
        kind,
        comparisonDelta != null && Number.isFinite(comparisonDelta) ? comparisonDelta : null,
        comparisonPrior != null && Number.isInteger(comparisonPrior) ? comparisonPrior : null,
      )
    }
    for (const average of scope.querySelectorAll<HTMLElement>('.tri-swim-trend-value')) {
      const kind = average.dataset.swimAverageKind === 'stroke' ? 'stroke' : 'pace'
      const value = Number(average.dataset.swimAverageValue)
      if (Number.isFinite(value))
        average.textContent = swimActivityHeaderValue(
          presentation().locale,
          kind,
          value,
          clock(value),
        )
    }
    for (const svg of scope.querySelectorAll<SVGSVGElement>('.tri-curve-svg')) {
      const data = curveData(svg)
      const curve = data.curve
      const reference = curveReference(svg, data)
      if (curve.length < 2) continue
      const index = Math.min(curve.length - 1, Math.max(0, selectedCurveIndex(svg)))
      const point = curve[index]
      const referenceWatts = reference.find(candidate => candidate.s === point.s)?.w ?? null
      const referenceLabel = svg
        .closest<HTMLElement>('.tri-zone')
        ?.querySelector<HTMLElement>('.tri-curve-readout-label--ref')
      if (referenceLabel)
        referenceLabel.textContent = powerCurveReferenceLabel(
          presentation().locale,
          curveReferenceYear(svg),
        )
      svg.setAttribute('aria-label', text('power curve'))
      svg.setAttribute('aria-valuenow', String(point.s))
      svg.setAttribute('aria-valuetext', curveValueText(svg, point, referenceWatts))
    }
    for (const svg of scope.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
      svg.setAttribute('aria-label', swimAriaLabel(svg))
      const wrap = svg.closest<HTMLElement>('.tri-zone')
      const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
      if (wrap && Number.isFinite(totalDistanceM)) {
        const distances = [0, totalDistanceM / 2, totalDistanceM]
        wrap.querySelectorAll<HTMLElement>('.tri-cax-xt').forEach((tick, index) => {
          const distanceM = distances[index]
          if (distanceM != null)
            tick.textContent = swimActivityDistanceText(presentation().locale, distanceM)
        })
      }
      showSwimIndex(
        svg,
        selectedSwimIndex(svg),
        wrap?.classList.contains('tri-chart--hover') ?? false,
      )
    }
  }
  const onPointerLeave = (): void => showFocused()
  scope.addEventListener('pointermove', onPointer)
  scope.addEventListener('pointerdown', onPointer)
  scope.addEventListener('pointerleave', onPointerLeave)
  scope.addEventListener('pointercancel', onPointerLeave)
  scope.addEventListener('focusin', onFocus)
  scope.addEventListener('focusout', onBlur)
  scope.addEventListener('keydown', onKey)
  scope.addEventListener('click', onChartClick)
  scope.addEventListener('tri:swim-restore', onSwimRestore)
  window.addEventListener('tri:locale', onLocale)
  onLocale()
  return () => {
    clear()
    for (const animation of swimAnimations.values()) animation.cancel()
    swimAnimations.clear()
    scope.removeEventListener('pointermove', onPointer)
    scope.removeEventListener('pointerdown', onPointer)
    scope.removeEventListener('pointerleave', onPointerLeave)
    scope.removeEventListener('pointercancel', onPointerLeave)
    scope.removeEventListener('focusin', onFocus)
    scope.removeEventListener('focusout', onBlur)
    scope.removeEventListener('keydown', onKey)
    scope.removeEventListener('click', onChartClick)
    scope.removeEventListener('tri:swim-restore', onSwimRestore)
    window.removeEventListener('tri:locale', onLocale)
  }
}
