import type { Locale } from '../../../util/triathlon-presentation'
import { daySleepStageLabel } from '../../../util/triathlon-card'
import { wallClock, wallMin } from '../analytics/panels/recovery'

export type DaySleepValueLabel = (value: number | null) => string

type DaySleepGeometry = { x: (index: number) => number; indexAt: (fraction: number) => number }

type DaySleepSeries = {
  values: readonly (number | null)[]
  startMinute: number
  intervalSeconds: number
  label: DaySleepValueLabel
  geometry: DaySleepGeometry
}

const finitePositive = (value: string | undefined): number | null => {
  if (!value) return null
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? number : null
}

export const decodeDaySleepValues = (encoded: string): (number | null)[] | null => {
  const values = encoded.split(',').map(value => {
    if (value === '') return null
    const number = Number(value)
    return Number.isFinite(number) ? number : undefined
  })
  return values.length >= 2 && values.every(value => value !== undefined)
    ? values.map(value => value ?? null)
    : null
}

export const daySleepUnitLabel =
  (unit: 'ms' | 'bpm'): DaySleepValueLabel =>
  value =>
    `${value == null ? '—' : Math.round(value)} ${unit}`

export const daySleepReadout = (
  startMinute: number,
  intervalSeconds: number,
  values: readonly (number | null)[],
  index: number,
  label: DaySleepValueLabel,
): string => {
  const boundedIndex = Math.min(Math.max(Math.round(index), 0), values.length - 1)
  const time = wallClock(startMinute + (boundedIndex * intervalSeconds) / 60)
  return `${time} · ${label(values[boundedIndex])}`
}

const pointGeometry = (count: number, width: number): DaySleepGeometry => ({
  x: index => (index / (count - 1)) * width,
  indexAt: fraction => Math.round(fraction * (count - 1)),
})

const bandGeometry = (count: number, width: number): DaySleepGeometry => ({
  x: index => ((index + 0.5) / count) * width,
  indexAt: fraction => Math.min(count - 1, Math.floor(fraction * count)),
})

const seriesFromElement = (wrap: HTMLElement, locale: () => Locale): DaySleepSeries | null => {
  const values = wrap.dataset.daySleepValues
    ? decodeDaySleepValues(wrap.dataset.daySleepValues)
    : null
  const startTs = wrap.dataset.daySleepStart
  const intervalSeconds = finitePositive(wrap.dataset.daySleepInterval)
  const width = finitePositive(wrap.dataset.daySleepWidth)
  const startMinute = startTs ? wallMin(startTs) : Number.NaN
  if (!values || !Number.isFinite(startMinute) || intervalSeconds == null || width == null)
    return null
  const unit = wrap.dataset.daySleepUnit
  const stages = wrap.dataset.daySleepSeries === 'stages'
  if (!stages && unit !== 'ms' && unit !== 'bpm') return null
  return {
    values,
    startMinute,
    intervalSeconds,
    label: stages
      ? value => daySleepStageLabel(locale(), value)
      : daySleepUnitLabel(unit === 'ms' ? 'ms' : 'bpm'),
    geometry: stages ? bandGeometry(values.length, width) : pointGeometry(values.length, width),
  }
}

const latestMeasuredIndex = (values: readonly (number | null)[]): number => {
  for (let index = values.length - 1; index >= 0; index -= 1)
    if (values[index] != null) return index
  return values.length - 1
}

const mountDaySleepChart = (wrap: HTMLElement, locale: () => Locale): (() => void) => {
  const series = seriesFromElement(wrap, locale)
  const svg = wrap.querySelector<SVGSVGElement>('.tri-ana-svg')
  const cursor = wrap.querySelector<SVGLineElement>('.tri-ana-cursor')
  const readout = wrap.querySelector<HTMLElement>('.tri-chart-readout')
  if (!series || !svg || !cursor || !readout) return () => {}

  let currentIndex = latestMeasuredIndex(series.values)
  const reveal = (index: number, input: 'pointer' | 'keyboard'): void => {
    currentIndex = Math.min(Math.max(Math.round(index), 0), series.values.length - 1)
    const x = series.geometry.x(currentIndex)
    const text = daySleepReadout(
      series.startMinute,
      series.intervalSeconds,
      series.values,
      currentIndex,
      series.label,
    )
    cursor.setAttribute('x1', x.toFixed(2))
    cursor.setAttribute('x2', x.toFixed(2))
    readout.textContent = text
    svg.setAttribute('aria-valuenow', currentIndex.toString())
    svg.setAttribute('aria-valuetext', text)
    wrap.dataset.sleepInput = input
    wrap.classList.add('tri-chart--hover')
  }
  const hide = (): void => {
    wrap.classList.remove('tri-chart--hover')
    delete wrap.dataset.sleepInput
  }
  const onPointerMove = (event: PointerEvent): void => {
    if (event.pointerType === 'touch') return
    const bounds = svg.getBoundingClientRect()
    if (bounds.width <= 0) return
    const fraction = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1)
    reveal(series.geometry.indexAt(fraction), 'pointer')
  }
  const onPointerLeave = (): void => {
    if (document.activeElement !== svg) hide()
  }
  const onFocus = (): void => reveal(currentIndex, 'keyboard')
  const onBlur = (): void => hide()
  const onKeyDown = (event: KeyboardEvent): void => {
    let next = currentIndex
    if (event.key === 'ArrowLeft') next--
    else if (event.key === 'ArrowRight') next++
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = series.values.length - 1
    else return
    event.preventDefault()
    reveal(next, 'keyboard')
  }

  svg.addEventListener('pointermove', onPointerMove)
  svg.addEventListener('pointerleave', onPointerLeave)
  svg.addEventListener('focus', onFocus)
  svg.addEventListener('blur', onBlur)
  svg.addEventListener('keydown', onKeyDown)
  return () => {
    svg.removeEventListener('pointermove', onPointerMove)
    svg.removeEventListener('pointerleave', onPointerLeave)
    svg.removeEventListener('focus', onFocus)
    svg.removeEventListener('blur', onBlur)
    svg.removeEventListener('keydown', onKeyDown)
  }
}

export const mountDaySleepCharts = (scope: ParentNode, locale: () => Locale): (() => void) => {
  const cleanups = Array.from(
    scope.querySelectorAll<HTMLElement>('[data-day-sleep-series]'),
    wrap => mountDaySleepChart(wrap, locale),
  )
  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}
