import { wallClock, wallMin } from '../analytics/panels/recovery'

type DaySleepSeries = {
  values: readonly (number | null)[]
  startMinute: number
  intervalSeconds: number
  unit: 'ms' | 'bpm'
  width: number
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

export const daySleepReadout = (
  startMinute: number,
  intervalSeconds: number,
  values: readonly (number | null)[],
  index: number,
  unit: 'ms' | 'bpm',
): string => {
  const boundedIndex = Math.min(Math.max(Math.round(index), 0), values.length - 1)
  const time = wallClock(startMinute + (boundedIndex * intervalSeconds) / 60)
  const value = values[boundedIndex]
  return `${time} · ${value == null ? '—' : Math.round(value)} ${unit}`
}

const seriesFromElement = (wrap: HTMLElement): DaySleepSeries | null => {
  const values = wrap.dataset.daySleepValues
    ? decodeDaySleepValues(wrap.dataset.daySleepValues)
    : null
  const startTs = wrap.dataset.daySleepStart
  const intervalSeconds = finitePositive(wrap.dataset.daySleepInterval)
  const width = finitePositive(wrap.dataset.daySleepWidth)
  const unit = wrap.dataset.daySleepUnit
  const startMinute = startTs ? wallMin(startTs) : Number.NaN
  if (
    !values ||
    !Number.isFinite(startMinute) ||
    intervalSeconds == null ||
    width == null ||
    (unit !== 'ms' && unit !== 'bpm')
  )
    return null
  return { values, startMinute, intervalSeconds, unit, width }
}

const latestMeasuredIndex = (values: readonly (number | null)[]): number => {
  for (let index = values.length - 1; index >= 0; index -= 1)
    if (values[index] != null) return index
  return values.length - 1
}

const mountDaySleepChart = (wrap: HTMLElement): (() => void) => {
  const series = seriesFromElement(wrap)
  const svg = wrap.querySelector<SVGSVGElement>('.tri-day-sleep-line-svg')
  const cursor = wrap.querySelector<SVGLineElement>('.tri-ana-cursor')
  const readout = wrap.querySelector<HTMLElement>('.tri-chart-readout')
  if (!series || !svg || !cursor || !readout) return () => {}

  let currentIndex = latestMeasuredIndex(series.values)
  const reveal = (index: number, input: 'pointer' | 'keyboard'): void => {
    currentIndex = Math.min(Math.max(Math.round(index), 0), series.values.length - 1)
    const x = (currentIndex / (series.values.length - 1)) * series.width
    const text = daySleepReadout(
      series.startMinute,
      series.intervalSeconds,
      series.values,
      currentIndex,
      series.unit,
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
    reveal(fraction * (series.values.length - 1), 'pointer')
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

export const mountDaySleepCharts = (scope: ParentNode): (() => void) => {
  const cleanups = Array.from(
    scope.querySelectorAll<HTMLElement>('[data-day-sleep-series]'),
    mountDaySleepChart,
  )
  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}
