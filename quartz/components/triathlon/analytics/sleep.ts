import type { Analytics } from '../../../plugins/stores/analytics'
import type { OuraDayDetail, OuraSeries } from '../../../plugins/stores/oura'
import type { TriathlonContext } from '../runtime/context'
import {
  buildOuraDayDetail,
  buildSleeplessRock,
  OURA_STAGE,
  wallClock,
  wallMin,
} from './panels/recovery'
import { scrubBind } from './scrub-primitives'
import { ANA_W, clampN } from './shared'

const mountSeriesScrub = (
  scope: HTMLElement,
  key: 'stages' | 'hrv' | 'hr',
  count: number,
  width: number,
  text: (index: number) => string,
): (() => void) => {
  const wrap = scope.querySelector<HTMLElement>(`[data-oura-series="${key}"]`)
  const chart = wrap?.querySelector<SVGElement>('.tri-ana-svg')
  const cursor = wrap?.querySelector<SVGElement>('.tri-ana-cursor')
  const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
  if (!wrap || !chart || !cursor || !readout) return () => {}
  return scrubBind(wrap, chart, cursor, readout, count, width, text)
}

const mountDetailScrubs = (
  scope: HTMLElement,
  detail: OuraDayDetail,
  context: TriathlonContext,
): (() => void) => {
  const cleanups: (() => void)[] = []
  const phase = detail.phase5Min
  if (phase && detail.bedtimeStart) {
    const start = wallMin(detail.bedtimeStart)
    cleanups.push(
      mountSeriesScrub(scope, 'stages', phase.length, phase.length, index => {
        const stage = OURA_STAGE[phase[index]]
        return `${wallClock(start + index * 5)} · ${stage ? context.formatter.text(stage.key) : '—'}`
      }),
    )
  }
  const mountValues = (key: 'hrv' | 'hr', series: OuraSeries | null, unit: string): void => {
    if (!series) return
    const start = wallMin(series.startTs)
    cleanups.push(
      mountSeriesScrub(scope, key, series.items.length, ANA_W, index => {
        const value = series.items[index]
        const time = wallClock(start + (index * series.intervalS) / 60)
        return `${time} · ${value != null ? Math.round(value) : '—'} ${unit}`
      }),
    )
  }
  mountValues('hrv', detail.hrv, 'ms')
  mountValues('hr', detail.hr, 'bpm')
  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}

export const mountSleepPanel = (
  panel: HTMLElement,
  data: Analytics,
  context: TriathlonContext,
): (() => void) => {
  const block = panel.querySelector<HTMLElement>('.tri-ana-sleep')
  const chart = block?.querySelector<SVGElement>('.tri-sleep-svg')
  const day = block?.querySelector<HTMLElement>('.tri-sleep-day')
  const dayInner = block?.querySelector<HTMLElement>('.tri-sleep-day-inner')
  if (!block || !chart || !day || !dayInner) return () => {}
  const nights = data.recovery.series
  let live = true
  let selectedDate: string | null = null
  let detailCleanup: (() => void) | null = null
  let animationFrame: number | null = null

  const setActive = (date: string | null): void => {
    for (const bar of block.querySelectorAll<SVGElement>('[data-sleep-date]'))
      bar.classList.toggle('tri-seg--active', bar.dataset.sleepDate === date)
  }
  const clearDetail = (): void => {
    detailCleanup?.()
    detailCleanup = null
  }
  const close = (): void => {
    selectedDate = null
    setActive(null)
    day.classList.remove('tri-sleep-day--open')
    clearDetail()
  }
  const reveal = (): void => {
    if (animationFrame != null) cancelAnimationFrame(animationFrame)
    animationFrame = requestAnimationFrame(() => {
      animationFrame = null
      if (live && selectedDate && day.isConnected) day.classList.add('tri-sleep-day--open')
    })
  }
  const renderDetail = (date: string, details: Record<string, OuraDayDetail> | null): void => {
    if (!live || context.signal.aborted || selectedDate !== date || !dayInner.isConnected) return
    clearDetail()
    const detail = details?.[date]
    if (!detail) {
      dayInner.replaceChildren(
        buildSleeplessRock(context.formatter.text('no detail for this night')),
      )
      reveal()
      return
    }
    dayInner.replaceChildren(buildOuraDayDetail(context.formatter, detail))
    detailCleanup = mountDetailScrubs(dayInner, detail, context)
    reveal()
  }
  const open = (date: string): void => {
    selectedDate = date
    setActive(date)
    const path = context.root?.dataset.ouraDetailPath
    if (!path) {
      renderDetail(date, null)
      return
    }
    void context.resources.oura.load(path).then(result => {
      if (result.status === 'ready') renderDetail(date, result.value)
      else if (result.status === 'error') renderDetail(date, null)
    })
  }
  const onChartClick = (event: MouseEvent): void => {
    const bounds = chart.getBoundingClientRect()
    if (bounds.width <= 0) return
    const fraction = clampN((event.clientX - bounds.left) / bounds.width, 0, 1)
    const date = nights[Math.min(nights.length - 1, Math.floor(fraction * nights.length))]?.date
    if (!date) return
    if (selectedDate === date) close()
    else open(date)
  }
  const onBlockClick = (event: MouseEvent): void => {
    if (event.target instanceof Element && event.target.closest('.tri-sleep-day-close')) close()
  }

  chart.addEventListener('click', onChartClick)
  block.addEventListener('click', onBlockClick)
  open(data.meta.today)
  return () => {
    live = false
    chart.removeEventListener('click', onChartClick)
    block.removeEventListener('click', onBlockClick)
    if (animationFrame != null) cancelAnimationFrame(animationFrame)
    clearDetail()
  }
}
