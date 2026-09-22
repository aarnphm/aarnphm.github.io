import type { Analytics } from '../../../plugins/stores/analytics'
import type { OuraDayDetail } from '../../../plugins/stores/oura'
import type { TriathlonContext } from '../runtime/context'
import { ouraRestorationBaseline } from '../../../util/oura-health'
import { daySleepBarIndex, mountDaySleepCharts, setSleepView } from '../activity/day-sleep'
import { buildSleepDayDetail, buildSleeplessRock } from './panels/recovery'
import { clampN } from './shared'

export const mountSleepPanel = (
  panel: HTMLElement,
  data: Analytics,
  context: TriathlonContext,
): (() => void) => {
  const block = panel.querySelector<HTMLElement>('.tri-ana-sleep')
  const charts = block?.querySelectorAll<SVGElement>('.tri-sleep-svg')
  const day = block?.querySelector<HTMLElement>('.tri-sleep-day')
  const dayInner = block?.querySelector<HTMLElement>('.tri-sleep-day-inner')
  if (!block || !charts || !day || !dayInner) return () => {}
  const nights = data.recovery.series
  const supplementalByDate = new Map(data.daily.map(day => [day.date, day.sleepMetrics]))
  const healthByDate = new Map(data.daily.map(day => [day.date, day.garminHealth]))
  let live = true
  let selectedDate: string | null = null
  let selectedView: 'night' | 'naps' = 'night'
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
    const supplemental = supplementalByDate.get(date) ?? null
    const health = healthByDate.get(date) ?? null
    if (!detail && !supplemental && !health) {
      dayInner.replaceChildren(
        buildSleeplessRock(context.formatter.text('no detail for this night')),
      )
      reveal()
      return
    }
    dayInner.replaceChildren(
      buildSleepDayDetail(
        context.formatter,
        date,
        detail ?? null,
        supplemental,
        health,
        ouraRestorationBaseline(date, details ?? {}),
      ),
    )
    detailCleanup = mountDaySleepCharts(dayInner, () => context.presentation.locale)
    const views = dayInner.querySelector<HTMLElement>('[data-sleep-views]')
    if (views) setSleepView(views, selectedView)
    reveal()
  }
  const open = (date: string): void => {
    selectedDate = date
    setActive(date)
    if (supplementalByDate.get(date) || healthByDate.get(date)) renderDetail(date, null)
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
    const chart = event.currentTarget
    if (!(chart instanceof SVGElement)) return
    const bounds = chart.getBoundingClientRect()
    if (bounds.width <= 0) return
    const fraction = clampN((event.clientX - bounds.left) / bounds.width, 0, 1)
    const date = nights[daySleepBarIndex(fraction, nights.length)]?.date
    if (!date) return
    if (selectedDate === date) close()
    else open(date)
  }
  const onBlockClick = (event: MouseEvent): void => {
    if (event.target instanceof Element && event.target.closest('.tri-sleep-day-close')) close()
    const button =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('button[data-sleep-view]')
        : null
    const mode = button?.dataset.sleepView
    if (mode !== 'night' && mode !== 'naps') return
    selectedView = mode
    for (const views of block.querySelectorAll<HTMLElement>('[data-sleep-views]'))
      setSleepView(views, mode)
  }

  for (const chart of charts) chart.addEventListener('click', onChartClick)
  block.addEventListener('click', onBlockClick)
  open(data.meta.today)
  return () => {
    live = false
    for (const chart of charts) chart.removeEventListener('click', onChartClick)
    block.removeEventListener('click', onBlockClick)
    if (animationFrame != null) cancelAnimationFrame(animationFrame)
    clearDetail()
  }
}
