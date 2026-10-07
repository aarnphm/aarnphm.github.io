import type { BestEffortCategory, BestEffortEntry } from '../../../../util/best-efforts'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import type { BestEffortDisplay } from './best-efforts-format'
import { axisFrame, type AxisXTick, type AxisYTick } from '../../../../util/triathlon-card'
import { powerCurveActivityLinkAttributes } from '../../../../util/triathlon-power-activity'
import { createDomFactory, el, svg } from '../../runtime/dom'
import { clampN } from '../shared'
import {
  bestEffortAxisLabel,
  bestEffortCategoryLabel,
  bestEffortDisplay,
  buildBestEffortRank,
  isPodiumRank,
} from './best-efforts-format'

export interface BestEffortsChartOptions {
  category: BestEffortCategory
  context: TriathlonContext
  /** Calendar year to emphasise; efforts from other years render muted. null keeps every year at full weight. */
  year: number | null
  /** Right edge of the x axis (YYYY-MM-DD), normally `data.meta.today`. The left edge is the first effort. */
  today: string
  /** Inclusive local dates selected by the whole-ride filter. */
  range?: { from: string; to: string }
}

export interface BestEffortsChartView {
  element: HTMLElement
  mount: () => () => void
}

// The SVG stretches to a fixed CSS height (triathlon-best-efforts-chart.scss), so one vertical
// viewBox unit is 1% of the plot height. Y_BEST leaves room for a rank box hung above the best
// effort; the x inset keeps edge dots and rank boxes inside the analytics block's paint containment.
const W = 100
const H = 100
const X_LEFT = 3
const X_RIGHT = 97
const Y_BEST = 15
const Y_WORST = 91
// Efforts past the outlier fence sit in a lane under the worst tick instead of stretching the axis.
const Y_FLOOR = 97
// Quartiles need enough efforts to mean anything.
const FENCE_MIN_EFFORTS = 8
const DAY_MS = 86_400_000
// Same-day efforts closer than this (viewBox units) would hide each other; nudge them apart in px.
const OVERLAP_UNITS = 1
const NUDGE_PX = 5
const MOUSE_RADIUS_PX = 28
const TOUCH_RADIUS_PX = 40
const MONTH_STEPS = [1, 2, 3, 6, 12]

interface ChartPoint {
  entry: BestEffortEntry
  display: BestEffortDisplay
  /** viewBox units, equal to percent of the plot box. */
  x: number
  y: number
  /** Horizontal pixel nudge separating overlapping same-day efforts. */
  nudge: number
  /** Past the outlier fence: drawn in the floor lane, value still exact in the readout. */
  floored: boolean
}

const quantile = (sorted: readonly number[], q: number): number => {
  const position = (sorted.length - 1) * q
  const low = Math.floor(position)
  const high = Math.ceil(position)
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low)
}

/** Tukey fence on the worse side: a walk logged as a 5K would otherwise squeeze every real run into a sliver. */
const worseFence = (category: BestEffortCategory): number | null => {
  if (category.efforts.length < FENCE_MIN_EFFORTS) return null
  const sorted = category.efforts.map(entry => entry.value).sort((a, b) => a - b)
  const q1 = quantile(sorted, 0.25)
  const q3 = quantile(sorted, 0.75)
  const spread = 1.5 * (q3 - q1)
  return category.better === 'lower' ? q3 + spread : q1 - spread
}

const utcDay = (iso: string): number => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!match) return Number.NaN
  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])) / DAY_MS
}

const entryYear = (entry: BestEffortEntry): number => Number(entry.date.slice(0, 4))

const monthIso = (year: number, month: number): string =>
  `${year}-${String(month + 1).padStart(2, '0')}-01`

/** Calendar-aligned month ticks (January shows the year); the caption carries the exact range. */
const timeTicks = (
  formatter: TriathlonFormatter,
  firstDay: number,
  lastDay: number,
  x: (day: number) => number,
): AxisXTick[] => {
  if (lastDay <= firstDay) return []
  const months = (lastDay - firstDay) / 30.44
  const step = MONTH_STEPS.find(candidate => months / candidate <= 6) ?? 12 * Math.ceil(months / 72)
  const start = new Date(firstDay * DAY_MS)
  let year = start.getUTCFullYear()
  let month = start.getUTCMonth() + 1
  const ticks: AxisXTick[] = []
  for (;;) {
    if (month > 11) {
      year += 1
      month = 0
    }
    const day = Date.UTC(year, month, 1) / DAY_MS
    if (day > lastDay) break
    const aligned = step <= 12 ? month % step === 0 : month === 0 && year % (step / 12) === 0
    if (aligned) {
      const pct = x(day)
      ticks.push({
        label: month === 0 ? String(year) : formatter.month(monthIso(year, month)),
        pct,
        cls: pct < 5 ? 'tri-cax-xt--first' : pct > 95 ? 'tri-cax-xt--last' : undefined,
      })
    }
    month += 1
  }
  return ticks
}

const rankText = (
  entry: BestEffortEntry,
  formatter: TriathlonFormatter,
  separator: string,
): string =>
  `${formatter.text('all-time rank')} #${entry.rank}${separator}${entryYear(entry)} #${entry.yearRank}`

const describe = (point: ChartPoint, formatter: TriathlonFormatter): string => {
  const { entry, display } = point
  const parts = [formatter.longDate(entry.date), display.primary]
  if (display.secondary) parts.push(display.secondary)
  parts.push(rankText(entry, formatter, ', '))
  if (entry.pr) parts.push(formatter.text('PR'))
  if (entry.indoor) parts.push(formatter.text('indoor'))
  if (entry.name) parts.push(entry.name)
  return parts.join(', ')
}

const buildTip = (point: ChartPoint, formatter: TriathlonFormatter): HTMLElement[] => {
  const { entry, display } = point
  const head = el(
    'span',
    'tri-gloss-h',
    entry.indoor
      ? `${formatter.longDate(entry.date)} · ${formatter.text('indoor')}`
      : formatter.longDate(entry.date),
  )
  const value = el('span', 'tri-be-chart-tip-value')
  value.appendChild(el('strong', undefined, display.primary))
  if (display.secondary)
    value.appendChild(el('span', 'tri-be-chart-tip-secondary', display.secondary))
  const rank = el(
    'span',
    'tri-gloss-def',
    entry.pr
      ? `${rankText(entry, formatter, ' · ')} · ${formatter.text('PR')}`
      : rankText(entry, formatter, ' · '),
  )
  const parts = [head, value, rank]
  if (entry.name) parts.push(el('span', 'tri-gloss-def tri-be-chart-tip-name', entry.name))
  return parts
}

export const buildBestEffortsChart = ({
  category,
  context,
  year,
  today,
  range: dateRange,
}: BestEffortsChartOptions): BestEffortsChartView => {
  const formatter = context.formatter
  const efforts = category.efforts
  const figure = el('figure', 'tri-be-chart', undefined, { 'data-category': category.key })
  if (year != null) figure.dataset.year = String(year)
  const firstDay = efforts.length > 0 ? utcDay(dateRange?.from ?? efforts[0].date) : Number.NaN
  if (!Number.isFinite(firstDay)) {
    figure.appendChild(el('div', 'tri-ana-empty', formatter.text('not enough data')))
    return { element: figure, mount: () => () => {} }
  }

  const todayDay = utcDay(dateRange?.to ?? today)
  const lastDay = Math.max(
    firstDay,
    Number.isFinite(todayDay) ? todayDay : utcDay(efforts[efforts.length - 1].date),
  )
  const span = lastDay - firstDay
  const x = (day: number): number =>
    span === 0 ? W / 2 : X_LEFT + clampN((day - firstDay) / span, 0, 1) * (X_RIGHT - X_LEFT)

  const worse = (a: number, b: number): boolean => (category.better === 'lower' ? a > b : a < b)
  const fence = worseFence(category)
  const floored = (value: number): boolean => fence != null && worse(value, fence)
  let best = efforts[0].value
  let worst: number | null = null
  for (const entry of efforts) {
    if (worse(best, entry.value)) best = entry.value
    if (!floored(entry.value) && (worst == null || worse(entry.value, worst))) worst = entry.value
  }
  worst ??= best
  const range = worst - best
  const y = (value: number): number =>
    floored(value)
      ? Y_FLOOR
      : range === 0
        ? (Y_BEST + Y_WORST) / 2
        : Y_BEST + ((value - best) / range) * (Y_WORST - Y_BEST)

  const placedByDate = new Map<string, number[]>()
  const points: ChartPoint[] = efforts.map(entry => {
    const point: ChartPoint = {
      entry,
      display: bestEffortDisplay(category, entry, formatter),
      x: x(utcDay(entry.date)),
      y: y(entry.value),
      nudge: 0,
      floored: floored(entry.value),
    }
    const placed = placedByDate.get(entry.date) ?? []
    // PR dots stay on their staircase corner; later same-day duplicates step aside, alternating sides.
    const overlaps = placed.filter(other => Math.abs(other - point.y) < OVERLAP_UNITS).length
    if (overlaps > 0 && !entry.pr)
      point.nudge = (overlaps % 2 === 1 ? 1 : -1) * Math.ceil(overlaps / 2) * NUDGE_PX
    placed.push(point.y)
    placedByDate.set(entry.date, placed)
    return point
  })

  const plot = svg('svg', {
    class: 'tri-be-chart-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  if (year != null && span > 0) {
    const yearStart = Date.UTC(year, 0, 1) / DAY_MS
    const yearEnd = Date.UTC(year + 1, 0, 1) / DAY_MS
    if (yearEnd > firstDay && yearStart <= lastDay) {
      const left = yearStart <= firstDay ? 0 : x(yearStart)
      const right = yearEnd > lastDay ? W : x(yearEnd)
      plot.appendChild(
        svg('rect', {
          class: 'tri-be-chart-year',
          x: left.toFixed(2),
          y: 0,
          width: (right - left).toFixed(2),
          height: H,
        }),
      )
    }
  }

  const yTicks: AxisYTick[] =
    range === 0
      ? [{ label: bestEffortAxisLabel(category, best, formatter), vbY: y(best) }]
      : [best, (best + worst) / 2, worst].map(value => ({
          label: bestEffortAxisLabel(category, value, formatter),
          vbY: y(value),
        }))
  for (const tick of yTicks)
    plot.appendChild(
      svg('line', {
        class: 'tri-be-chart-grid',
        x1: 0,
        x2: W,
        y1: tick.vbY.toFixed(2),
        y2: tick.vbY.toFixed(2),
      }),
    )

  const prs = points.filter(point => point.entry.pr)
  if (prs.length > 0) {
    const commands = [`M ${prs[0].x.toFixed(2)} ${prs[0].y.toFixed(2)}`]
    for (const point of prs.slice(1))
      commands.push(`H ${point.x.toFixed(2)} V ${point.y.toFixed(2)}`)
    commands.push(`H ${x(lastDay).toFixed(2)}`)
    plot.appendChild(svg('path', { class: 'tri-be-chart-best', d: commands.join(' ') }))
  }

  const categoryLabel = bestEffortCategoryLabel(category, formatter)
  const rangeText = `${formatter.longDate(dateRange?.from ?? efforts[0].date)} – ${
    !dateRange || dateRange.to === today
      ? formatter.text('today')
      : formatter.longDate(dateRange.to)
  }`
  const goldIndex = points.findIndex(point => point.entry.rank === 1)
  const initialIndex = goldIndex >= 0 ? goldIndex : points.length - 1

  const pointsLayer = el('div', 'tri-be-chart-points', undefined, {
    role: 'group',
    'aria-label': `${categoryLabel}, ${rangeText}`,
  })
  const links = points.map((point, index) => {
    const classes = ['tri-be-chart-dot']
    if (point.entry.pr) classes.push('tri-be-chart-dot--pr')
    if (year != null && entryYear(point.entry) !== year) classes.push('tri-be-chart-dot--other')
    if (point.floored) classes.push('tri-be-chart-dot--floor')
    const left =
      point.nudge === 0
        ? `${point.x.toFixed(3)}%`
        : `calc(${point.x.toFixed(3)}% + ${point.nudge}px)`
    return el('a', classes.join(' '), undefined, {
      ...powerCurveActivityLinkAttributes({
        activityId: point.entry.id,
        activityDate: point.entry.date,
      }),
      tabindex: index === initialIndex ? '0' : '-1',
      'aria-label': describe(point, formatter),
      'data-effort-index': String(index),
      style: `left:${left};top:${point.y.toFixed(3)}%`,
    })
  })
  pointsLayer.append(...links)

  const ranksLayer = el('div', 'tri-be-chart-ranks', undefined, { 'aria-hidden': 'true' })
  const rankPins: HTMLElement[] = []
  const podium = points
    .map((point, index) => ({ point, index }))
    .filter(({ point }) => isPodiumRank(point.entry.rank))
    // The PR paints last so it sits above #2 and #3 when they crowd.
    .sort((a, b) => b.point.entry.rank - a.point.entry.rank)
  for (const { point, index } of podium) {
    const pin = el('span', 'tri-be-chart-rank', undefined, {
      'data-effort-index': String(index),
      style: `left:calc(${point.x.toFixed(3)}% + ${point.nudge}px);top:${point.y.toFixed(3)}%`,
    })
    pin.appendChild(buildBestEffortRank(point.entry.rank, formatter))
    ranksLayer.appendChild(pin)
    rankPins.unshift(pin)
  }

  const frame = axisFrame(
    createDomFactory(context.presentation),
    plot,
    yTicks,
    H,
    timeTicks(formatter, firstDay, lastDay, x),
    true,
    { top: Y_BEST, bottom: H },
    [pointsLayer, ranksLayer],
  ) as HTMLElement
  const stage = frame.querySelector<HTMLElement>('.tri-cax-stage')
  stage?.setAttribute('data-site-cursor-crosshair', '')

  const caption = el('figcaption', 'tri-be-chart-caption')
  caption.appendChild(el('span', 'tri-be-chart-range', rangeText))
  if (prs.length > 0 && span > 0) {
    const key = el('span', 'tri-be-chart-key', undefined, { 'aria-hidden': 'true' })
    key.append(
      el('span', 'tri-be-chart-swatch'),
      el('span', undefined, formatter.text('best to date')),
    )
    caption.appendChild(key)
  }
  figure.append(frame, caption)

  return {
    element: figure,
    mount: () => {
      if (!stage) return () => {}
      const tip = el('div', 'tri-gloss tri-be-chart-tip', undefined, {
        role: 'tooltip',
        'aria-hidden': 'true',
      })
      // Analytics blocks use content-visibility, which would trap a fixed tooltip inside the block.
      const host = context.root ?? document.body
      host.appendChild(tip)

      let active = initialIndex
      let hovered: number | null = null
      let pinned: number | null = null
      let focused: number | null = null
      let shown: number | null = null
      let pointerType = 'mouse'

      const clientPoint = (index: number, rect: DOMRect): { x: number; y: number } => ({
        x: rect.left + (points[index].x / W) * rect.width + points[index].nudge,
        y: rect.top + (points[index].y / H) * rect.height,
      })
      const nearest = (clientX: number, clientY: number, radius: number): number | null => {
        const rect = stage.getBoundingClientRect()
        if (rect.width <= 0 || rect.height <= 0) return null
        let found: number | null = null
        let bestDistance = radius * radius
        for (let index = 0; index < points.length; index++) {
          const point = clientPoint(index, rect)
          const distance = (point.x - clientX) ** 2 + (point.y - clientY) ** 2
          // `<=` lets the later, top-painted dot win exact ties.
          if (distance <= bestDistance) {
            found = index
            bestDistance = distance
          }
        }
        return found
      }
      const place = (): void => {
        if (shown == null) return
        const point = clientPoint(shown, stage.getBoundingClientRect())
        const box = tip.getBoundingClientRect()
        const gap = 12
        const inset = 8
        const left = clampN(
          point.x - box.width / 2,
          inset,
          Math.max(inset, window.innerWidth - box.width - inset),
        )
        const above = point.y - gap - box.height
        const top =
          above >= inset
            ? above
            : clampN(point.y + gap, inset, Math.max(inset, window.innerHeight - box.height - inset))
        tip.style.left = `${left.toFixed(0)}px`
        tip.style.top = `${top.toFixed(0)}px`
      }
      const sync = (): void => {
        const next = hovered ?? pinned ?? focused
        if (next !== shown) {
          if (shown != null) links[shown].classList.remove('tri-be-chart-dot--hot')
          shown = next
          if (next != null) {
            links[next].classList.add('tri-be-chart-dot--hot')
            tip.replaceChildren(...buildTip(points[next], formatter))
            place()
          }
        }
        tip.classList.toggle('tri-gloss--on', shown != null)
        figure.classList.toggle('tri-be-chart--hot', shown != null)
      }
      const setActive = (index: number): void => {
        if (index === active) return
        links[active].tabIndex = -1
        links[index].tabIndex = 0
        active = index
      }
      const indexOf = (target: EventTarget | null): number | null => {
        if (!(target instanceof Element)) return null
        const link = target.closest<HTMLElement>('.tri-be-chart-dot')
        const index = Number(link?.dataset.effortIndex)
        return link && Number.isInteger(index) ? index : null
      }

      const onPointerDown = (event: PointerEvent): void => {
        pointerType = event.pointerType
      }
      const onPointerMove = (event: PointerEvent): void => {
        if (event.pointerType !== 'mouse') return
        hovered = nearest(event.clientX, event.clientY, MOUSE_RADIUS_PX)
        sync()
      }
      const onPointerLeave = (): void => {
        hovered = null
        sync()
      }
      const onClick = (event: MouseEvent): void => {
        // Keyboard activation and the forwarded click below land on the link itself.
        if (indexOf(event.target) != null) return
        const touch = pointerType !== 'mouse'
        const index = nearest(
          event.clientX,
          event.clientY,
          touch ? TOUCH_RADIUS_PX : MOUSE_RADIUS_PX,
        )
        if (index == null) {
          pinned = null
          sync()
          return
        }
        // Touch has no hover: the first tap shows the readout, a second tap opens the activity.
        if (touch && pinned !== index) {
          pinned = index
          sync()
          return
        }
        pinned = null
        sync()
        links[index].click()
      }
      const onFocusIn = (event: FocusEvent): void => {
        const index = indexOf(event.target)
        if (index == null) return
        setActive(index)
        focused = index
        hovered = null
        pinned = null
        sync()
      }
      const onFocusOut = (): void => {
        queueMicrotask(() => {
          if (figure.contains(document.activeElement)) return
          focused = null
          sync()
        })
      }
      const onKey = (event: KeyboardEvent): void => {
        const index = indexOf(event.target)
        if (index == null) return
        let next: number | null = null
        if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = index - 1
        else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = index + 1
        else if (event.key === 'Home') next = 0
        else if (event.key === 'End') next = links.length - 1
        else if (event.key === 'Escape') {
          event.preventDefault()
          links[index].blur()
          return
        }
        if (next == null) return
        event.preventDefault()
        links[clampN(next, 0, links.length - 1)].focus()
      }
      const onDocumentPointerDown = (event: PointerEvent): void => {
        if (pinned == null || (event.target instanceof Node && figure.contains(event.target)))
          return
        pinned = null
        sync()
      }
      const onViewport = (): void => place()
      // Top-3 efforts often land within days of each other; rank boxes that would cover one another
      // step sideways in rank order, PR first, so every box stays visible.
      const layoutRanks = (): void => {
        if (rankPins.length < 2) return
        const bounds = stage.getBoundingClientRect()
        if (bounds.width <= 0) return
        for (const pin of rankPins) pin.style.removeProperty('--tri-be-chart-rank-shift')
        const boxes = rankPins.map(pin => pin.getBoundingClientRect())
        const placed: { left: number; right: number; top: number; bottom: number }[] = []
        rankPins.forEach((pin, rankIndex) => {
          const box = boxes[rankIndex]
          const clear = (shift: number): boolean =>
            placed.every(
              other =>
                box.top >= other.bottom ||
                box.bottom <= other.top ||
                box.left + shift >= other.right ||
                box.right + shift <= other.left,
            )
          const inside = (shift: number): boolean =>
            box.left + shift >= bounds.left && box.right + shift <= bounds.right
          // Candidates: where it hangs, or flush beside any box already placed.
          const candidates = [
            0,
            ...placed.flatMap(other => [other.left - box.right - 1, other.right - box.left + 1]),
          ]
            .filter(clear)
            .sort((a, b) => Math.abs(a) - Math.abs(b))
          const shift = candidates.find(inside) ?? candidates[0] ?? 0
          if (shift !== 0)
            pin.style.setProperty('--tri-be-chart-rank-shift', `${shift.toFixed(1)}px`)
          placed.push({
            left: box.left + shift,
            right: box.right + shift,
            top: box.top,
            bottom: box.bottom,
          })
        })
      }
      const resize = new ResizeObserver(() => {
        layoutRanks()
        place()
      })
      resize.observe(stage)

      stage.addEventListener('pointerdown', onPointerDown)
      stage.addEventListener('pointermove', onPointerMove)
      stage.addEventListener('pointerleave', onPointerLeave)
      stage.addEventListener('click', onClick)
      figure.addEventListener('focusin', onFocusIn)
      figure.addEventListener('focusout', onFocusOut)
      figure.addEventListener('keydown', onKey)
      document.addEventListener('pointerdown', onDocumentPointerDown)
      window.addEventListener('scroll', onViewport, { capture: true, passive: true })
      window.addEventListener('resize', onViewport, { passive: true })
      return () => {
        stage.removeEventListener('pointerdown', onPointerDown)
        stage.removeEventListener('pointermove', onPointerMove)
        stage.removeEventListener('pointerleave', onPointerLeave)
        stage.removeEventListener('click', onClick)
        figure.removeEventListener('focusin', onFocusIn)
        figure.removeEventListener('focusout', onFocusOut)
        figure.removeEventListener('keydown', onKey)
        document.removeEventListener('pointerdown', onDocumentPointerDown)
        window.removeEventListener('scroll', onViewport, { capture: true })
        window.removeEventListener('resize', onViewport)
        resize.disconnect()
        if (shown != null) links[shown].classList.remove('tri-be-chart-dot--hot')
        figure.classList.remove('tri-be-chart--hot')
        tip.remove()
      }
    },
  }
}
