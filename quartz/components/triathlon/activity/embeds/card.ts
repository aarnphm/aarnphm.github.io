import type { DayCardExtras } from '../../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { DetailPayload } from '../data'
import { scopeDomIds } from '../../../../util/scope-element-ids'
import {
  buildDayCard as buildDayCardNode,
  dayCardActivitiesExpanded,
} from '../../../../util/triathlon-card'
import { createDomFactory } from '../../runtime/dom'
import { setupPowerCurveTicks } from '../../shell/power-curve-ticks'
import { alignedActivitySectionMargins } from '../activity-layout'
import { setActivityExpanded, onCardToggle } from '../comparison'
import { mountDaySleepCharts } from '../day-sleep'
import { mountDayTrack } from '../day-track'
import { renderDetail } from '../render'

const ALIGNED_ACTIVITY_SECTIONS: { selector: string; margin: string }[] = [
  { selector: '.tri-act-toggle', margin: '--tri-toggle-margin' },
  { selector: '.tri-act-more > .tri-workout-analysis', margin: '--tri-workout-margin' },
  { selector: '.tri-act-more > [data-tri-trace="stamina"]', margin: '--tri-stamina-margin' },
  {
    selector: '.tri-act-more > [data-tri-trace="performance-condition"]',
    margin: '--tri-performance-condition-margin',
  },
  { selector: '.tri-act-more > .tri-training-effect', margin: '--tri-training-effect-margin' },
  {
    selector:
      '.tri-act-more > :is(.tri-zone-duo, [data-tri-trace="heart-rate-zones"], [data-tri-trace="power-zones"])',
    margin: '--tri-zones-margin',
  },
  {
    selector: '.tri-act-more > [data-tri-trace="heat-strain-index"]',
    margin: '--tri-heat-strain-index-margin',
  },
  {
    selector: '.tri-act-more > [data-tri-trace="core-temperature"]',
    margin: '--tri-core-temperature-margin',
  },
  {
    selector: '.tri-act-more > [data-tri-trace="skin-temperature"]',
    margin: '--tri-skin-temperature-margin',
  },
  { selector: '.tri-act-more > [data-tri-trace="hr"]', margin: '--tri-hr-margin' },
  {
    selector: '.tri-act-more > [data-tri-trace="temperature"]',
    margin: '--tri-temperature-margin',
  },
  { selector: '.tri-act-more > [data-tri-trace="cadence"]', margin: '--tri-cadence-margin' },
  {
    selector: '.tri-act-more > [data-tri-trace="respiration"]',
    margin: '--tri-respiration-margin',
  },
  { selector: '.tri-act-more > .tri-environment', margin: '--tri-environment-margin' },
]

// Multi-activity embeds hold their activities in a paging track.
const cardActivitySelector = (rest = ''): string =>
  `:scope > .tri-act${rest}, :scope > .tri-day-track > .tri-act${rest}`

export const mountActivitySectionAlignment = (card: HTMLElement): (() => void) => {
  if (card.querySelectorAll(cardActivitySelector()).length < 2) return () => {}
  let frame = 0
  const align = (): void => {
    frame = 0
    const terrainCaptions: { chart: HTMLElement; activityTop: number; contentHeight: number }[] = []
    for (const chart of card.querySelectorAll<HTMLElement>(
      cardActivitySelector(' > .tri-act-more > .tri-cycling-power'),
    )) {
      const activity = chart.closest<HTMLElement>('.tri-act')
      const caption = chart.querySelector<HTMLElement>(':scope > .tri-elev-cap')
      if (!activity || !caption || chart.getClientRects().length === 0) {
        chart.style.removeProperty('--tri-terrain-caption-height')
        continue
      }
      const top = caption.getBoundingClientRect().top
      const contentHeight = Math.max(
        0,
        ...Array.from(caption.children)
          .filter(child => window.getComputedStyle(child).position !== 'absolute')
          .map(child => child.getBoundingClientRect().bottom - top),
      )
      terrainCaptions.push({
        chart,
        activityTop: activity.getBoundingClientRect().top,
        contentHeight,
      })
    }
    for (const target of terrainCaptions) {
      const height = Math.ceil(
        Math.max(
          ...terrainCaptions
            .filter(other => Math.abs(other.activityTop - target.activityTop) < 1)
            .map(other => other.contentHeight),
        ),
      )
      const current = Number.parseFloat(
        target.chart.style.getPropertyValue('--tri-terrain-caption-height'),
      )
      if (current !== height)
        target.chart.style.setProperty('--tri-terrain-caption-height', `${height}px`)
    }
    for (const { selector, margin: marginProperty } of ALIGNED_ACTIVITY_SECTIONS) {
      const effects = Array.from(
        card.querySelectorAll<HTMLElement>(cardActivitySelector(` > ${selector}`)),
      )
      const targets: {
        effect: HTMLElement
        activityTop: number
        sectionTop: number
        marginTop: number
      }[] = []
      for (const effect of effects) {
        const activity = effect.closest<HTMLElement>('.tri-act')
        if (!activity || effect.getClientRects().length === 0) {
          effect.style.removeProperty(marginProperty)
          continue
        }
        const marginTop = Number.parseFloat(window.getComputedStyle(effect).marginTop)
        targets.push({
          effect,
          activityTop: activity.getBoundingClientRect().top,
          sectionTop: effect.getBoundingClientRect().top,
          marginTop: Number.isFinite(marginTop) ? marginTop : 0,
        })
      }
      const margins = alignedActivitySectionMargins(targets)
      for (const [index, target] of targets.entries()) {
        const margin = margins[index]
        const current = Number.parseFloat(target.effect.style.getPropertyValue(marginProperty))
        if (!Number.isFinite(current) || Math.abs(current - margin) >= 0.1)
          target.effect.style.setProperty(marginProperty, `${margin}px`)
      }
    }
  }
  const schedule = (): void => {
    if (frame !== 0) window.cancelAnimationFrame(frame)
    frame = window.requestAnimationFrame(align)
  }
  const resize = new ResizeObserver(schedule)
  resize.observe(card)
  for (const legend of card.querySelectorAll('.tri-cycling-power-legend')) resize.observe(legend)
  card.addEventListener('click', schedule)
  schedule()
  return () => {
    if (frame !== 0) window.cancelAnimationFrame(frame)
    resize.disconnect()
    card.removeEventListener('click', schedule)
    for (const chart of card.querySelectorAll<HTMLElement>('.tri-cycling-power'))
      chart.style.removeProperty('--tri-terrain-caption-height')
    for (const { selector, margin } of ALIGNED_ACTIVITY_SECTIONS)
      card
        .querySelectorAll<HTMLElement>(cardActivitySelector(` > ${selector}`))
        .forEach(section => section.style.removeProperty(margin))
  }
}

export const buildDayCard = (
  presentation: TriathlonPresentation,
  dateIso: string,
  payload: DetailPayload | null,
  extras: DayCardExtras = {},
): { element: HTMLElement; mount: () => () => void } => {
  const domF = createDomFactory(presentation)
  const activityViews: ReturnType<typeof renderDetail>[] = []
  const card = buildDayCardNode(domF, dateIso, payload, extras, (detail, reserveFueling) => {
    const view = renderDetail(
      presentation,
      detail,
      payload,
      extras.event != null,
      extras.embedded === true,
      extras.dayRouteHref,
      extras.settings,
      reserveFueling,
    )
    activityViews.push(view)
    return view.element
  }) as HTMLElement
  if (extras.occurrence) scopeDomIds(card, extras.occurrence)
  if (dayCardActivitiesExpanded(extras)) {
    card
      .querySelectorAll<HTMLElement>('.tri-act')
      .forEach(activity => setActivityExpanded(activity, true))
  }
  return {
    element: card,
    mount: () => {
      card.addEventListener('click', onCardToggle)
      const cleanups = [
        setupPowerCurveTicks(card),
        mountActivitySectionAlignment(card),
        mountDayTrack(card, presentation),
        mountDaySleepCharts(card, () => presentation.locale),
        ...activityViews.map(view => view.mount()),
      ]
      return () => {
        card.removeEventListener('click', onCardToggle)
        for (const cleanup of cleanups) cleanup()
      }
    },
  }
}
