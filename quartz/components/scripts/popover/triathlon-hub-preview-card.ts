import type { ActivityKind } from '../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { distCombined, dur } from '../../../util/triathlon-card'
import { HUB_PREVIEW_DAYS, type TriathlonHubPreview } from '../../../util/triathlon-hub-preview'
import { miniatureList, windowPreview } from '../triathlon-preview-card'

const SVG_NS = 'http://www.w3.org/2000/svg'
// Bikes cover the most ground, so they go down first and runs and swims stay readable on top.
const DRAW_ORDER: readonly ActivityKind[] = ['bike', 'walk', 'run', 'swim']

/** Every route at a low alpha on one plane: a street ridden twenty times draws twenty times darker. */
function heatFigure(preview: TriathlonHubPreview): SVGSVGElement | null {
  if (preview.routes.length === 0) return null
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.dataset.chart = 'heat'
  svg.setAttribute('viewBox', `0 0 ${preview.box.w} ${preview.box.h}`)
  svg.setAttribute('aria-hidden', 'true')
  // The route frame sizes a single trace at 78%; the sheet fills it.
  svg.style.width = '100%'
  svg.style.height = '100%'
  const routes = [...preview.routes].sort(
    (left, right) => DRAW_ORDER.indexOf(left.sport) - DRAW_ORDER.indexOf(right.sport),
  )
  for (const route of routes) {
    const path = document.createElementNS(SVG_NS, 'path')
    path.setAttribute('d', route.d)
    path.dataset.sport = route.sport
    // The frame's path rule sets a 2-unit stroke for a 100-unit trace; the sheet is drawn at
    // screen width so overlaps, not line weight, carry the heat.
    path.style.strokeWidth = '1.25px'
    path.style.vectorEffect = 'non-scaling-stroke'
    path.style.strokeOpacity = '0.4'
    svg.appendChild(path)
  }
  return svg
}

const sportTotals = (p: TriathlonPresentation, preview: TriathlonHubPreview): string[] =>
  (Object.entries(preview.km) as [ActivityKind, number][])
    .filter(([, km]) => km > 0)
    .sort((left, right) => right[1] - left[1])
    .map(([sport, km]) => `${sport} · ${distCombined(p, km)}`)

export function renderTriathlonHubPreview(
  preview: TriathlonHubPreview,
  href: string,
  p: TriathlonPresentation,
  popoverInner: HTMLElement,
) {
  const totalKm = Object.values(preview.km).reduce((sum, km) => sum + km, 0)
  const lead = (Object.entries(preview.km) as [ActivityKind, number][]).sort(
    (left, right) => right[1] - left[1],
  )[0]?.[0]
  const trace = heatFigure(preview)
  const card = windowPreview({
    href,
    title: `last ${HUB_PREVIEW_DAYS} days`,
    icon: lead ?? 'day',
    meta: [
      `${preview.sessions} sessions`,
      distCombined(p, totalKm),
      dur(preview.hours * 3600),
      ...(preview.away > 0 ? [`${preview.away} routes away`] : []),
    ],
    trace,
    body: trace ? null : miniatureList(sportTotals(p, preview)),
  })
  if (trace) card.appendChild(miniatureList(sportTotals(p, preview)))
  popoverInner.dataset.contentType = 'text/x-triathlon-hub'
  popoverInner.appendChild(card)
}
