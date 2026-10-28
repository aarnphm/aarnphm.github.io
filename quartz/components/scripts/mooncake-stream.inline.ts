import {
  LANE_X,
  LANE_H,
  markerPath,
  streamBars,
  streamContext,
  streamHit,
  streamLink,
  streamPlan,
  streamReadout,
  type Bar,
} from '../../util/mooncake-stream'
import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

const render = (root: HTMLElement) => {
  const plan = streamPlan(
    streamContext(root.dataset.mksContext),
    streamHit(root.dataset.mksHit),
    streamLink(root.dataset.mksLink),
  )
  for (const svg of root.querySelectorAll<SVGSVGElement>('[data-mks-lane-w]')) {
    const laneW = Number(svg.dataset.mksLaneW)
    const bars = streamBars(plan, laneW)
    const lanes: Record<string, Bar[]> = {
      load: bars.load,
      compute: bars.compute,
      stream: bars.stream,
      serial: bars.serial,
    }
    for (const rect of svg.querySelectorAll<SVGRectElement>('[data-mks-bar]')) {
      const [kind, i] = rect.dataset.mksBar!.split('-')
      const bar = lanes[kind]?.[Number(i)]
      if (!bar) continue
      rect.setAttribute('x', bar.x.toFixed(2))
      rect.setAttribute('width', bar.w.toFixed(2))
    }
    const marker = bars.marker.toFixed(2)
    for (const path of svg.querySelectorAll<SVGPathElement>('[data-mks-marker]')) {
      const laneYs = [...svg.querySelectorAll<SVGRectElement>('.mks-lane')]
        .slice(0, 3)
        .map(lane => Number(lane.getAttribute('y')))
      const axisY = Number(svg.querySelector('.mks-axis')!.getAttribute('y1'))
      path.setAttribute('d', markerPath(bars.marker, laneYs, LANE_H, axisY))
    }
    for (const label of svg.querySelectorAll<SVGTextElement>('[data-mks-marker-label]')) {
      label.setAttribute('x', marker)
      label.setAttribute('text-anchor', bars.marker > LANE_X + laneW / 2 ? 'end' : 'start')
    }
  }
  const values = streamReadout(plan)
  for (const el of root.querySelectorAll<Element>('[data-mks-val]')) {
    const text = values[(el as HTMLElement).dataset.mksVal!]
    if (text !== undefined && el.textContent !== text) el.textContent = text
  }
}

registerMdxContent('[data-mooncake-stream]', root => {
  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement
    if (input.matches('[data-mks-context-input]')) root.dataset.mksContext = input.value
    else if (input.matches('[data-mks-hit-input]')) root.dataset.mksHit = input.value
    else if (input.matches('[data-mks-link-input]')) root.dataset.mksLink = input.value
    else return
    render(root)
  }
  root.addEventListener('change', onChange)
  window.addCleanup(() => root.removeEventListener('change', onChange))
  window.addCleanup(observeSvgScale(root.querySelectorAll<SVGSVGElement>('.mks-graph'), '--mck-u'))
  render(root)
})
