import { svgPathGeometry } from '../../util/svg-geometry'

type Bounds = Pick<DOMRect, 'left' | 'right' | 'top' | 'bottom'>
type RouteSample = { distance: number; x: number; y: number }

const gap = 12

function overlap(bounds: Bounds, other: Bounds) {
  const width = Math.max(
    0,
    Math.min(bounds.right, other.right + gap) - Math.max(bounds.left, other.left - gap),
  )
  const height = Math.max(
    0,
    Math.min(bounds.bottom, other.bottom + gap) - Math.max(bounds.top, other.top - gap),
  )
  return width * height
}

function placePath(hint: SVGPathElement, samples: RouteSample[], end: number) {
  const points = samples.filter(
    point => point.distance >= Math.max(0, end - 140) && point.distance <= end,
  )
  const start = points[0].distance
  end = points[points.length - 1].distance
  const reverse = points[points.length - 1].x < points[0].x
  if (reverse) points.reverse()
  hint.setAttribute(
    'd',
    points
      .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
      .join(''),
  )
  hint.dataset.start = String(start)
  hint.dataset.end = String(end)
  hint.dataset.reverse = String(reverse)
}

function readableText(text: SVGTextElement) {
  const centers: { x: number; y: number }[] = []
  for (let index = 0; index < text.getNumberOfChars(); index++) {
    const rect = text.getExtentOfChar(index)
    if (!rect.width || Math.abs(text.getRotationOfChar(index)) > 65) return false
    const center = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
    if (centers.slice(0, -1).some(other => Math.hypot(center.x - other.x, center.y - other.y) <= 5))
      return false
    centers.push(center)
  }
  return true
}

export function placeCampHints(scene: HTMLElement) {
  if (scene.dataset.campTools !== 'open') return
  const canvas = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!canvas) return
  const bounds = canvas.getBoundingClientRect()
  const obstacles: Bounds[] = Array.from(
    scene.querySelectorAll(
      '.trail-camp-toggle, .trail-tree-toggle, .trail-tree-hint, [data-camp-solid], [data-trail-collision]',
    ),
  )
    .filter(
      element =>
        scene.ownerDocument.defaultView?.getComputedStyle(element).visibility === 'visible',
    )
    .map(element => element.getBoundingClientRect())
    .filter(rect => rect.width > 0 && rect.height > 0)
  const landing = scene.ownerDocument.querySelector('.landing')
  if (landing) obstacles.push(landing.getBoundingClientRect())

  for (const hint of canvas.querySelectorAll<SVGPathElement>('[data-camp-hint-route]')) {
    const route = canvas.querySelector<SVGPathElement>(
      `[data-trail-route="${hint.dataset.campHintRoute}"]`,
    )
    const text = canvas
      .querySelector<SVGTextPathElement>(`textPath[href="#${hint.id}"]`)
      ?.closest<SVGTextElement>('text')
    if (!route?.hasAttribute('d') || !text) continue
    const geometry = svgPathGeometry(route)
    const nearestEnd = Math.max(0, geometry.getTotalLength() - 24)
    const earliestStart = Math.max(0, nearestEnd - 460)
    const steps = Math.max(1, Math.ceil(nearestEnd - earliestStart))
    const samples = Array.from({ length: steps + 1 }, (_, index) => {
      const distance = earliestStart + ((nearestEnd - earliestStart) * index) / steps
      const point = geometry.getPointAtLength(distance)
      return { distance, x: point.x, y: point.y }
    })
    let bestEnd = nearestEnd
    let bestOffset = 0
    let bestScore = Infinity
    const distance = Math.min(320, Math.max(0, nearestEnd - 140))
    const candidates = Array.from({ length: Math.ceil(distance / 16) + 1 }, (_, index) =>
      Math.max(140, nearestEnd - index * 16),
    )
      .flatMap(end =>
        [0, -16, 16, -32, 32, -48, 48, -64, 64, -80, 80, -96, 96].map(offset => ({
          end,
          offset,
          distance: nearestEnd - end + Math.abs(offset),
        })),
      )
      .sort((left, right) => left.distance - right.distance)
    // Prefer the closest clear position along or beside the same trail.
    for (const { end, offset } of candidates) {
      text.setAttribute('transform', `translate(0 ${offset})`)
      placePath(hint, samples, end)
      const rect = text.getBoundingClientRect()
      const fits =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.left >= bounds.left + gap &&
        rect.right <= bounds.right - gap &&
        rect.top >= bounds.top + gap &&
        rect.bottom <= bounds.bottom - gap &&
        text.getComputedTextLength() + gap * 2 <= svgPathGeometry(hint).getTotalLength() &&
        readableText(text)
      const score = fits
        ? obstacles.reduce((sum, other) => sum + overlap(rect, other), 0)
        : Infinity
      if (score < bestScore) {
        bestScore = score
        bestEnd = end
        bestOffset = offset
      }
      if (bestScore === 0) break
    }
    text.setAttribute('transform', `translate(0 ${bestOffset})`)
    placePath(hint, samples, bestEnd)
    obstacles.push(text.getBoundingClientRect())
  }
}
