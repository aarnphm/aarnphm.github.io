export type Camera = { x: number; y: number; scale: number }
export type LiftGeometry = {
  x: number
  y: number
  width: number
  height: number
  px: number
  py: number
}

export const PULL_MS = 280

export type Drag = {
  id: number
  x: number
  y: number
  camera: Camera
  book: number
  moved: boolean
}

export function readBookBounds(book: SVGGElement): DOMRect {
  const candidate = book.lastElementChild?.previousElementSibling
  const edge = candidate?.classList.contains('bc-edge') ? candidate : book.querySelector('.bc-edge')
  const points = edge?.getAttribute('points')
  if (points) {
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const pair of points.trim().split(/\s+/)) {
      const comma = pair.indexOf(',')
      if (comma < 0) continue
      const x = Number(pair.slice(0, comma))
      const y = Number(pair.slice(comma + 1))
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    if (Number.isFinite(minX) && Number.isFinite(minY)) {
      return new DOMRect(minX, minY, maxX - minX, maxY - minY)
    }
  }
  return book.getBBox()
}
