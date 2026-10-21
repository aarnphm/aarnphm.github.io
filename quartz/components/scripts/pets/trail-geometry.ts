import type { Pond } from '../../visitor-trail/landscape'
import { svgPathGeometry } from '../../../util/svg-geometry'
import { easeTrailDepth } from '../../../util/trail-depth'
import { trailHost, trailScene } from '../../../util/trail-scene'
import { clamp, petState, trailScreenMatrix, type Point, type TrailFoot } from './actor-model'

type ScaleStop = readonly [number, number]
type PetTerrain = {
  top: number
  base: number
  treeline: number
  meadowTop: number
  pond: Pond | null
}
export const trailData = (value: string | undefined): unknown => {
  if (!value) return null
  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
export const pair = (value: unknown): value is ScaleStop =>
  Array.isArray(value) && value.length === 2 && finite(value[0]) && finite(value[1])
export const petTerrain = (path: SVGPathElement): PetTerrain | null => {
  const raw =
    path.ownerSVGElement?.querySelector<SVGElement>('[data-trail-terrain]')?.dataset.trailTerrain
  const value = trailData(raw)
  if (
    !value ||
    typeof value !== 'object' ||
    !('alpine' in value) ||
    !('treeline' in value) ||
    !('meadowTop' in value) ||
    !finite(value.treeline) ||
    !finite(value.meadowTop)
  )
    return null
  const alpine = value.alpine
  if (
    !alpine ||
    typeof alpine !== 'object' ||
    !('top' in alpine) ||
    !('base' in alpine) ||
    !finite(alpine.top) ||
    !finite(alpine.base)
  )
    return null
  const pond = 'pond' in value ? value.pond : null
  return {
    top: alpine.top,
    base: alpine.base,
    treeline: value.treeline,
    meadowTop: value.meadowTop,
    pond:
      pond &&
      typeof pond === 'object' &&
      'x' in pond &&
      'y' in pond &&
      'rx' in pond &&
      'ry' in pond &&
      finite(pond.x) &&
      finite(pond.y) &&
      finite(pond.rx) &&
      finite(pond.ry)
        ? { x: pond.x, y: pond.y, rx: pond.rx, ry: pond.ry }
        : null,
  }
}
const trailScaleStops = (path: SVGPathElement) => {
  const stops: ScaleStop[] = []
  const values = trailData(path.dataset.petScaleStops)
  if (Array.isArray(values))
    for (const value of values)
      if (pair(value) && value[0] >= 0 && value[0] <= 1 && value[1] > 0 && value[1] <= 1)
        stops.push(value)
  return stops.sort((left, right) => left[0] - right[0])
}
export const trailScale = (
  position: number,
  point: Point,
  stops: ScaleStop[],
  terrain: PetTerrain | null,
) => {
  if (stops.length) {
    const first = stops[0]
    if (position <= first[0]) return first[1]
    for (let index = 1; index < stops.length; index++) {
      const from = stops[index - 1]
      const to = stops[index]
      if (position <= to[0]) {
        const fraction = clamp((position - from[0]) / Math.max(0.000001, to[0] - from[0]), 0, 1)
        return from[1] + (to[1] - from[1]) * fraction
      }
    }
    return stops[stops.length - 1][1]
  }
  if (!terrain || point.y < terrain.top) return 1
  if (point.y < terrain.base)
    return (
      0.18 + 0.17 * clamp((point.y - terrain.top) / Math.max(1, terrain.base - terrain.top), 0, 1)
    )
  const depth =
    (point.y - terrain.treeline) / Math.max(1, terrain.meadowTop + 160 - terrain.treeline)
  return 0.35 + 0.65 * clamp(depth, 0, 1)
}
export const trailKey = (path: SVGPathElement) => `expedition:${path.dataset.petExpedition ?? ''}`
export const petTrail = (element: Element | null): element is SVGPathElement =>
  element instanceof SVGPathElement && element.hasAttribute('data-pet-expedition')
export const currentTrailPath = (path: SVGPathElement) =>
  path.isConnected && !!path.getAttribute('d') && trailScene(document)?.contains(path) === true

type TrailSample = { point: DOMPoint; progress: number }
const trailPoints = new WeakMap<SVGPathElement, { d: string; step: number; pts: TrailSample[] }>()
export const trailRoutes = () => {
  const scene = trailScene(document)
  const host = scene && trailHost(scene)
  if (
    !scene ||
    scene.dataset.trailRendered !== 'true' ||
    host?.hidden !== false ||
    host.dataset.trailResizing === 'true'
  )
    return []
  return Array.from(scene.querySelectorAll<SVGPathElement>('[data-pet-expedition]')).flatMap(
    path => {
      const d = path.getAttribute('d')
      const m = trailScreenMatrix(path)
      if (!d || !m) return []
      const step = 4 / Math.max(Math.hypot(m.a, m.b), Math.hypot(m.c, m.d), 0.001)
      let hit = trailPoints.get(path)
      if (hit?.d !== d || hit.step !== step) {
        const geometry = svgPathGeometry(path)
        const length = geometry.getTotalLength()
        const segments = Math.max(1, Math.ceil(length / step))
        const pts = Array.from({ length: segments + 1 }, (_, k): TrailSample => {
          const point = geometry.getPointAtLength((k * length) / segments)
          return { point: new DOMPoint(point.x, point.y), progress: k / segments }
        })
        hit = { d, step, pts }
        trailPoints.set(path, hit)
      }
      const stops = trailScaleStops(path)
      const terrain = petTerrain(path)
      const span = trailData(path.dataset.petAlpineSpan)
      const base = path.ownerSVGElement ? trailScreenMatrix(path.ownerSVGElement) : null
      const foothill = path.ownerSVGElement?.querySelector<SVGGElement>(
        '[data-trail-depth="foothill"]',
      )
      const depth = foothill ? trailScreenMatrix(foothill) : null
      const dx = base && depth ? depth.e - base.e : 0
      const dy = base && depth ? depth.f - base.f : 0
      // Capture parallax once for a compositor walk, blending beyond the mountain's boundaries.
      const points = hit.pts
      const feet = points.map(({ point: p, progress }): TrailFoot => {
        const c = p.matrixTransform(m)
        let blend = 0
        if (pair(span) && span[1] > span[0] && progress >= span[0]) {
          const edge = Math.min(0.015, (span[1] - span[0]) / 4)
          blend = easeTrailDepth((span[1] + edge - progress) / edge)
        }
        return {
          x: c.x + window.scrollX + dx * blend,
          y: c.y + window.scrollY + dy * blend,
          progress,
          scale: trailScale(progress, p, stops, terrain),
        }
      })
      return [{ path, feet, key: trailKey(path) }]
    },
  )
}

export const trailGeometry = (path: SVGPathElement) => {
  if (!currentTrailPath(path)) return ''
  const m = trailScreenMatrix(path)
  if (!m) return ''
  return `${path.getAttribute('d')}|${path.dataset.petScaleStops ?? ''}|${path.dataset.petAlpineSpan ?? ''}|${m.a},${m.b},${m.c},${m.d},${m.e + window.scrollX},${m.f + window.scrollY}|${window.innerWidth},${window.innerHeight}`
}

// Screen pixels per scene unit of a trail's drawing: the trail pet is sized in scene units like the
// folk there, and the walk planner works in screen pixels.
export const trailPx = (path: SVGPathElement) => {
  const scene = path.ownerSVGElement
  const m = scene ? trailScreenMatrix(scene) : null
  return m ? Math.hypot(m.a, m.b) : 1
}

export const expeditionPath = () => {
  const scene = trailScene(document)
  if (!scene || trailHost(scene)?.hidden !== false || scene.dataset.trailRendered !== 'true')
    return null
  const path = scene.querySelector<SVGPathElement>('[data-pet-expedition]')
  return path && petTrail(path) && currentTrailPath(path) ? path : null
}

// The walking path's progress nearest a spot, so the wander resumes the expedition from there.
export const nearestTrailProgress = (path: SVGPathElement, point: Point) => {
  let best = petState.perch?.progress ?? 0
  let near = Infinity
  for (const route of trailRoutes()) {
    if (route.path !== path) continue
    for (const foot of route.feet) {
      const d = Math.hypot(foot.x - point.x, foot.y - point.y)
      if (d < near) {
        near = d
        best = foot.progress
      }
    }
  }
  return best
}
