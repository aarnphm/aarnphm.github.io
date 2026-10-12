import type { CampSettings } from './camp-settings'
import type { RouteSites } from './pasture'
import {
  holidayLabels,
  subscribeVisitorHoliday,
  type HolidaySelection,
  type TrailHoliday,
} from '../../util/visitor-holiday'
import { circleClearance, readActors, readTrunks } from './actor-spacing'
import { campSites } from './camp-layout'
import {
  holidayAnimalSprite,
  holidayArt,
  holidaySprite,
  type HolidaySpriteName,
} from './holiday-sprites'
import { onDryLand, pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

export function mountTrailHolidays(scene: HTMLElement, signal: AbortSignal, changed: () => void) {
  let automatic: TrailHoliday | null = null
  let selection: HolidaySelection = 'auto'
  const current = () => (selection === 'auto' ? automatic : selection === 'none' ? null : selection)
  const state = () => {
    const holiday = current()
    scene.dataset.holiday = holiday ?? 'none'
    scene.dataset.holidaySelection = selection
    scene.parentElement?.dispatchEvent(new CustomEvent('trail:holiday', { detail: holiday }))
    const status = scene.querySelector<HTMLElement>('[data-holiday-status]')
    if (status)
      status.textContent = holiday
        ? `${holidayLabels[holiday]} · ${selection === 'auto' ? 'calendar' : 'preview'}`
        : selection === 'none'
          ? 'off'
          : 'outside the holiday dates'
  }
  subscribeVisitorHoliday(
    scene.ownerDocument,
    holiday => {
      if (automatic === holiday) return
      automatic = holiday
      state()
      if (selection === 'auto') changed()
    },
    signal,
  )
  state()

  const refreshTrees = () => {
    scene.querySelectorAll('[data-holiday-tree]').forEach(element => element.remove())
    if (current() !== 'christmas') return
    const trees = [...scene.querySelectorAll<SVGGElement>('.trail-tree[data-tree-depth="meadow"]')]
      .filter(tree => ['pine', 'spruce', 'cypress'].includes(tree.dataset.tree ?? ''))
      .sort((a, b) => Number(b.dataset.treeScale) - Number(a.dataset.treeScale))
      .slice(0, 3)
    for (const tree of trees) {
      const canopy = tree.querySelector<SVGGElement>('.trail-tree-canopy')
      if (!canopy) continue
      const box = canopy.getBBox()
      const centre = box.x + box.width / 2
      const star = holidayArt.star
      const wreath = holidayArt.wreath
      const size = Math.max(0.65, Math.min(1, box.width / 28))
      const ornament = scene.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'g')
      ornament.dataset.holidayTree = 'christmas'
      const lights = Array.from({ length: 5 }, (_, index) => {
        const x = centre + (index % 2 ? -1 : 1) * box.width * (0.09 + index * 0.025)
        const y = box.y + box.height * (0.35 + index * 0.11)
        return `M${x.toFixed(1)} ${y.toFixed(1)}h1.5v1.5h-1.5Z`
      })
      ornament.innerHTML = `<g transform="translate(${centre - (star[0].length * size) / 2} ${box.y - star.length * size * 0.6})">${holidaySprite('star', size)}</g><g transform="translate(${centre - (wreath[0].length * size) / 2} ${box.y + box.height * 0.8})">${holidaySprite('wreath', size)}</g><g class="trail-holiday-lights"><path class="trail-holiday-gold" d="${lights.filter((_, index) => index % 2).join('')}"/><path class="trail-holiday-red" d="${lights.filter((_, index) => !(index % 2)).join('')}"/></g>`
      canopy.append(ornament)
    }
  }

  const draw = (landscape: Landscape, routes: RouteSites[], seed: number, camps: CampSettings) => {
    const layer = scene.querySelector<SVGGElement>('.trail-holidays')
    if (!layer) return
    const holiday = current()
    scene
      .querySelectorAll('[data-holiday-decoration], [data-holiday-animal]')
      .forEach(element => element.remove())
    layer.replaceChildren()
    refreshTrees()
    if (!holiday) return
    const random = trailRandom(seed ^ 0x68259d01)
    const occupied: Point[] = []
    const solids = readTrunks(scene)
    const bodies = readActors(scene)
    const road = routes.flatMap(route => route.samples)
    const lowerRoad = road.filter(point => point.y >= landscape.meadowTop + 80)
    const sites = campSites(landscape, seed, camps)
    const safeProp = (anchor: Point) => {
      for (let attempt = 0; attempt < 96; attempt++) {
        const angle = attempt * 2.399963
        const reach = attempt ? 6 + Math.floor(attempt / 12) * 8 : 0
        const candidate = onDryLand(
          { x: anchor.x + Math.cos(angle) * reach, y: anchor.y + Math.sin(angle) * reach },
          landscape,
          1.8,
        )
        if (
          pondDistance(candidate, landscape.pond) >= 1.8 &&
          solids.every(solid => circleClearance(candidate, 12, solid) >= 0) &&
          bodies.every(body => circleClearance(candidate, 12, body) >= 0) &&
          occupied.every(point => Math.hypot(candidate.x - point.x, candidate.y - point.y) > 32) &&
          road.every(point => Math.hypot(candidate.x - point.x, candidate.y - point.y) > 24)
        ) {
          occupied.push(candidate)
          return candidate
        }
      }
      return null
    }
    const props: string[] = []
    const prop = (name: HolidaySpriteName, anchor: Point, size: number) => {
      const point = safeProp(anchor)
      if (!point) return
      const rows = holidayArt[name]
      const x = Math.round(point.x)
      const y = Math.round(point.y)
      props.push(
        `<g class="trail-holiday-${name}" data-holiday-decoration="${name}" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x} ${y})"><g transform="translate(${(-rows[0].length * size) / 2} ${-rows.length * size})"><g${name === 'ghost' ? ' class="trail-holiday-float"' : ''}>${holidaySprite(name, size)}</g></g></g>`,
      )
    }
    for (const site of sites) {
      if (holiday === 'thanksgiving') prop('gourd', { x: site.x + 58, y: site.y + 18 }, 1)
      if (holiday === 'halloween') {
        prop('pumpkin', { x: site.x + 56, y: site.y + 18 }, 1)
        prop('ghost', { x: site.x - 66, y: site.y - 18 }, 1)
      }
      if (holiday === 'christmas') prop('gift', { x: site.x + 60, y: site.y + 18 }, 1)
    }
    if (holiday === 'easter') {
      for (let index = 0; index < 3; index++)
        prop(
          'egg',
          {
            x: landscape.pond.x + landscape.pond.rx * (index % 2 ? -1.9 : 1.9),
            y: landscape.pond.y + (index - 1) * 42,
          },
          1,
        )
    }
    const animals: string[] = []
    const kind = holiday === 'thanksgiving' ? 'turkey' : holiday === 'easter' ? 'rabbit' : null
    if (kind) {
      const count = landscape.width < 700 ? 1 : 2
      for (let index = 0; index < count; index++) {
        const anchor =
          lowerRoad[Math.floor(random() * lowerRoad.length)] ??
          landscape.groves[index % landscape.groves.length]
        const point = onDryLand(
          { x: anchor.x + (index % 2 ? 38 : -38), y: anchor.y + 24 },
          landscape,
          1.9,
        )
        const rows = holidayArt[kind]
        const size = index ? 1 : 1.5
        const x = Math.round(point.x)
        const y = Math.round(point.y)
        animals.push(
          `<g class="trail-${kind}" data-holiday-animal="${kind}" data-trail-wander="holiday-${kind}-${index}" data-trail-animal="${kind}" data-animal-home-x="${x}" data-animal-home-y="${y}" data-animal-size="${size}" data-phase="waiting" data-motion="paused" data-stride="0" transform="translate(${x} ${y})" style="--trail-animal-step:${size}px"><g class="trail-animal-facing"><g transform="translate(${(-rows[0].length * size) / 2} ${-(rows.length - 1) * size})">${holidayAnimalSprite(kind, size)}</g></g></g>`,
        )
      }
    }
    layer.innerHTML = `<g class="trail-holiday-props">${props.join('')}</g><g class="trail-holiday-animals">${animals.join('')}</g>`
  }

  return {
    draw,
    refreshTrees,
    select(value: HolidaySelection) {
      if (value === selection) return
      selection = value
      state()
      changed()
    },
  }
}
