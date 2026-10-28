import type { HolidaySelection } from '../../util/visitor-holiday'
import type { RouteSites } from './pasture'
import {
  readTrailHolidayChoice,
  subscribeTrailHolidayChoice,
} from '../../util/trail-holiday-choice'
import { trailHost } from '../../util/trail-scene'
import { subscribeVisitorHoliday, type TrailHoliday } from '../../util/visitor-holiday'
import { campSites } from './camp-layout'
import { holidayAnimalSprite, holidayArt } from './holiday-sprites'
import { drawHalloween } from './holidays/halloween'
import { createPropPlacer } from './holidays/props'
import { christmasTree, cobweb, taper, treeCells } from './holidays/tree-ornaments'
import { onDryLand, type Landscape } from './landscape'
import { trailRandom } from './random'
import { sceneSettings, type CampSettings } from './scene-settings'

export function mountTrailHolidays(scene: HTMLElement, signal: AbortSignal, changed: () => void) {
  let automatic: TrailHoliday | null = null
  // The visitor's choice on the boulder switch wins over the authored setting.
  let selection: HolidaySelection =
    readTrailHolidayChoice(scene.ownerDocument.defaultView) ?? sceneSettings.holiday
  let decorated: SVGGElement[] = []
  const current = () => (selection === 'auto' ? automatic : selection === 'none' ? null : selection)
  const state = () => {
    const holiday = current()
    scene.dataset.holiday = holiday ?? 'none'
    scene.dataset.holidaySelection = selection
    trailHost(scene)?.dispatchEvent(new CustomEvent('trail:holiday', { detail: holiday }))
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
  subscribeTrailHolidayChoice(
    scene.ownerDocument,
    choice => {
      const next = choice ?? sceneSettings.holiday
      if (next === selection) return
      const before = current()
      selection = next
      state()
      if (current() !== before) changed()
    },
    signal,
  )
  state()

  const refreshTrees = () => {
    const previous = new Set(scene.querySelectorAll<SVGGElement>('[data-holiday-tree]'))
    decorated = []
    if (!scene.isConnected) return
    const holiday = current()
    if (holiday !== 'christmas' && holiday !== 'halloween') {
      for (const ornament of previous) ornament.remove()
      return
    }
    const trees = [
      ...scene.querySelectorAll<SVGGElement>('.trail-tree[data-tree-depth="meadow"]'),
    ].flatMap(tree => {
      const canopy = tree.querySelector<SVGGElement>('.trail-tree-canopy')
      const cells = treeCells(canopy?.querySelector('.trail-tree-hollow'))
      return canopy && cells.length >= 12 ? [{ tree, canopy, cells }] : []
    })
    trees.sort((a, b) => Number(b.tree.dataset.treeScale) - Number(a.tree.dataset.treeScale))
    const chosen: typeof trees = []
    // Halloween webs the ten largest meadow crowns.
    if (holiday === 'halloween') chosen.push(...trees.slice(0, 10))
    // The largest true cones, spread across the meadow. Pine tufts and cypress columns are not
    // Christmas-tree shaped even when they taper.
    else
      for (const entry of trees)
        if (
          chosen.length < 3 &&
          ['spruce', 'fir', 'hemlock'].includes(entry.tree.dataset.tree ?? '') &&
          taper(entry.cells) < 0.5 &&
          chosen.every(
            other =>
              Math.abs(Number(other.tree.dataset.groundX) - Number(entry.tree.dataset.groundX)) >
              160,
          )
        )
          chosen.push(entry)
    for (const [index, { tree, canopy, cells }] of chosen.entries()) {
      const existing = canopy.parentElement?.querySelector<SVGGElement>('[data-holiday-tree]')
      if (existing?.dataset.holidayTree === holiday) {
        previous.delete(existing)
        decorated.push(canopy)
        continue
      }
      const markup =
        holiday === 'christmas'
          ? christmasTree(cells)
          : cobweb(
              cells,
              treeCells(tree.querySelector('.trail-tree-motion > .trail-tree-trunk')),
              index % 2 ? -1 : 1,
            )
      if (!markup) continue
      const ornament = scene.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'g')
      ornament.dataset.holidayTree = holiday
      ornament.innerHTML = markup
      canopy.after(ornament)
      decorated.push(canopy)
    }
    for (const ornament of previous) ornament.remove()
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
    const placer = createPropPlacer(scene, landscape, routes)
    const { prop, props, lowerRoad } = placer
    const sites = campSites(landscape, seed, camps)
    const sky =
      holiday === 'halloween'
        ? drawHalloween({ landscape, sites, random, placer, decorated, layer })
        : ''
    for (const site of sites) {
      if (holiday === 'thanksgiving') prop('harvest', { x: site.x + 58, y: site.y + 18 })
      if (holiday === 'columbus-day') prop('trailMap', { x: site.x + 58, y: site.y + 18 })
    }
    if (holiday === 'christmas')
      for (const [index, canopy] of decorated.entries()) {
        const tree = canopy.closest<SVGGElement>('.trail-tree')
        if (!tree) continue
        prop('gifts', {
          x: Number(tree.dataset.groundX) + (index % 2 ? -16 : 16),
          y: Number(tree.dataset.groundY) + 6,
        })
      }
    if (holiday === 'easter') {
      for (let index = 0; index < 3; index++)
        prop(
          'egg',
          {
            x: landscape.pond.x + landscape.pond.rx * (index % 2 ? -1.9 : 1.9),
            y: landscape.pond.y + (index - 1) * 42,
          },
          { variant: `trail-holiday-egg-${index}` },
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
    layer.innerHTML = `<g class="trail-holiday-bats">${sky}</g><g class="trail-holiday-props">${props.join('')}</g><g class="trail-holiday-animals">${animals.join('')}</g>`
  }

  signal.addEventListener(
    'abort',
    () => {
      scene
        .querySelectorAll('[data-holiday-tree], [data-holiday-decoration], [data-holiday-animal]')
        .forEach(element => element.remove())
      scene.querySelector('.trail-holidays')?.replaceChildren()
      scene.dataset.holiday = 'none'
      delete scene.dataset.holidaySelection
    },
    { once: true },
  )

  return { draw, refreshTrees }
}
