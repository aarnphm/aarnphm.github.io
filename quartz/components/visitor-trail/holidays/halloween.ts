import type { CampSite } from '../camp-layout'
import type { Landscape, Point } from '../landscape'
import { holidaySprite, type HolidaySpriteName } from '../holiday-sprites'
import { lanternGlow, type PropPlacer } from './props'

export type HalloweenScene = {
  landscape: Landscape
  sites: CampSite[]
  random: () => number
  placer: PropPlacer
  // Webbed canopies, largest first, from refreshTrees.
  decorated: SVGGElement[]
  layer: SVGGElement
}

const shadow = '<path class="trail-holiday-shadow" d="M-6-2h12v2h-12Z"/>'

// Piece `index` of `count` takes a sample from its own slice of the lower trail, so the wayside
// pieces do not bunch up.
function stretch(lowerRoad: Point[], index: number, count: number, random: () => number) {
  return lowerRoad[Math.floor(((index + random()) / count) * lowerRoad.length)]
}

function crownPoint(canopy: SVGGElement, toLayer: DOMMatrix): Point | null {
  const matrix = canopy.getScreenCTM()
  if (!matrix) return null
  const box = canopy.getBBox()
  // DOMMatrix.transformPoint is missing on SVG CTMs in some engines.
  const x = matrix.a * (box.x + box.width / 2) + matrix.c * box.y + matrix.e
  const y = matrix.b * (box.x + box.width / 2) + matrix.d * box.y + matrix.f
  return {
    x: toLayer.a * x + toLayer.c * y + toLayer.e,
    y: toLayer.b * x + toLayer.d * y + toLayer.f,
  }
}

const flyer = (name: HolidaySpriteName, x: number, y: number) =>
  `<g class="trail-holiday-bat" transform="translate(${Math.round(x)} ${Math.round(y)})">${holidaySprite(name, 2)}</g>`

// Places the Halloween ground props and returns the sky markup (bats and ravens), which the caller
// puts in the bats group. Placed once per layout; nothing here moves.
export function drawHalloween({
  landscape,
  sites,
  random,
  placer,
  decorated,
  layer,
}: HalloweenScene): string {
  const { prop, lowerRoad } = placer
  const { width, meadowTop } = landscape
  const wide = width >= 700
  // Camps: a pair of jack-o'-lanterns and a figure past the fire. Tents and the camp spur crowd
  // some sites, so the figure tries three spots.
  for (const [index, site] of sites.entries()) {
    prop('lanterns', { x: site.x + 56, y: site.y + 18 }, { under: lanternGlow(26, 8) })
    const figure = index % 2 ? 'ghost' : 'skeleton'
    for (const [dx, dy] of [
      [-66, -18],
      [-48, 34],
      [12, 44],
    ])
      if (
        prop(figure, { x: site.x + dx, y: site.y + dy }, { lift: index % 2 ? 6 : 0, under: shadow })
      )
        break
  }
  // Gallows trees stand in the upper meadow at 3 px, the size of the living trees around them.
  const gallows: Point[] = []
  for (let index = 0; index < (wide ? 2 : 1); index++)
    for (let attempt = 0; attempt < 4 && gallows.length <= index; attempt++) {
      const point = prop(
        'gallowsTree',
        {
          x: width * (0.15 + index * 0.5 + attempt * 0.08) + random() * width * 0.08,
          y: meadowTop + 60 + random() * 70,
        },
        { radius: 20, size: 3 },
      )
      if (point) gallows.push(point)
    }
  // Wayside pieces, one per stretch of the lower trail.
  const wayside: [HolidaySpriteName, number][] = [
    ['gibbet', 18],
    ['pumpkinPatch', 20],
    ['graveyard', 30],
  ]
  if (wide) wayside.push(['guillotine', 16], ['hearse', 30])
  let plot: Point | null = null
  for (const [index, [name, radius]] of wayside.entries()) {
    // Either side of the trail, at two samples of the stretch, before giving the piece up.
    let point: Point | null = null
    for (let attempt = 0; attempt < 4 && !point; attempt++) {
      const anchor = stretch(lowerRoad, index, wayside.length, random)
      if (!anchor) break
      const side = (index + attempt) % 2 ? -1 : 1
      point = prop(
        name,
        { x: anchor.x + side * (radius + 20), y: anchor.y + 14 },
        { radius, under: name === 'pumpkinPatch' ? lanternGlow(26, 7) : '' },
      )
    }
    if (name === 'graveyard') plot = point
  }
  // The plot grows around the graveyard: a crypt, railings, a widow, candles, the resurrection
  // men at work, and two more of the dead.
  if (plot) {
    prop('mausoleum', { x: plot.x + 58, y: plot.y - 4 }, { radius: 22 })
    prop('ironFence', { x: plot.x - 54, y: plot.y + 6 }, { radius: 30 })
    prop('mourner', { x: plot.x + 20, y: plot.y + 22 }, { radius: 6, under: shadow })
    prop('candles', { x: plot.x - 12, y: plot.y + 26 }, { radius: 8, under: lanternGlow(14, 4) })
    prop(
      'resurrectionists',
      { x: plot.x - 78, y: plot.y + 30 },
      { radius: 22, under: lanternGlow(16, 5) },
    )
    for (const dx of [-30, 44])
      prop('ghost', { x: plot.x + dx, y: plot.y + 40 }, { lift: 8, under: shadow })
  }
  // Gaslight along the lower trail; the glass burns after dark.
  const lamps = wide ? 4 : 2
  for (let index = 0; index < lamps; index++) {
    const anchor = lowerRoad[Math.floor(((index + 0.5) / lamps) * lowerRoad.length)]
    if (anchor)
      prop(
        'gaslamp',
        { x: anchor.x + (index % 2 ? -22 : 22), y: anchor.y + 8 },
        { radius: 6, under: lanternGlow(18, 6) },
      )
  }
  // The pasture's scarecrow, and the crows it no longer frightens.
  const post = prop(
    'scarecrow',
    { x: width * (0.55 + random() * 0.3), y: meadowTop + 140 + random() * 80 },
    { radius: 10 },
  )
  if (post)
    for (const dx of [44, -44])
      if (prop('crowLine', { x: post.x + dx, y: post.y + 14 }, { radius: 20 })) break
  // A coffin leans on every other webbed trunk.
  for (const [index, canopy] of decorated.entries()) {
    const tree = canopy.closest<SVGGElement>('.trail-tree')
    if (index % 2 || !tree) continue
    prop(
      'coffin',
      { x: Number(tree.dataset.groundX) + 16, y: Number(tree.dataset.groundY) + 4 },
      { radius: 6 },
    )
  }
  // Bats leave the webbed crowns at dusk and ravens the gallows.
  const sky: string[] = []
  const toLayer = layer.getScreenCTM()?.inverse()
  if (toLayer)
    for (const [index, canopy] of decorated.slice(0, 4).entries()) {
      const crown = crownPoint(canopy, toLayer)
      if (!crown) continue
      for (const [name, dx, dy] of [
        ['bat', index % 2 ? 14 : -26, -22],
        ['batGlide', index % 2 ? -20 : 18, -38],
      ] as const)
        sky.push(flyer(name, crown.x + dx, crown.y + dy))
    }
  for (const point of gallows)
    for (const [dx, dy] of [
      [-34, -82],
      [-8, -96],
      [20, -86],
    ])
      sky.push(flyer('raven', point.x + dx, point.y + dy))
  return sky.join('')
}
