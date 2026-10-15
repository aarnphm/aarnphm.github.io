import type { ForestSettings } from './scene-settings'
import { createForestSprites } from './forest-sprites'
import { creekDistance, groundOffset, type Landscape, type Point } from './landscape'
import { trailMountainBackdrop } from './mountain-backdrop'
import { trailRandom } from './random'
import { trailCreek } from './ravine'
import { createTreePainter, trailPatchTrees } from './scenery'
import { trailBoulder } from './understory'
import { trailSnowBase } from './winter'

type Region = { id: string; left: number; top: number; right: number; bottom: number }
type Patch = {
  region: Region
  trees: SVGGElement
  neighbours: Point[]
  sprites: ReturnType<typeof createForestSprites>
}

function patchLandscape(base: Landscape, region: Region) {
  const outlet = base.creek[0].at(-1)
  if (!outlet || region.right <= outlet.x) return base
  const course: Point[] = []
  for (let x = Math.max(region.left, outlet.x); x <= region.right + 6; x += 6) {
    const distance = x - outlet.x
    course.push({
      x,
      y: outlet.y + 32 * (1 - Math.exp(-distance / 320)) + Math.sin(distance / 113) * 8,
    })
  }
  return {
    ...base,
    width: region.right,
    height: Math.max(base.height, region.bottom),
    creek: [...base.creek, course],
  }
}

function terrainPatch(landscape: Landscape, seed: number, region: Region, water: Landscape) {
  const levels = [0.16, 0.39, 0.64, 0.91]
  const paths = levels
    .map((level, index) => {
      const base = landscape.meadowTop + (landscape.camp.y - landscape.meadowTop) * level
      let d = ''
      for (let x = region.left; x <= region.right; x += 8) {
        const y = Math.max(
          region.top,
          Math.round((base + groundOffset(x + index * 36, landscape, base)) / 4) * 4,
        )
        d += `${d ? 'H' : 'M'}${x}${d ? 'V' : ' '}${y}`
      }
      d += `H${region.right}V${region.bottom}H${region.left}Z`
      return `<path class="trail-terrace trail-terrace-${index}" d="${d}"/>`
    })
    .join('')
  const random = trailRandom(seed ^ 0x74bc938d)
  const marks: string[] = []
  const top = Math.max(region.top + 12, landscape.meadowTop + 20)
  for (let row = top; row < region.bottom - 20; row += 48)
    for (let column = region.left + 12; column < region.right - 12; column += 44) {
      const x = Math.round((column + random() * 20) / 2) * 2
      const y = Math.round((row + random() * 20) / 2) * 2
      if (
        creekDistance({ x, y }, water) < 24 ||
        [18, 50].some(offset => Math.abs(y - landscape.alpine.base - offset) < 12)
      )
        continue
      if (random() < 0.25)
        marks.push(
          trailBoulder({ x, y }, seed ^ (x * 31 + y), 1_000_000 + (seed >>> 0) + x + y).replace(
            'class="trail-understory"',
            `class="trail-understory" data-world-patch-object="${region.id}" data-trail-chunk="${region.id}"`,
          ),
        )
      else if (random() < 0.6)
        marks.push(
          `<path class="trail-meadow-tuft" d="M${x} ${y}h2v-4h-2ZM${x + 4} ${y + 2}h2v-6h-2Z"/>`,
        )
    }
  const tracks =
    region.top <= landscape.alpine.base
      ? [18, 50]
          .map((offset, index) => {
            const y = landscape.alpine.base + offset
            return `<path class="trail-mountain-track" data-world-track="${region.id}-${index}" d="M${region.left} ${y}Q${(region.left + region.right) / 2} ${y + Math.sin(region.left / 113) * 12} ${region.right} ${y}"/>`
          })
          .join('')
      : ''
  const course = water.creek.at(-1)
  const runoff =
    course &&
    course[0].x >= region.left &&
    course.some(point => point.y >= region.top && point.y < region.bottom) &&
    region.right > landscape.pond.x
      ? trailCreek({ ...water, creek: [course], stream: [] }, seed)
      : ''
  return { base: paths, detail: runoff + tracks + marks.join('') }
}

export function mountWorldPatches(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  settings: () => ForestSettings,
  walks: () => Point[],
  refresh: () => void,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('World patches need the scene window')
  const paint = createTreePainter()
  const patches = new Map<string, Patch>()
  let landscape: Landscape | null = null
  let seed = 0
  let clip = ''
  let width = 0
  let height = 0
  let epoch = 0
  let running = false
  const pending: Region[] = []
  const neighbours = () =>
    [...scene.querySelectorAll<SVGGElement>('[data-tree-id]')].map(tree => ({
      x: Number(tree.dataset.groundX),
      y: Number(tree.dataset.groundY),
    }))
  const append = (selector: string, markup: string) =>
    scene.querySelector(selector)?.insertAdjacentHTML('beforeend', markup)
  const generate = async () => {
    if (running || !landscape || !pending.length || signal.aborted || document.hidden) return
    running = true
    scene.dataset.trailPatches = 'generating'
    const generation = epoch
    const base = landscape
    const neighboursAtStart = neighbours()
    while (pending.length && generation === epoch && !signal.aborted && !document.hidden) {
      const region = pending.shift()
      if (!region) break
      const patchSeed =
        seed ^
        Math.imul(Math.round(region.left) + 1, 374761393) ^
        Math.imul(Math.round(region.top) + 1, 668265263)
      const land = patchLandscape(base, region)
      const bounds = `data-world-patch="${region.id}" data-trail-chunk="${region.id}" data-patch-left="${region.left}" data-patch-top="${region.top}" data-patch-right="${region.right}" data-patch-bottom="${region.bottom}" data-patch-bounds='${JSON.stringify(region)}'`
      const known = patches.get(region.id)
      if (!known && region.top < base.alpine.base) {
        const narrow = base.width < 700
        const extended = {
          ...base,
          width,
          alpine: {
            ...base.alpine,
            left: narrow ? region.left : base.alpine.left,
            width: narrow ? region.right - region.left : width,
          },
        }
        append(
          '[data-trail-backdrop-container]',
          `<g ${bounds}>${trailMountainBackdrop(extended, narrow ? patchSeed : seed, clip, undefined, region)}</g>`,
        )
      }
      if (!known) {
        const terrain = terrainPatch(base, patchSeed, region, land)
        append(
          '.trail-ground-base',
          `<g ${bounds}>${terrain.base}${trailSnowBase(base, seed, region)}</g>`,
        )
        append('.trail-ground', `<g ${bounds}>${terrain.detail}</g>`)
      }
      const trees = known?.trees ?? document.createElementNS('http://www.w3.org/2000/svg', 'g')
      if (!known) {
        trees.dataset.worldPatchForest = region.id
        scene.querySelector('.trail-extension-forest')?.append(trees)
      }
      const patch = known ?? {
        region,
        trees,
        neighbours: neighboursAtStart,
        sprites: createForestSprites(document),
      }
      patches.set(region.id, patch)
      const wanted = new Set<string>()
      let batch = view.performance.now()
      for (const sprite of trailPatchTrees(
        land,
        patchSeed,
        region,
        settings(),
        patch.neighbours,
        walks(),
        paint,
      )) {
        if (generation !== epoch || signal.aborted) break
        wanted.add(sprite.id)
        patch.sprites.update(sprite, trees)
        if (view.performance.now() - batch >= 6) {
          await new Promise<void>(resolve => view.setTimeout(resolve, 0))
          batch = view.performance.now()
        }
      }
      if (generation !== epoch || signal.aborted) break
      patch.sprites.retain(wanted)
      refresh()
      await new Promise<void>(resolve => view.setTimeout(resolve, 0))
    }
    running = false
    scene.dataset.trailPatches = 'ready'
    if (generation !== epoch && !signal.aborted) void generate()
  }
  host.addEventListener(
    'trail:show',
    () => {
      void generate()
    },
    { signal },
  )
  document.addEventListener('visibilitychange', () => void generate(), { signal })
  return {
    reset(base: Landscape, nextSeed: number, clipId: string) {
      epoch++
      pending.length = 0
      for (const tree of scene.querySelectorAll('[data-tree-id^="tree-patch-"]')) tree.remove()
      for (const element of scene.querySelectorAll(
        '[data-world-patch], [data-world-patch-forest], [data-world-patch-object]',
      ))
        element.remove()
      patches.clear()
      landscape = base
      seed = nextSeed
      clip = clipId
      width = base.width
      height = base.height
    },
    update(nextWidth: number, nextHeight: number) {
      if (!landscape) return
      while (width < nextWidth) {
        pending.push({ id: `${width}-0`, left: width, right: width + 256, top: 0, bottom: height })
        width += 256
      }
      while (height < nextHeight) {
        pending.push({
          id: `0-${height}`,
          left: 0,
          right: width,
          top: height,
          bottom: height + 256,
        })
        height += 256
      }
      void generate()
    },
    refreshForest() {
      if (!landscape) return
      epoch++
      const requests = new Map([...patches.values()].map(patch => [patch.region.id, patch.region]))
      for (const region of pending) requests.set(region.id, region)
      pending.splice(0, pending.length, ...requests.values())
      void generate()
    },
  }
}
