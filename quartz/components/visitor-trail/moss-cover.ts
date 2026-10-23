import type { TreePlacement } from './scenery'
import { bayerThreshold } from '../../util/bayer'
import { creekDistance, type Landscape, type Point, pondDistance } from './landscape'
import { mossWeights } from './moss-model'
import { type MossRequest, type MossResponse, mossSize, mossTones } from './moss-nca'
import { trailRandom } from './random'

declare const TRAIL_MOSS_WORKER_URL: string

// Moss clumps live on one canvas under the meadow, one scene pixel per canvas pixel. A clump's
// cells are one pixel on the far meadow and two near the viewer. A worker runs the cellular
// automaton and sends back tone indices; this side paints them in the theme's colours and tells
// the compositor the surface changed. Boot prints that land on a clump kill its cells, and the
// automaton grows them back.
const spacing = 52
const limit = 40
const trackLayers =
  '.trail-hiker-tracks, .trail-reader-tracks, .trail-camp-tracks, .trail-visitor-tracks, .trail-ranger-tracks, .trail-athlete-tracks'

// `left` and `top` are canvas pixels; `cell` is the pixel size of one automaton cell.
type Clump = { id: string; left: number; top: number; cell: number }

export function mountMossCover(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-ground-cover')
  if (!view || !layer) throw new Error('Moss needs the trail scene')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let worker: Worker | null = null
  let frame: SVGForeignObjectElement | null = null
  let canvas: HTMLCanvasElement | null = null
  let context: CanvasRenderingContext2D | null = null
  let clumps = new Map<string, Clump>()
  let tones = new Map<string, Uint8Array>()
  let top = 0
  let palette: Uint8ClampedArray[] = []
  let paletteSamples: HTMLElement[] = []
  let paletteFrame = 0
  let pendingHits: { id: string; x: number; y: number; radius: number }[] = []
  let hitFlush = 0
  let hits = 0
  let cost = { steps: 0, worker: 0, paint: 0 }
  const images = [new ImageData(mossSize, mossSize), new ImageData(mossSize * 2, mossSize * 2)]
  const probe = document.createElement('canvas')
  probe.width = probe.height = 1
  const probeContext = probe.getContext('2d', { willReadFrequently: true })

  const send = (message: MossRequest) => worker?.postMessage(message)
  const stopWorker = () => {
    worker?.terminate()
    worker = null
  }
  // The weather's say over the moss (world-schedule.ts mossClimate, stamped by the trail's
  // weather): snow buries cells, rime whitens them, rain darkens them a tone and drought bleaches
  // them one, each through the shared Bayer screen in scene pixels so the dither holds still.
  let climate = { moisture: 0, growth: 1, buried: 0, rime: 0, lean: 0 }
  const readClimate = () => {
    const [moisture = 0, growth = 1, buried = 0, rime = 0, lean = 0] = (scene.dataset.moss ?? '')
      .split(' ')
      .map(Number)
      .map(value => (Number.isFinite(value) ? value : undefined))
    const next = { moisture, growth, buried, rime, lean }
    const changed = Object.entries(next).some(
      ([key, value]) => climate[key as keyof typeof climate] !== value,
    )
    climate = next
    return changed
  }
  const weathered = (tone: number, column: number, row: number) => {
    const threshold = bayerThreshold(column, row)
    if (climate.buried > threshold) return 0
    if (climate.rime > threshold) return mossTones
    const shift =
      climate.moisture > 0
        ? climate.moisture * 0.7 > threshold
          ? -1
          : 0
        : -climate.moisture * 0.8 > threshold
          ? 1
          : 0
    return Math.max(1, Math.min(mossTones, tone + shift))
  }
  const paintClump = (clump: Clump, cells: Uint8Array) => {
    if (!context) return
    const image = images[clump.cell - 1]
    const { data } = image
    const stride = mossSize * clump.cell
    for (let index = 0; index < cells.length; index++) {
      const x = (index % mossSize) * clump.cell
      const y = Math.floor(index / mossSize) * clump.cell
      for (let dy = 0; dy < clump.cell; dy++)
        for (let dx = 0; dx < clump.cell; dx++) {
          const offset = ((y + dy) * stride + x + dx) * 4
          const tone =
            cells[index] && weathered(cells[index], clump.left + x + dx, clump.top + y + dy)
          if (tone) data.set(palette[tone - 1], offset)
          else data[offset + 3] = 0
        }
    }
    context.putImageData(image, clump.left, clump.top)
  }
  const repaint = () => {
    for (const [id, cells] of tones) {
      const clump = clumps.get(id)
      if (clump) paintClump(clump, cells)
    }
    frame?.dispatchEvent(new Event('trail:canopy-paint', { bubbles: true }))
  }
  const readPalette = () => {
    const surface = canvas
    if (!surface || !probeContext) return
    palette = paletteSamples.map(sample => {
      probeContext.fillStyle = view.getComputedStyle(sample).color
      probeContext.clearRect(0, 0, 1, 1)
      probeContext.fillRect(0, 0, 1, 1)
      return probeContext.getImageData(0, 0, 1, 1).data
    })
  }
  const startWorker = () => {
    if (worker) return
    try {
      worker = new Worker(TRAIL_MOSS_WORKER_URL, { name: 'trail-moss' })
    } catch {
      stopWorker()
      return
    }
    worker.addEventListener('message', (event: MessageEvent<MossResponse>) => {
      const message = event.data
      if (message.type !== 'paint') return
      const started = performance.now()
      const rects = []
      for (const { id, tones: cells } of message.paints) {
        const clump = clumps.get(id)
        if (!clump) continue
        tones.set(id, cells)
        paintClump(clump, cells)
        const size = mossSize * clump.cell
        rects.push({
          left: clump.left,
          top: clump.top,
          right: clump.left + size,
          bottom: clump.top + size,
        })
      }
      // Name the clumps so the compositor repaints them rather than the whole meadow.
      if (rects.length)
        frame?.dispatchEvent(
          new CustomEvent('trail:canopy-paint', { bubbles: true, detail: { rects } }),
        )
      cost.steps++
      cost.worker += message.cost
      cost.paint += performance.now() - started
      layer.dataset.mossAwake = String(message.awake)
      layer.dataset.mossCost = `${cost.steps}:${cost.worker.toFixed(1)}:${cost.paint.toFixed(1)}`
    })
    worker.addEventListener('error', event => {
      event.preventDefault()
      layer.dataset.mossError = event.message || 'worker error'
      stopWorker()
    })
    send({ type: 'init', weights: mossWeights, seed: Math.floor(Math.random() * 4294967296) })
    send({ type: 'climate', growth: climate.growth, lean: climate.lean })
  }
  const clear = () => {
    view.cancelAnimationFrame(paletteFrame)
    paletteFrame = 0
    send({ type: 'clear' })
    clumps = new Map()
    tones = new Map()
    pendingHits = []
    layer.replaceChildren()
    frame = null
    canvas = null
    context = null
    paletteSamples = []
  }
  // Boot prints arrive as footprint paths appended to the track layers.
  const trample = (x: number, y: number) => {
    for (const clump of clumps.values()) {
      const lx = (x - clump.left) / clump.cell
      const ly = (y - top - clump.top) / clump.cell
      if (lx < -2 || ly < -2 || lx > mossSize + 2 || ly > mossSize + 2) continue
      // A boot is about six scene pixels long.
      pendingHits.push({ id: clump.id, x: lx, y: ly, radius: 3.6 / clump.cell })
      layer.dataset.mossLastHit = `${clump.id}:${clump.left}:${clump.top + top}:${mossSize * clump.cell}`
    }
    if (pendingHits.length && !hitFlush)
      hitFlush = view.setTimeout(() => {
        hitFlush = 0
        hits += pendingHits.length
        layer.dataset.mossHits = String(hits)
        send({ type: 'damage', hits: pendingHits })
        pendingHits = []
      }, 50)
  }
  const observer = new view.MutationObserver(records => {
    if (!clumps.size) return
    for (const record of records)
      for (const node of record.addedNodes) {
        if (!(node instanceof view.Element)) continue
        // Journeys append a bare path per step; the patrol wraps its prints in a group.
        const mark = node.localName === 'path' ? node : node.querySelector('path')
        const place = mark
          ?.getAttribute('transform')
          ?.match(/translate\(\s*(-?[\d.]+)[\s,]+(-?[\d.]+)/)
        if (place) trample(Number(place[1]), Number(place[2]))
      }
  })
  for (const tracks of scene.querySelectorAll(trackLayers))
    observer.observe(tracks, { childList: true, subtree: true })

  // Seeds go where moss grows: at trunk feet in the groves, beside fallen logs and boulders,
  // and on the trail verges where boots will cross them.
  const plan = (landscape: Landscape, trees: readonly TreePlacement[], walks: Point[][]) => {
    const random = trailRandom(landscape.width ^ 0x5f3759df ^ trees.length)
    const { meadowTop, height, width, camp, pond } = landscape
    const near = meadowTop + (camp.y - meadowTop) * 0.45
    const candidates: Point[] = []
    for (const tree of trees)
      if (!tree.foothill && random() < 0.35)
        candidates.push({ x: tree.x + (random() - 0.5) * 20, y: tree.y + 4 })
    for (const plant of scene.querySelectorAll<SVGGElement>(
      '.trail-understory:is([data-plant="log"], [data-plant="boulder"])',
    )) {
      const x = Number(plant.dataset.groundX)
      const y = Number(plant.dataset.groundY)
      if (Number.isFinite(x) && Number.isFinite(y))
        candidates.push({ x: x + (random() < 0.5 ? -12 : 12), y: y + 2 })
    }
    for (const route of walks)
      for (let index = 14; index < route.length - 14; index += 28) {
        const a = route[index - 2]
        const b = route[index + 2]
        const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
        const side = random() < 0.5 ? -1 : 1
        candidates.push({
          x: route[index].x - ((b.y - a.y) / length) * side * 11,
          y: route[index].y + ((b.x - a.x) / length) * side * 11,
        })
      }
    const placed: Point[] = []
    const planned: Clump[] = []
    for (const point of candidates) {
      if (planned.length >= limit) break
      if (
        point.y < meadowTop + 30 ||
        point.y > height - 20 ||
        point.x < 20 ||
        point.x > width - 20 ||
        pondDistance(point, pond) < 1.5 ||
        creekDistance(point, landscape) < 14 ||
        Math.hypot(point.x - camp.x, point.y - camp.y) < 100 ||
        placed.some(other => Math.hypot(other.x - point.x, other.y - point.y) < spacing)
      )
        continue
      placed.push(point)
      const cell = point.y > near ? 2 : 1
      const half = (mossSize * cell) / 2
      planned.push({
        id: `moss-${planned.length}`,
        left: Math.round(point.x) - half,
        top: Math.round(point.y - top) - half,
        cell,
      })
    }
    return planned
  }

  const plant = (landscape: Landscape, trees: readonly TreePlacement[], walks: Point[][]) => {
    clear()
    top = landscape.meadowTop
    const width = Math.ceil(landscape.width)
    const rows = Math.ceil(landscape.height - top)
    if (width < 1 || rows < 1) return
    frame = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject')
    frame.setAttribute('x', '0')
    frame.setAttribute('y', String(top))
    frame.setAttribute('width', String(width))
    frame.setAttribute('height', String(rows))
    canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = rows
    canvas.className = 'trail-moss'
    context = canvas.getContext('2d')
    // Resolve all tones without interleaving style writes and reads on the live surface.
    paletteSamples = Array.from({ length: mossTones }, (_, index) => {
      const sample = document.createElement('span')
      sample.hidden = true
      sample.style.color = `var(--trail-moss-${index + 1})`
      return sample
    })
    frame.append(canvas, ...paletteSamples)
    layer.append(frame)
    readPalette()
    for (const clump of plan(landscape, trees, walks)) {
      const size = mossSize * clump.cell
      clump.left = Math.max(0, Math.min(width - size, clump.left))
      clump.top = Math.max(0, Math.min(rows - size, clump.top))
      clumps.set(clump.id, clump)
    }
    layer.dataset.mossClumps = String(clumps.size)
    cost = { steps: 0, worker: 0, paint: 0 }
    hits = 0
    if (!clumps.size) return
    startWorker()
    send({ type: 'plant', ids: [...clumps.keys()], grown: reducedMotion.matches })
    if (host.hidden || document.hidden) send({ type: 'pause' })
  }
  const pause = () => send({ type: 'pause' })
  const resume = () => {
    if (!host.hidden && !document.hidden) send({ type: 'resume' })
  }
  const restyle = () => {
    if (!canvas || paletteFrame) return
    // Theme listeners still have scene writes to make. Resolve their final palette together
    // on the rendering frame, without forcing layout inside the view-transition callback.
    paletteFrame = view.requestAnimationFrame(() => {
      paletteFrame = 0
      if (!canvas) return
      readPalette()
      repaint()
    })
  }
  const reweather = () => {
    if (!readClimate()) return
    send({ type: 'climate', growth: climate.growth, lean: climate.lean })
    if (canvas) repaint()
  }
  document.addEventListener('themechange', restyle, { signal })
  const seasons = new view.MutationObserver(records => {
    if (records.some(record => record.oldValue !== scene.dataset.foliageSeason)) restyle()
  })
  seasons.observe(scene, {
    attributes: true,
    attributeOldValue: true,
    attributeFilter: ['data-foliage-season'],
  })
  const weather = new view.MutationObserver(reweather)
  weather.observe(scene, { attributes: true, attributeFilter: ['data-moss'] })
  readClimate()
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      seasons.disconnect()
      weather.disconnect()
      view.clearTimeout(hitFlush)
      view.cancelAnimationFrame(paletteFrame)
      stopWorker()
    },
    { once: true },
  )
  return { plant, clear, pause, resume }
}
