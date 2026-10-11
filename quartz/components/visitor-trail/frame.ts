import { playhtml } from 'playhtml'
import { clearSpace, trailLayout, type TrailId } from './layout'
import { trailTrees } from './scenery'

type TrailData = { stones: Record<string, number>; routes?: Record<string, TrailId> }
type TrailSample = { x: number; y: number; position: number }
type TrailGeometry = { id: TrailId; length: number; samples: TrailSample[] }
const trailIds: TrailId[] = ['main', 'pines', 'grove']

const document = parent.document
const host = window.frameElement?.parentElement
if (!host?.matches('.homepage-trail-scene') || parent === window)
  throw new Error('The trail needs its homepage host')
const controller = new AbortController()
const { signal } = controller
const scene = document.createElement('div')
scene.className = 'trail-scene'
const clip = `trail-clear-${crypto.randomUUID()}`
scene.innerHTML = `
    <svg class="trail-canvas" aria-hidden="true">
      <defs><clipPath id="${clip}"><path data-trail-clear clip-rule="evenodd"/></clipPath></defs>
      ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
      <g clip-path="url(#${clip})">
      <g class="trail-steps"></g>
      <g class="trail-landmarks"></g>
      <g data-trail-stones></g>
      </g>
    </svg>
    <div class="trail-stone" role="slider" aria-label="Your stone on the shared trail" tabindex="-1"
      aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-describedby="trail-hint" aria-disabled="true"
      aria-keyshortcuts="Shift+ArrowLeft Shift+ArrowRight">
      <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 8 4 3 8 2 11 7 9 10 3 10Z"/></svg>
    </div>
    <div class="trail-campsite">
      <button class="trail-camp-toggle" type="button" aria-label="Trail controls" aria-expanded="false" aria-controls="trail-camp-tools">
        <svg viewBox="0 0 64 64" aria-hidden="true"><g class="trail-camp"></g></svg>
      </button>
      <div class="trail-camp-tools" id="trail-camp-tools" hidden>
        <span id="trail-hint">drag between trails; arrow keys walk, shift + arrows switch</span>
        <div class="trail-tools">
          <button type="button" data-trail-retry hidden>reconnect</button>
        </div>
        <span class="trail-status" data-trail-count role="status">connecting</span>
      </div>
    </div>`
host.append(scene)
const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
const stones = scene.querySelector<SVGGElement>('[data-trail-stones]')
const stone = scene.querySelector<HTMLElement>('.trail-stone')
const count = scene.querySelector<HTMLElement>('[data-trail-count]')
const retry = scene.querySelector<HTMLButtonElement>('[data-trail-retry]')
const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
const campToggle = scene.querySelector<HTMLButtonElement>('.trail-camp-toggle')
const campTools = scene.querySelector<HTMLElement>('.trail-camp-tools')
if (!svg || !stones || !stone || !count || !retry || !campsite || !campToggle || !campTools)
  throw new Error('Shared trail markup is incomplete')

let campTimer = 0
let campPinned = false
const showTools = (open: boolean) => {
  parent.clearTimeout(campTimer)
  campTools.hidden = !open
  campToggle.setAttribute('aria-expanded', String(open))
}
const leaveCamp = () => {
  parent.clearTimeout(campTimer)
  campTimer = parent.setTimeout(() => {
    if (campPinned || campsite.contains(document.activeElement)) return
    showTools(false)
  }, 180)
}
campsite.addEventListener(
  'pointerenter',
  event => {
    if (event.pointerType !== 'touch') showTools(true)
  },
  { signal },
)
campsite.addEventListener('pointerleave', leaveCamp, { signal })
campsite.addEventListener('focusin', () => showTools(true), { signal })
campsite.addEventListener('focusout', leaveCamp, { signal })
campToggle.addEventListener(
  'click',
  () => {
    campPinned = !campPinned
    showTools(campPinned)
  },
  { signal },
)
campsite.addEventListener(
  'keydown',
  event => {
    if (event.key !== 'Escape' || campTools.hidden) return
    event.preventDefault()
    event.stopPropagation()
    campPinned = false
    campToggle.focus({ preventScroll: true })
    showTools(false)
  },
  { signal },
)
retry.addEventListener('click', () => location.reload(), { signal })

let ownPosition = 0.1
let ownRoute: TrailId = 'main'
let geometries: TrailGeometry[] = []
let redrawStones = () => {}
let layoutSignature = ''
let obstacles: { left: number; right: number; top: number; bottom: number }[] = []
const point = (position: number, route: TrailId = ownRoute) => {
  const geometry = geometries.find(geometry => geometry.id === route)
  if (!geometry) return { x: 0, y: 0 }
  const { length, samples } = geometry
  const index = Math.min(samples.length - 1, Math.floor((position * length) / 4))
  const a = samples[index]
  const b = samples[Math.min(index + 1, samples.length - 1)]
  if (!a || !b) return { x: 0, y: 0 }
  const fraction =
    a.position === b.position ? 0 : (position - a.position) / (b.position - a.position)
  return { x: a.x + (b.x - a.x) * fraction, y: a.y + (b.y - a.y) * fraction }
}
const placeOwnStone = (position: number, route: TrailId = ownRoute) => {
  ownPosition = position
  ownRoute = route
  stone.dataset.route = route
  const p = point(position)
  stone.style.left = `${p.x}px`
  stone.style.top = `${p.y}px`
  // A slider's generous hit area must leave the adjacent text selectable.
  const cuts = obstacles.filter(
    rect =>
      rect.left < p.x + 22 &&
      rect.right > p.x - 22 &&
      rect.top < p.y + 22 &&
      rect.bottom > p.y - 22,
  )
  const outline = clearSpace(
    44,
    44,
    cuts.map(rect => ({
      left: rect.left - p.x + 22,
      right: rect.right - p.x + 22,
      top: rect.top - p.y + 22,
      bottom: rect.bottom - p.y + 22,
    })),
  )
  stone.style.clipPath = `path(evenodd, '${outline}')`
  stone.setAttribute('aria-valuenow', String(Math.round(position * 100)))
  stone.setAttribute(
    'aria-valuetext',
    `${route} trail, ${Math.round(position * 100)} percent along the path`,
  )
}
const drawScene = () => {
  const layout = trailLayout(scene)
  if (!layout || !scene.isConnected) return
  const signature = `${JSON.stringify(layout.routes)}|${layout.clear}`
  if (signature === layoutSignature) return
  layoutSignature = signature
  const { width, height, bottom } = layout
  obstacles = layout.obstacles
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  svg.querySelector('[data-trail-clear]')?.setAttribute('d', layout.clear)
  geometries = trailIds.flatMap(id => {
    const path = svg.querySelector<SVGPathElement>(`[data-trail-route="${id}"]`)
    if (!path) return []
    path.setAttribute('d', layout.routes[id])
    const length = path.getTotalLength()
    const samples = Array.from({ length: Math.ceil(length / 4) + 1 }, (_, index) => {
      const position = Math.min(1, (index * 4) / length)
      const p = path.getPointAtLength(position * length)
      return { position, x: p.x, y: p.y }
    })
    return [{ id, length, samples }]
  })
  const steps = scene.querySelector('.trail-steps')
  if (steps) {
    steps.innerHTML = geometries
      .map(({ id, length }) => {
        const marks = []
        for (let distance = 6, step = 0; distance < length - 6; distance += 12, step++) {
          const p = point(distance / length, id)
          const ahead = point(Math.min(1, (distance + 1) / length), id)
          const angle = Math.atan2(ahead.y - p.y, ahead.x - p.x)
          const side = step % 2 ? 2 : -2
          const x = p.x - Math.sin(angle) * side
          const y = p.y + Math.cos(angle) * side
          marks.push(
            `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="2" ry="1.3" transform="rotate(${((angle * 180) / Math.PI).toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})"/>`,
          )
        }
        return `<g data-route="${id}">${marks.join('')}</g>`
      })
      .join('')
  }
  campsite.style.left = `${layout.camp.x}px`
  campsite.style.top = `${layout.camp.y}px`
  const camp = scene.querySelector('.trail-camp')
  if (camp) {
    camp.setAttribute('transform', 'translate(32 48)')
    // The 404's hearth uses five flame columns, logs and a ring of stones.
    camp.innerHTML = `<ellipse class="trail-hearth" cx="0" cy="2" rx="25" ry="8"/>
      <path class="trail-log" d="M-12 1 10 5M-10 5 12 1"/>
      ${[2, 4, 7, 5, 2]
        .map(
          (rows, column) =>
            `<g class="trail-flame" style="--flicker:${560 + column * 90}ms;--phase:${column * -170}ms">${Array.from(
              { length: rows },
              (_, row) => {
                const t = (row + 0.5) / rows
                const tone =
                  t < 0.34 && column > 0 && column < 4
                    ? 'core'
                    : t < 0.6
                      ? 'gold'
                      : t < 0.85
                        ? 'orange'
                        : 'tip'
                return `<rect class="trail-flame-${tone}" x="${(column - 2) * 3}" y="${-(row + 1) * 3}" width="3" height="3"/>`
              },
            ).join('')}</g>`,
        )
        .join('')}
      <g class="trail-smoke"><rect x="0" y="-32" width="3" height="3"/><rect x="6" y="-43" width="3" height="3"/></g>
      <g class="trail-rocks"><rect x="-17" y="3" width="5" height="3"/><rect x="13" y="3" width="5" height="3"/><rect x="-8" y="9" width="5" height="3"/><rect x="4" y="9" width="5" height="3"/></g>
      `
  }
  const landmarks = scene.querySelector('.trail-landmarks')
  if (landmarks) landmarks.innerHTML = trailTrees(width, bottom, layout.camp.y)
  placeOwnStone(ownPosition)
  redrawStones()
}
host.addEventListener('trail:reflow', drawScene, { signal })
drawScene()

const unavailable = () => {
  count.textContent = 'trail unavailable'
  retry.hidden = false
  scene.dataset.trailState = 'unavailable'
}
const timeout = parent.setTimeout(unavailable, 12000)
let disposeConnection = () => {}
const dispose = () => {
  if (signal.aborted) return
  controller.abort()
  parent.clearTimeout(campTimer)
  parent.clearTimeout(timeout)
  disposeConnection()
  scene.remove()
}
host.addEventListener('trail:dispose', dispose, { once: true, signal })
window.addEventListener('pagehide', dispose, { once: true })

const connect = async () => {
  await playhtml.init({ room: '/shared-trail-v1', host: 'api.playhtml.fun', onError: unavailable })
  if (signal.aborted || !scene.isConnected) return
  parent.clearTimeout(timeout)
  const id = playhtml.users.me.pid
  const channel = playhtml.createPageData<TrailData>('mountain-trail-v1', {
    stones: {},
    routes: {},
  })
  let position = 0.1
  let route: TrailId = 'main'
  let pointer: number | null = null
  let frame = 0
  const readPosition = (value: unknown) =>
    typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null
  const readRoute = (value: unknown): TrailId =>
    value === 'pines' || value === 'grove' ? value : 'main'

  const render = (data: TrailData) => {
    if (!data || typeof data !== 'object' || !data.stones || typeof data.stones !== 'object') return
    const entries = Object.entries(data.stones).filter(([, value]) => readPosition(value) !== null)
    stones.replaceChildren()
    for (const [visitor, value] of entries) {
      if (visitor === id) continue
      const position = readPosition(value)
      if (position === null) continue
      const visitorRoute = readRoute(data.routes?.[visitor])
      const p = point(position, visitorRoute)
      const pebble = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      pebble.setAttribute('d', 'M-3 1-2-2 1-3 3 0 2 2-2 2Z')
      pebble.setAttribute('transform', `translate(${p.x} ${p.y})`)
      pebble.dataset.route = visitorRoute
      stones.append(pebble)
    }
    const saved = readPosition(data.stones[id])
    if (saved !== null && pointer === null && frame === 0) {
      position = saved
      if (data.routes?.[id] !== undefined) route = readRoute(data.routes[id])
      placeOwnStone(position, route)
    }
    const label = `${entries.length} ${entries.length === 1 ? 'stone' : 'stones'}`
    if (count.textContent !== label) count.textContent = label
    retry.hidden = true
    scene.dataset.trailState = 'ready'
  }
  const publish = () =>
    channel.setData(data => {
      data.stones[id] = position
      data.routes ??= {}
      data.routes[id] = route
    })
  const move = (next: number, nextRoute: TrailId = route) => {
    position = Math.max(0, Math.min(1, next))
    route = nextRoute
    placeOwnStone(position, route)
    // Coalesce pointer traffic to one shared update per animation frame.
    if (!frame)
      frame = parent.requestAnimationFrame(() => {
        frame = 0
        publish()
      })
  }
  const initial = readPosition(channel.getData().stones[id])
  if (initial === null) {
    // A stable identity spreads first visits along the trail and deduplicates concurrent tabs.
    const hash = [...id].reduce((total, letter) => (total * 31 + letter.charCodeAt(0)) >>> 0, 0)
    position = 0.08 + (hash % 840) / 1000
    route = trailIds[hash % trailIds.length]
    publish()
  }
  channel.onUpdate(data => {
    // Concurrent first visits can replace the initial map. Restore only our entry on sync.
    if (
      readPosition(data.stones[id]) === null ||
      (pointer === null && frame === 0 && route !== 'main' && data.routes?.[id] === undefined)
    )
      publish()
    render(data)
  })
  redrawStones = () => render(channel.getData())
  render(channel.getData())
  stone.setAttribute('aria-disabled', 'false')
  stone.tabIndex = 0

  const nearest = (target: { x: number; y: number }, routes: TrailId[]) => {
    let best = { position, route }
    let distance = Infinity
    for (const id of routes) {
      const geometry = geometries.find(geometry => geometry.id === id)
      if (!geometry) continue
      for (const sample of geometry.samples) {
        const squared = (sample.x - target.x) ** 2 + (sample.y - target.y) ** 2
        if (squared < distance) {
          distance = squared
          best = { position: sample.position, route: id }
        }
      }
    }
    return best
  }
  stone.addEventListener('keydown', event => {
    if (event.shiftKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      event.preventDefault()
      event.stopPropagation()
      const offset = event.key === 'ArrowRight' ? 1 : -1
      const nextRoute =
        trailIds[(trailIds.indexOf(route) + offset + trailIds.length) % trailIds.length]
      const next = nearest(point(position, route), [nextRoute])
      move(next.position, next.route)
      stone.scrollIntoView({ block: 'nearest', inline: 'nearest' })
      return
    }
    const length = geometries.find(geometry => geometry.id === route)?.length
    if (!length) return
    const steps: Record<string, number> = {
      ArrowRight: 16 / length,
      ArrowUp: 16 / length,
      ArrowLeft: -16 / length,
      ArrowDown: -16 / length,
      PageUp: 160 / length,
      PageDown: -160 / length,
    }
    if (event.key === 'Home' || event.key === 'End' || event.key in steps) {
      event.preventDefault()
      event.stopPropagation()
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : position + steps[event.key]
      if (next < 0 && route === 'grove') {
        const stem = nearest(point(0, route), ['main'])
        const main = geometries.find(geometry => geometry.id === 'main')
        if (main) move(stem.position + (next * length) / main.length, 'main')
      } else move(next)
      stone.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    }
  })
  const project = (event: PointerEvent) => {
    const matrix = svg.getScreenCTM()
    if (!matrix) return
    const target = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    const best = nearest(target, [route, ...trailIds.filter(id => id !== route)])
    move(best.position, best.route)
  }
  stone.addEventListener('pointerdown', event => {
    if (pointer !== null || event.button !== 0) return
    pointer = event.pointerId
    stone.setPointerCapture(pointer)
    stone.dataset.dragging = 'true'
    event.preventDefault()
    stone.focus({ preventScroll: true })
  })
  stone.addEventListener('pointermove', event => {
    if (event.pointerId === pointer) project(event)
  })
  const finish = (event: PointerEvent) => {
    if (event.pointerId !== pointer) return
    if (stone.hasPointerCapture(event.pointerId)) stone.releasePointerCapture(event.pointerId)
    pointer = null
    delete stone.dataset.dragging
    if (frame) {
      parent.cancelAnimationFrame(frame)
      frame = 0
    }
    publish()
  }
  stone.addEventListener('pointerup', finish)
  stone.addEventListener('pointercancel', finish)
  stone.addEventListener('lostpointercapture', finish)
  disposeConnection = () => {
    if (frame) {
      parent.cancelAnimationFrame(frame)
      frame = 0
      publish()
    }
    channel.destroy()
  }
}
void connect().catch(unavailable)
