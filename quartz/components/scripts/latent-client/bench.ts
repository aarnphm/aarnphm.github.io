import type { LatentKind } from '../../../util/latent/types'
import { buildSelect } from '../../controls/select'
import { BENCH_VIEW, bindBenchGestures } from '../latent-bench-gestures'
import { axisBase, drawBench, fitBenchCanvas, hoverBench } from './bench-canvas'
import { goTo } from './reader'
import { metaOf, requestVec, vecWorker, type Bench, type Runtime } from './runtime'

// ---- 3-d bench --------------------------------------------------------------------------

const KIND_INDEX: Record<LatentKind, number> = { note: 0, pdf: 1, image: 2, base: 3 }
// named as the sources ledger names them
const KIND_LABEL: Record<LatentKind, string> = {
  note: 'notes',
  pdf: 'pdf',
  image: 'images',
  base: 'bases',
}

export async function openBench(rt: Runtime): Promise<void> {
  if (rt.bench) return
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  const canvas = rt.probe.querySelector<HTMLCanvasElement>('.lat-bench-canvas')
  const controls = rt.probe.querySelector<HTMLElement>('.lat-bench-controls')
  if (!status || !canvas || !controls) return
  status.textContent = 'loading document vectors…'
  const { worker } = vecWorker(rt)
  const bench: Bench = {
    worker,
    seq: 0,
    level: 'docs',
    coords: null,
    extent: 1,
    ...BENCH_VIEW,
    kinds: null,
    hidden: 0,
    links: null,
    picked: false,
    axes: [0, 1, 2],
    meta: null,
    frame: 0,
    hover: -1,
    w: canvas.clientWidth || canvas.width,
    h: canvas.clientHeight || canvas.height,
    dpr: 1,
  }
  rt.bench = bench
  fitBenchCanvas(rt, canvas)
  const meta = await metaOf(rt)
  bench.meta = meta
  // open on the first three principal components; a probe without them keeps the first three axes
  const pcs = [1, 2, 3].map(n => meta.axes.findIndex(axis => axisBase(axis.name) === `pc${n}`))
  if (pcs.every(i => i >= 0)) bench.axes = pcs
  bench.kinds = new Uint8Array(rt.docs.length)
  for (const doc of rt.docs) bench.kinds[doc.d] = KIND_INDEX[doc.s]
  // x, y, z and the level as rows of a ruled table: the axis and a garden select for its
  // component. The menus float in the top layer because the probe scrolls
  // and would clip them
  const table = document.createElement('table')
  table.className = 'lat-bench-table'
  const body = document.createElement('tbody')
  table.append(body)
  const row = (label: string | null, ...cells: HTMLTableCellElement[]) => {
    const tr = document.createElement('tr')
    if (label !== null) {
      const th = document.createElement('th')
      th.scope = 'row'
      th.textContent = label
      tr.append(th)
    }
    tr.append(...cells)
    body.append(tr)
  }
  const cell = (className: string, ...children: (Node | string)[]) => {
    const td = document.createElement('td')
    td.className = className
    td.append(...children)
    return td
  }
  // the variance share is on the axes tab; here an axis is its name
  const options = meta.axes.map((axis, i) => ({ value: i, label: axisBase(axis.name) }))
  for (const slot of [0, 1, 2]) {
    const name = 'xyz'[slot]
    const select = buildSelect<number>({
      id: `lat-bench-${name}`,
      label: `${name} axis`,
      groups: [{ options }],
      selected: bench.axes[slot],
      onSelect: value => {
        bench.axes[slot] = value
        void projectBench(rt)
      },
      floating: true,
    })
    const unmount = select.mount()
    rt.signal.addEventListener('abort', unmount, { once: true })
    row(name, cell('lat-bench-pick', select.element))
  }
  const level = (value: 'docs' | 'rows', text: string) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'lat-bench-level'
    button.dataset.level = value
    button.setAttribute('aria-pressed', String(value === 'docs'))
    button.textContent = text
    return button
  }
  const levels = cell('lat-bench-levels', level('docs', 'documents'), level('rows', 'chunks'))
  // the sources share the level's line; each button is also the legend for its colour
  const source = (kind: LatentKind) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'lat-bench-source'
    button.dataset.kind = kind
    button.setAttribute('aria-pressed', 'true')
    button.textContent = KIND_LABEL[kind]
    return button
  }
  const kinds = Object.keys(KIND_INDEX) as LatentKind[]
  const sources = cell('lat-bench-levels lat-bench-sources', ...kinds.map(source))
  sources.setAttribute('role', 'group')
  sources.setAttribute('aria-label', 'sources')
  // documents and chunks name the level themselves, so the row has no heading
  levels.setAttribute('role', 'group')
  levels.setAttribute('aria-label', 'level')
  row(null, levels, sources)
  controls.replaceChildren(table)
  controls.addEventListener(
    'click',
    event => {
      const toggle = (event.target as HTMLElement).closest<HTMLButtonElement>('.lat-bench-source')
      if (toggle) {
        const shown = toggle.getAttribute('aria-pressed') !== 'true'
        toggle.setAttribute('aria-pressed', String(shown))
        bench.hidden ^= 1 << KIND_INDEX[toggle.dataset.kind as LatentKind]
        if (bench.hover >= 0 && bench.kinds && (bench.hidden >> bench.kinds[bench.hover]) & 1)
          bench.hover = -1
        drawBench(rt)
        return
      }
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.lat-bench-level')
      if (!button || button.getAttribute('aria-pressed') === 'true') return
      for (const other of controls.querySelectorAll<HTMLButtonElement>('.lat-bench-level'))
        other.setAttribute('aria-pressed', String(other === button))
      void loadBench(rt, button.dataset.level === 'rows' ? 'rows' : 'docs')
    },
    { signal: rt.signal },
  )
  bindFoot(status)
  bindBenchPointer(rt, canvas)
  await loadBench(rt, 'docs')
}

async function loadBench(rt: Runtime, level: 'docs' | 'rows'): Promise<void> {
  const bench = rt.bench
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  if (!bench || !bench.meta || !status) return
  bench.level = level
  const rows = level === 'docs' ? bench.meta.docs : bench.meta.rows
  const url = `${rt.base}/vec/${level}-256-int8.bin`
  status.textContent = `loading ${level === 'docs' ? 'document' : 'chunk'} vectors (${((rows * 256) / 1e6).toFixed(1)} MB)…`
  const t0 = performance.now()
  await requestVec(rt, { type: 'load', url, rows, dims: 256 })
  if (level === 'rows') {
    bench.kinds = new Uint8Array(rows)
    for (const doc of rt.docs) bench.kinds.fill(KIND_INDEX[doc.s], doc.r0, doc.r0 + doc.n)
  } else {
    bench.kinds = new Uint8Array(rt.docs.length)
    for (const doc of rt.docs) bench.kinds[doc.d] = KIND_INDEX[doc.s]
  }
  status.textContent = `${rows} ${level === 'docs' ? 'documents' : 'chunks'} in ${(performance.now() - t0).toFixed(0)} ms`
  await projectBench(rt)
}

async function projectBench(rt: Runtime): Promise<void> {
  const bench = rt.bench
  if (!bench || !bench.meta) return
  const dirs = new Float32Array(3 * 256)
  bench.axes.forEach((axis, slot) => dirs.set(bench.meta!.axes[axis].dir, slot * 256))
  const mean = new Float32Array(bench.meta.mean)
  const t0 = performance.now()
  const url = `${rt.base}/vec/${bench.level}-256-int8.bin`
  const result = await requestVec(
    rt,
    { type: 'project', url, dirs, mean, scale: bench.meta.scale },
    [dirs.buffer, mean.buffer],
  )
  if (result.type !== 'projected') return
  bench.coords = result.coords
  bench.extent = result.extent || 1
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  if (status)
    status.textContent = `${bench.coords.length / 3} points · projected in ${(performance.now() - t0).toFixed(0)} ms`
  drawBench(rt)
}

// the status line sits in a foot under the canvas (latent.scss sets its spacing)
function bindFoot(status: HTMLElement): void {
  const foot = document.createElement('div')
  foot.className = 'lat-bench-foot'
  status.before(foot)
  foot.append(status)
}

function bindBenchPointer(rt: Runtime, canvas: HTMLCanvasElement): void {
  bindBenchGestures(canvas, rt.signal, {
    view: () => rt.bench,
    hovering: () => (rt.bench?.hover ?? -1) >= 0,
    hover: (x, y) => hoverBench(rt, x, y),
    redraw: () => drawBench(rt),
    open: () => {
      const bench = rt.bench
      if (!bench || bench.hover < 0) return
      bench.picked = true
      drawBench(rt)
      const row = bench.level === 'docs' ? rt.docs[bench.hover].r0 : bench.hover
      void goTo(rt, row)
    },
  })
}
