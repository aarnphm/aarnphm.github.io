import type { FeatureDict, FeatureState } from './latent-client/features-types'
import { decodeCoords } from '../../util/latent/embed/coords'
import { readSharedView, sharedViewUrl, type SharedView } from '../../util/latent/share'
import { titleText } from '../../util/latent/title'
import { rowHref, type LatentDoc, type LatentKind } from '../../util/latent/types'
import { bindBenchGestures, type BenchView } from './latent-bench-gestures'
import { drawBench, fitBenchCanvas, hoverBench } from './latent-client/bench-canvas'
import { bindFeatures, snapshotFeatures } from './latent-client/features'
import { loadDict } from './latent-client/features-data'
import { featureName, termOf } from './latent-client/features-markup'
import {
  knnOf,
  previewOf,
  fetchBytes,
  fetchJson,
  type Bench,
  type Runtime,
} from './latent-client/runtime'
import { currentNavSignal } from './nav-lifecycle'
import { rootNavSignal } from './root-lifecycle'

/**
 * A latent embed in a note (`![[embeddings#focus=N#view=…]]`, docs/latent-embed.md §5–7). The
 * build rendered the whole view: an SVG poster of the bench, the features column and the
 * view's results. This turns the poster into the canvas from the build's coordinates asset
 * (no worker, no vectors) and makes the column live: a steer, a ± or a feature chip loads what
 * /embeddings would (the documents, the dictionaries, the 16 MB rows matrix) once per sidecar
 * for every embed on the page, and writes the results into the band's fixed rows.
 */

/** What every embed of one sidecar on the page shares: one worker, one rows download. */
interface Store {
  signal: AbortSignal
  docs: Promise<{ docs: LatentDoc[]; byRow: LatentDoc[] }> | null
  vec: Runtime['vec']
  knn: Runtime['knn']
  prev: Runtime['prev']
  text: Runtime['text']
  parts: Runtime['parts']
  dicts: FeatureState['dicts']
  channels: FeatureState['channels']
  rowsLoaded: boolean
}

const stores = new Map<string, Store>()

function storeFor(base: string): Store {
  const known = stores.get(base)
  if (known && !known.signal.aborted) return known
  const signal = currentNavSignal()
  const store: Store = {
    signal,
    docs: null,
    vec: null,
    knn: new Map(),
    prev: new Map(),
    text: new Map(),
    parts: new Map(),
    dicts: new Map(),
    channels: new Set(),
    rowsLoaded: false,
  }
  signal.addEventListener('abort', () => store.vec?.worker.terminate(), { once: true })
  stores.set(base, store)
  return store
}

function json<T>(raw: string | undefined, fallback: T): T {
  try {
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

/** One text node's data, written only when it changes (site-has-restyle-cost). */
function setText(el: Element | null | undefined, text: string): void {
  if (!el) return
  const node = el.firstChild
  if (node && node === el.lastChild && node.nodeType === Node.TEXT_NODE) {
    if ((node as Text).data !== text) (node as Text).data = text
  } else if (el.textContent !== text) el.textContent = text
}

export function mountLatentEmbed(figure: HTMLElement): void {
  if (figure.dataset.latentReady !== undefined) return
  const box = figure.querySelector<HTMLElement>(':scope > .lat-embed-box')
  if (!box || figure.dataset.state === 'notice' || !figure.dataset.latentBase) return
  figure.dataset.latentReady = ''
  const signal = rootNavSignal(figure)
  const data = figure.dataset
  const base = data.latentBase!
  const store = storeFor(base)
  const band = box.querySelector<HTMLOListElement>('ol.lat-embed-links')
  const column = box.querySelector<HTMLElement>('.lat-embed-features')
  const page = data.latentPage ?? '/embeddings'
  const payload = data.latentView ?? ''
  const view: SharedView | null = payload ? readSharedView(`#view=${payload}`) : null
  const focus = Number(data.latentFocus)
  const rt = {
    root: figure,
    base,
    k: Number(data.latentK),
    block: Number(data.latentBlock),
    axes: Number(data.latentAxes),
    kinds: (data.latentKinds ?? '').split(',').filter(Boolean) as LatentKind[],
    scope: (data.latentScope ?? '').split(',').filter(Boolean),
    signal,
    docs: [],
    track: [],
    knn: store.knn,
    prev: store.prev,
    text: store.text,
    parts: store.parts,
    mounting: new Map(),
    regions: new Map(),
    docByRow: [],
    focus,
    open: -1,
    rail: null,
    reader: null,
    macros: {},
    links: band ?? box,
    probe: box,
    bench: null,
    vec: null,
    semantic: null,
    model: '',
    query: null,
    railLayout: null,
    railHeight: 0,
    railItems: [],
    toast: null,
    wanted: '',
    filter: view?.filter?.length ? { kinds: [...view.filter], mask: null } : null,
    mode: 'embed',
  } as unknown as Runtime
  // one vectors worker per sidecar on the page, started by the first steer (runtime.ts vecWorker)
  Object.defineProperty(rt, 'vec', {
    get: () => store.vec,
    set: (value: Runtime['vec']) => (store.vec = value),
  })

  // ---- documents: titles for the hover line and the band, loaded on first need ----------
  const docs = () => {
    store.docs ??= fetchJson<LatentDoc[]>(rt, `${base}/docs.json`).then(list => {
      const byRow: LatentDoc[] = Array.from({ length: Number(data.latentRows) })
      for (const doc of list) byRow.fill(doc, doc.r0, doc.r0 + doc.n)
      return { docs: list, byRow }
    })
    store.docs.catch(() => (store.docs = null))
    return store.docs.then(loaded => {
      rt.docs = loaded.docs
      rt.docByRow = loaded.byRow
      rt.open = loaded.byRow[focus]?.d ?? -1
      return loaded
    })
  }

  // ---- the results band: fixed rows, rewritten in place ---------------------------------
  const limit = Number(data.latentLinks ?? 8)
  const initialBand = band?.innerHTML ?? ''
  const writeBand = async (label: string, rows: [number, number][]) => {
    if (!band) return
    await docs()
    if (signal.aborted) return
    const kinds = rt.filter?.kinds
    const kept = rows.filter(([r]) => {
      const doc = rt.docByRow[r]
      return doc && (!kinds || kinds.includes(doc.s))
    })
    rt.query = {
      text: label,
      items: kept.map(([r, score]) => {
        const doc = rt.docByRow[r]
        return {
          doc,
          score,
          inScope: true,
          preview: { r, d: doc.d, s: doc.s, t: doc.t, c: r - doc.r0, p: null, l: '', x: '' },
        }
      }),
    }
    const shown = kept.slice(0, limit)
    const head = band.querySelector(':scope > .lat-embed-links-head')
    setText(head?.children[0], label)
    setText(head?.children[1], `top ${shown.length} of ${kept.length}`)
    const pool = [...band.querySelectorAll<HTMLLIElement>(':scope > li[data-r]')]
    while (pool.length < shown.length) {
      const li = document.createElement('li')
      const swatch = document.createElement('span')
      swatch.className = 'lat-swatch'
      const a = document.createElement('a')
      a.className = 'internal'
      a.append('')
      const loc = document.createElement('span')
      loc.className = 'lat-embed-loc'
      loc.append('')
      li.append(swatch, a, loc)
      band.append(li)
      pool.push(li)
    }
    pool.forEach((li, i) => {
      const row = shown[i]
      if (!row) {
        if (li.style.display !== 'none') li.style.display = 'none'
        return
      }
      const [r, score] = row
      const doc = rt.docByRow[r]
      if (li.style.display) li.style.display = ''
      if (li.dataset.r !== String(r)) li.dataset.r = String(r)
      const swatch = li.children[0] as HTMLElement
      if (swatch.dataset.kind !== doc.s) {
        swatch.dataset.kind = doc.s
        swatch.title = doc.s
      }
      const a = li.children[1] as HTMLAnchorElement
      const href = rowHref(doc)
      if (a.getAttribute('href') !== href) a.setAttribute('href', href)
      setText(a, titleText(doc.t))
      setText(li.children[2], `${r - doc.r0 + 1}/${doc.n} · ${(score / 255).toFixed(2)}`)
      // a pdf passage opens on its page, which its preview names
      if (doc.s === 'pdf')
        void previewOf(rt, r).then(preview => {
          if (li.dataset.r === String(r) && preview?.p)
            a.setAttribute('href', rowHref(doc, preview.p))
        })
    })
    delete band.dataset.pending
  }

  // ---- the bench: canvas from the coordinates asset -------------------------------------
  const bench = box.querySelector<HTMLElement>('.lat-embed-bench')
  const poster = bench?.querySelector<SVGElement>('.lat-embed-poster')
  const status = bench?.querySelector<HTMLElement>('.lat-bench-status')
  const said = status?.textContent ?? ''
  const saved = json<{
    yaw: number
    pitch: number
    zoom: number
    panX: number
    panY: number
    axes: number[]
    hidden: number
  } | null>(data.latentBench, null)
  const level = (data.latentLevel ?? 'docs') as Bench['level']
  const home = (w: number, h: number): Partial<BenchView> =>
    saved
      ? {
          yaw: saved.yaw,
          pitch: saved.pitch,
          zoom: saved.zoom,
          panX: saved.panX * w,
          panY: saved.panY * h,
        }
      : {}
  let dirty = false
  const reset = box.querySelector<HTMLButtonElement>('.lat-embed-reset')
  const markDirty = () => {
    if (dirty) return
    dirty = true
    if (reset) reset.style.display = ''
  }

  // feature names at the features level: the atlas's point order (util/latent/atlas.ts)
  let names: string[] | null = null
  let naming: Promise<void> | null = null
  const ensureNames = () => {
    if (names || naming) return
    if (level !== 'feats') {
      naming = docs().then(() => {
        names = []
      })
    } else {
      const holder = { dicts: store.dicts } as FeatureState
      const dictNames = (data.latentFeatures ?? '').split(',').filter(Boolean)
      naming = Promise.all(dictNames.map(name => loadDict(rt, holder, name))).then(dicts => {
        const list: string[] = []
        for (const dict of dicts as FeatureDict[]) {
          if (dict.meta.dims !== 256) continue
          for (const row of dict.table)
            if (row.n > 0) list.push(`${dict.meta.name} · ${featureName(row)} · #${row.f}`)
        }
        names = list
      })
    }
    naming.catch(() => (naming = null))
  }
  const nameOf = (i: number): string => {
    if (!names) return ''
    if (level === 'feats') return names[i] ?? ''
    const doc = level === 'docs' ? rt.docs[i] : rt.docByRow[i]
    return doc ? `${doc.s} · ${titleText(doc.t)}` : ''
  }

  const mountBench = async () => {
    if (!bench || !poster || !data.latentCoords) return
    const n = Number(data.latentPoints)
    const buffer = await fetchBytes(rt, data.latentCoords)
    if (signal.aborted) return
    const asset = decodeCoords(buffer, n, Number(data.latentReach), {
      lvl: level === 'feats',
      heat: data.latentHeat !== undefined,
    })
    const canvas = document.createElement('canvas')
    canvas.className = 'lat-bench-canvas'
    canvas.tabIndex = 0
    canvas.setAttribute('role', 'img')
    canvas.setAttribute('aria-label', 'the view in 3-d: drag to turn, ⌘-scroll to zoom')
    poster.before(canvas)
    const w = canvas.clientWidth || 1
    const h = canvas.clientHeight || 1
    const axisNames = json<string[]>(data.latentAxisNames, ['x', 'y', 'z'])
    const b: Bench = {
      worker: null as unknown as Worker,
      seq: 0,
      level,
      coords: asset.coords,
      extent: Number(data.latentExtent) || 1,
      yaw: 0.6,
      pitch: 0.5,
      zoom: 1,
      panX: 0,
      panY: 0,
      ...home(w, h),
      kinds: asset.kinds,
      hidden: saved?.hidden ?? 0,
      links: null,
      picked: false,
      axes: [0, 1, 2],
      meta: {
        scale: 1,
        mean: [],
        axes: axisNames.map((name, i) => ({ id: String(i), name, dir: [] })),
        rows: Number(data.latentRows),
        docs: Number(data.latentDocs),
      },
      feature: null,
      featsOn: level === 'feats',
      atlas: null,
      heat: asset.heat,
      heatKey: '',
      frame: 0,
      hover: -1,
      w,
      h,
      dpr: Math.min(3, window.devicePixelRatio || 1),
      embed: {
        mark: Number(data.latentMark ?? -1),
        edges: json(data.latentEdges, null),
        lvl: asset.lvl,
        name: nameOf,
      },
    }
    canvas.width = Math.round(w * b.dpr)
    canvas.height = Math.round(h * b.dpr)
    rt.bench = b
    drawBench(rt)
    // the canvas's first frame lands, then the poster goes, in the same frame (style.display:
    // `hidden` is a :has() argument)
    requestAnimationFrame(() => {
      poster.style.display = 'none'
    })
    fitBenchCanvas(rt, canvas)
    bindBenchGestures(canvas, signal, {
      view: () => rt.bench,
      hovering: () => b.hover >= 0,
      hover: (x, y) => {
        ensureNames()
        hoverBench(rt, x, y)
        if (b.hover < 0) setText(status, said)
      },
      redraw: () => {
        drawBench(rt)
        markDirty()
      },
      open: () => pick(b),
      home: () => home(b.w, b.h),
    })
    canvas.addEventListener(
      'pointerleave',
      () => {
        if (b.hover < 0) return
        b.hover = -1
        drawBench(rt)
        setText(status, said)
      },
      { signal },
    )
    // keyboard: arrows turn, + and - zoom, 0 restores the view
    canvas.addEventListener(
      'keydown',
      event => {
        const turn: Record<string, [number, number]> = {
          ArrowLeft: [-0.1, 0],
          ArrowRight: [0.1, 0],
          ArrowUp: [0, -0.05],
          ArrowDown: [0, 0.05],
        }
        if (event.key in turn) {
          const [dy, dp] = turn[event.key]
          b.yaw += dy
          b.pitch = Math.max(0.05, Math.min(1.2, b.pitch + dp))
        } else if (event.key === '+' || event.key === '=') b.zoom = Math.min(12, b.zoom * 1.15)
        else if (event.key === '-') b.zoom = Math.max(0.4, b.zoom / 1.15)
        else if (event.key === '0') Object.assign(b, home(b.w, b.h))
        else return
        event.preventDefault()
        drawBench(rt)
        markDirty()
      },
      { signal },
    )
    // the scroll guard: the wheel scrolls the note until the bench is engaged (a click or drag
    // on it, until the pointer leaves the figure) or ⌘/ctrl is held
    let engaged = false
    let hint = 0
    canvas.addEventListener('pointerdown', () => (engaged = true), { signal })
    figure.addEventListener('pointerleave', () => (engaged = false), { signal })
    bench.addEventListener(
      'wheel',
      event => {
        if (event.target !== canvas || engaged || event.ctrlKey || event.metaKey) return
        event.stopPropagation()
        if (!status) return
        setText(status, '⌘-scroll to zoom, or click the bench first')
        window.clearTimeout(hint)
        hint = window.setTimeout(() => setText(status, said), 1500)
      },
      { signal, capture: true },
    )
  }

  // a click on a point names it on the status line, linking where it lives; the focus stays
  const pick = (b: Bench) => {
    if (b.hover < 0 || !status) return
    ensureNames()
    if (level === 'feats') return
    void docs().then(() => {
      const doc = level === 'docs' ? rt.docs[b.hover] : rt.docByRow[b.hover]
      if (!doc) return
      const a = document.createElement('a')
      a.className = 'internal'
      a.href = rowHref(doc)
      a.textContent = `${doc.s} · ${titleText(doc.t)} ↗`
      status.replaceChildren(a)
    })
  }

  // ---- the features column ----------------------------------------------------------------
  let st: FeatureState | null = null
  const initialColumn = column?.innerHTML ?? ''
  const seedRows = () =>
    band
      ? [...band.querySelectorAll<HTMLElement>(':scope > li[data-r]')].map(li =>
          Number(li.dataset.r),
        )
      : null
  let waking: Promise<void> | null = null
  const wake = () => {
    if (waking || !column) return waking
    waking = docs().then(() => {
      if (signal.aborted) return
      st = bindFeatures(rt, {
        panel: column,
        seed: view?.features ?? null,
        pinned: seedRows(),
        pin: async (label, rows) => {
          store.rowsLoaded = true
          // the explorer heads a steered list with every term; the band says how many, as the
          // build did
          const live = st ? [...st.steer.values()].filter(t => t.w !== 0).length : 0
          const short =
            live && label.startsWith('focus ')
              ? `focus, steered by ${live} term${live === 1 ? '' : 's'}`
              : label
          await writeBand(short, rows)
          if (status && status.dataset.loading !== undefined) {
            delete status.dataset.loading
            setText(status, said)
          }
          markDirty()
        },
        unpin: () => {
          // nothing steers: the focus's own neighbours
          void knnOf(rt, focus).then(links =>
            writeBand(
              'focus links',
              links.neighbours.map(n => [n.r, Math.round(Math.max(0, n.score) * 255)]),
            ),
          )
          markDirty()
        },
        dicts: store.dicts,
        channels: store.channels,
      })
      // a slider moved while the documents loaded: steer from where it was left
      for (const input of column.querySelectorAll<HTMLInputElement>('input[data-steer-range]')) {
        const row = input.closest<HTMLElement>('[data-term]')
        if (row && Number(row.dataset.w).toFixed(2) !== Number(input.value).toFixed(2)) {
          input.dispatchEvent(new Event('input', { bubbles: true }))
          input.dispatchEvent(new Event('change', { bubbles: true }))
        }
      }
    })
    waking.catch(error => {
      waking = null
      if (!signal.aborted) console.warn('[latent] embed', error)
    })
    return waking
  }
  if (column) {
    for (const input of column.querySelectorAll<HTMLInputElement>('input[disabled]'))
      input.disabled = false
    for (const type of ['pointerdown', 'focusin', 'keydown'] as const)
      column.addEventListener(type, () => void wake(), { signal, passive: true })
    // the first steer fetches the rows matrix: say so under the bench while it comes
    column.addEventListener(
      'input',
      () => {
        if (band && band.dataset.pending === undefined) band.dataset.pending = ''
        if (!store.rowsLoaded && status && status.dataset.loading === undefined) {
          status.dataset.loading = ''
          setText(status, 'loading 16 MB of rows for steering…')
        }
      },
      { signal },
    )
    // a click that lands before the column is bound (the documents still loading) is replayed
    // once it is
    column.addEventListener(
      'click',
      event => {
        if (st) return
        const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button')
        if (!button) return
        event.preventDefault()
        event.stopImmediatePropagation()
        void wake()?.then(() => {
          if (st && button.isConnected) button.click()
        })
      },
      { signal, capture: true },
    )
    column.addEventListener(
      'click',
      event => {
        const target = event.target as HTMLElement
        if (
          !target.closest('button[data-steer], button[data-steer-drop], button[data-steer-clear]')
        )
          return
        if (band && band.dataset.pending === undefined) band.dataset.pending = ''
      },
      { signal, capture: true },
    )
    // the explorer's `?focus=N` links open /embeddings from inside a note
    const fix = (event: Event) => {
      const a = (event.target as HTMLElement).closest?.<HTMLAnchorElement>('a[href^="?focus="]')
      if (a) a.setAttribute('href', `${page}${a.getAttribute('href')}`)
    }
    column.addEventListener('pointerover', fix, { signal })
    column.addEventListener('focusin', fix, { signal })
  }

  // ---- head: open ↗ carries the view as it stands; reset restores the authored one ---------
  const open = box.querySelector<HTMLAnchorElement>('a.lat-embed-open')
  const snapshot = (): SharedView | null => {
    if (!view) return null
    const b = rt.bench
    return {
      ...view,
      features: st ? (snapshotFeatures(rt) ?? view.features) : view.features,
      bench:
        b && view.bench
          ? {
              ...view.bench,
              yaw: b.yaw,
              pitch: b.pitch,
              zoom: b.zoom,
              panX: b.panX / Math.max(1, b.w),
              panY: b.panY / Math.max(1, b.h),
            }
          : view.bench,
    }
  }
  if (open && view) {
    const refresh = () => {
      if (!dirty) return
      const next = snapshot()
      if (next)
        open.setAttribute(
          'href',
          sharedViewUrl(new URL(`${page}?focus=${focus}`, location.href).href, next),
        )
    }
    open.addEventListener('pointerenter', refresh, { signal })
    open.addEventListener('focus', refresh, { signal })
  }
  reset?.addEventListener(
    'click',
    () => {
      dirty = false
      reset.style.display = 'none'
      const b = rt.bench
      if (b) {
        Object.assign(b, home(b.w, b.h))
        b.hover = -1
        drawBench(rt)
      }
      setText(status, said)
      if (status) delete status.dataset.loading
      if (st) {
        // the authored steering, card and list, without refetching anything
        st.steer.clear()
        for (const [key, w] of view?.features?.steer ?? []) st.steer.set(key, termOf(key, w))
        st.selected = view?.features?.selected ? { ...view.features.selected } : null
        st.explain = view?.features?.explain ?? true
        st.steerSeq++
        st.renderSeq++
      }
      if (column) column.innerHTML = initialColumn
      if (column)
        for (const input of column.querySelectorAll<HTMLInputElement>('input[disabled]'))
          input.disabled = false
      if (band) {
        band.innerHTML = initialBand
        delete band.dataset.pending
      }
      if (st) st.pinned = seedRows()
      if (open)
        open.setAttribute('href', `${page}?focus=${focus}${payload ? `#view=${payload}` : ''}`)
    },
    { signal },
  )

  void mountBench().catch(error => {
    if (!signal.aborted) console.warn('[latent] embed bench', error)
  })
}
