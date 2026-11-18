import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { visit } from 'unist-util-visit'
import type {
  FeatureDict,
  FeatureState,
} from '../../../components/scripts/latent-client/features-types'
import type { Runtime } from '../../../components/scripts/latent-client/runtime'
import type { SharedBench, SharedView } from '../share'
import { linkItems } from '../../../components/scripts/latent-client/document-list'
import {
  codeBlock,
  loadDict,
  steerTables,
  weights,
} from '../../../components/scripts/latent-client/features-data'
import { featureName, termOf } from '../../../components/scripts/latent-client/features-markup'
import { panelGroups } from '../../../components/scripts/latent-client/features-panel'
import { steeredRows } from '../../../components/scripts/latent-client/features-steering'
import { atlasFrom, projectAtlas, type Atlas } from '../atlas'
import { rowMask } from '../filter'
import { docOfRow, type Sidecar } from '../sidecar'
import { titleText } from '../title'
import { rowHref, type LatentKind } from '../types'
import { encodeCoords } from './coords'
import { decodeLatentView, parseLatentFragment } from './fragment'
import { nodeFeatureState, nodeLatent, nodeRuntime } from './node'
import { benchPoster, posterCount } from './poster'

/**
 * `![[embeddings#focus=N#view=…]]` in a note: a quotation of a view of /embeddings, rendered
 * whole at build time from the sidecar on disk (docs/latent-embed.md). The bench is an SVG
 * poster at the view's level, axes and camera; the features column is the client's own panel
 * builders run over the sidecar folder; the links are the view's steered results from the
 * client's own fusion, in-process. The client (latent-embed.client.ts) hydrates it on intent.
 */

export type EmbedPanes = 'both' | 'bench' | 'features'

export interface EmbedSettings {
  panes: EmbedPanes
  /** result rows in the band, 0..16; 0 drops it */
  links: number
  wide: boolean
}

export interface EmbedResult {
  figure: Element
  warnings: string[]
}

const KIND_INDEX: Record<LatentKind, number> = { note: 0, pdf: 1, image: 2, base: 3 }
const KIND_LABEL: Record<LatentKind, string> = {
  note: 'notes',
  pdf: 'pdf',
  image: 'images',
  base: 'bases',
}

export function embedSettings(
  metadata: Readonly<Record<string, unknown>> | undefined,
): EmbedSettings {
  const panes = metadata?.panes
  const links = Number(metadata?.links)
  return {
    panes: panes === 'bench' || panes === 'features' ? panes : 'both',
    links: Number.isFinite(links) ? Math.max(0, Math.min(16, Math.round(links))) : 8,
    wide: metadata?.wide !== false,
  }
}

/** `/embeddings` for `embeddings/index`. */
export function pageHref(slug: string): string {
  return `/${slug.replace(/(?:^|\/)index$/, '')}`
}

function axisBase(name: string): string {
  return name.replace(/\s+·\s+[\d.]+\s*%$/, '')
}

/** The ledger box with one line: a view that cannot be shown, and where it still opens. */
function notice(text: string, href: string | null, caption: string | undefined): Element {
  return h('figure.latent-embed', { dataLatentEmbed: '', dataState: 'notice' }, [
    h('div.lat-embed-box', [
      h('header.lat-embed-head', [
        h('span.lat-embed-kicker', 'latent'),
        h('span.lat-embed-state', text),
        href ? h('a.lat-embed-open.internal', { href }, 'open ↗') : null,
      ]),
    ]),
    caption ? h('figcaption.lat-embed-cap', caption) : null,
  ])
}

export interface EmbedRequest {
  sidecar: Sidecar
  /** the latent page's slug (`embeddings/index`) */
  pageSlug: string
  scope: string[]
  fragments: readonly string[]
  settings: EmbedSettings
  caption?: string
  /** build output, for the coordinates asset */
  output?: string
}

export async function renderLatentEmbed(request: EmbedRequest): Promise<EmbedResult> {
  const { sidecar, settings, caption } = request
  const warnings: string[] = []
  const href = pageHref(request.pageSlug)
  const parsed = parseLatentFragment(request.fragments)
  const view = parsed.view ? decodeLatentView(parsed.view) : null
  const plainOpen = (focus: number | undefined) =>
    focus !== undefined ? `${href}?focus=${focus}` : href
  if (parsed.view && !view) {
    warnings.push('the view payload does not decode')
    return {
      figure: notice('this view does not decode', plainOpen(parsed.focus), caption),
      warnings,
    }
  }
  for (const key of parsed.unsupported) warnings.push(`unsupported key: ${key}`)
  if (view && parsed.focus !== undefined && parsed.focus !== view.focus)
    warnings.push(
      `focus=${parsed.focus} differs from the view's focus ${view.focus}; the view wins`,
    )
  const focus = view?.focus ?? parsed.focus
  if (focus === undefined) {
    warnings.push('neither focus nor view names a row')
    return { figure: notice('this embed names no row', href, caption), warnings }
  }
  if (view && view.base !== sidecar.base) {
    // generations rotate and prune (docs/latent-embed.md §10); remapping is phase 3
    warnings.push(
      `the view was made on ${view.base.split('/').pop()?.slice(0, 8)}, now ${sidecar.index.hash.slice(0, 8)}`,
    )
    return {
      figure: notice('this view was made on an older embedding build', plainOpen(focus), caption),
      warnings,
    }
  }
  if (!Number.isSafeInteger(focus) || focus < 0 || focus >= sidecar.index.rows) {
    warnings.push(`focus ${focus} is out of range (${sidecar.index.rows} rows)`)
    return {
      figure: notice('this view’s row is not in the embedding build', href, caption),
      warnings,
    }
  }

  const latent = nodeLatent(sidecar)
  const rt = nodeRuntime(latent, focus, request.scope)
  const names = (sidecar.index.features ?? []).map(d => d.name)
  const mask = view?.filter ? rowMask(latent.docByRow, view.filter, sidecar.index.rows) : undefined
  const st = nodeFeatureState(latent, names, mask)
  const meta = await latent.meta
  const dicts = await Promise.all(names.map(name => loadDict(rt, st, name)))
  const tables = names.length ? await steerTables(rt, st) : null

  // the features state of the view, validated as restoreFeatures validates it
  const saved = view?.features
  const valid = (name: string, f: number) => dicts.some(d => d.meta.name === name && !!d.table[f])
  if (saved) {
    st.explain = saved.explain
    for (const [key, w] of saved.steer) {
      const term = termOf(key, w)
      const ok =
        term.kind === 'feature'
          ? valid(term.dict, term.f)
          : term.kind === 'tag'
            ? tables?.tags.some(tag => tag.t === term.id)
            : meta.axes.some(axis => axis.id === term.id)
      if (ok) st.steer.set(key, term)
      else warnings.push(`steering term ${key} is not in this build`)
    }
    const selected = saved.selected
    if (selected && valid(selected.dict, selected.f)) st.selected = { ...selected }
    else if (selected) warnings.push(`feature ${selected.dict}:${selected.f} is not in this build`)
    if (saved.query)
      warnings.push(
        `typed query "${saved.query}" needs the browser encoder; showing the focus links`,
      )
  }

  // results: a pinned feature's rows, the steered rows, or the focus's own neighbours
  let rows: [number, number][]
  let resultLabel: string
  const live = [...st.steer.values()].filter(t => t.w !== 0)
  if (saved?.pin && st.selected) {
    const dict = dicts.find(d => d.meta.name === st.selected!.dict)!
    const row = dict.table[st.selected.f]
    rows = [...row.top, ...row.cross.slice(0, 4)]
    resultLabel = `feature · ${featureName(row)}`
    if (mask) rows = rows.filter(([r]) => mask[r])
  } else if (live.length) {
    const steered = await steeredRows(rt, st)
    rows = steered?.rows ?? []
    resultLabel = `focus, steered by ${live.length} term${live.length === 1 ? '' : 's'}`
    st.pinned = rows.map(([r]) => r)
  } else {
    rows = sidecar.knn(focus).neighbours.map(n => [n.r, n.score] as [number, number])
    if (mask) rows = rows.filter(([r]) => mask[r])
    resultLabel = 'focus links'
  }
  if (saved?.pin || live.length) rt.query = { text: resultLabel, items: linkItems(rt, rows) }

  // the features column, built by the client's own panel code
  let column: ElementContent[] = []
  if (names.length && settings.panes !== 'bench') {
    const built = await panelGroups(rt, st, 0, true)
    column = built?.groups ?? []
  }

  // the bench
  let benchData: Record<string, string> = {}
  const bench = benchOf(view?.bench ?? null, meta, names.length > 0)
  let benchEl: Element | null = null
  if (settings.panes !== 'features') {
    const drawn = await drawBench(rt, st, latent.engine, bench, meta, dicts, sidecar, focus)
    benchEl = h('div.lat-embed-bench', [drawn.poster, h('p.lat-bench-status', drawn.status)])
    if (request.output && drawn.asset) {
      const file = path.join(request.output, 'static', 'latent-embed', `${drawn.asset.name}.bin`)
      if (!existsSync(file)) {
        mkdirSync(path.dirname(file), { recursive: true })
        writeFileSync(file, drawn.asset.bytes)
      }
    }
    benchData = drawn.data
  }

  const doc = docOfRow(sidecar, focus)
  const open = view ? `${href}?focus=${focus}#view=${parsed.view}` : `${href}?focus=${focus}`
  const state: string[] = []
  if (live.length) state.push(`steered · ${live.length}`)
  else if (saved?.pin && st.selected) state.push('feature')
  if (view?.filter) state.push(`filter ${view.filter.join(',')}`)
  if (saved?.query) state.push(`query “${saved.query.slice(0, 24)}”`)
  const head = h('header.lat-embed-head', [
    h('span.lat-embed-kicker', 'latent'),
    h('a.lat-embed-focus.internal', { href: rowHref(doc) }, titleText(doc.t)),
    h('span.lat-embed-locus', `${doc.s} · ${focus - doc.r0 + 1}/${doc.n}`),
    h('span.lat-embed-state', state.join(' · ')),
    h('button.lat-embed-reset', { type: 'button', style: 'display:none' }, 'reset'),
    h('a.lat-embed-open', { href: open, dataNoPopover: true }, 'open ↗'),
  ])

  // the band: the view's results, a fixed number of pooled rows
  const shown = rows.slice(0, settings.links)
  const band =
    settings.links > 0
      ? h('ol.lat-embed-links', [
          h('li.lat-embed-links-head', [
            h('span', resultLabel),
            h('span', `top ${shown.length} of ${rows.length}`),
          ]),
          ...shown.map(([r, score]) => resultRow(sidecar, latent.docByRow, r, score)),
        ])
      : null

  const features = column.length ? h('div.lat-embed-features.lat-features', column) : null
  if (features) rewriteFocusLinks(features, href)

  const figure = h(
    'figure.latent-embed',
    {
      dataLatentEmbed: '',
      dataWide: settings.wide ? '' : undefined,
      dataPanes: settings.panes,
      dataLatentBase: sidecar.base,
      dataLatentPage: href,
      dataLatentFocus: String(focus),
      dataLatentView: parsed.view,
      dataLatentK: String(sidecar.index.k),
      dataLatentBlock: String(sidecar.index.block),
      dataLatentAxes: String(sidecar.index.axes),
      dataLatentKinds: sidecar.index.kinds.join(','),
      dataLatentScope: request.scope.join(',') || undefined,
      dataLatentRows: String(sidecar.index.rows),
      dataLatentDocs: String(sidecar.index.docs),
      dataLatentFeatures: names.join(',') || undefined,
      dataLatentLinks: String(settings.links),
      ...benchData,
    },
    [
      h('div.lat-embed-box.lat-probe', [head, benchEl, features, band]),
      caption ? h('figcaption.lat-embed-cap', caption) : null,
    ],
  )
  return { figure, warnings }
}

/** One result row: swatch, title linking where the source lives, locus and score. */
export function resultRow(
  sidecar: Sidecar,
  docByRow: Runtime['docByRow'],
  r: number,
  score: number,
): Element {
  const doc = docByRow[r]
  const preview = sidecar.preview(r)
  const page = doc.s === 'pdf' ? preview.p : null
  return h('li', { dataR: String(r) }, [
    h('span.lat-swatch', { dataKind: doc.s, title: doc.s }),
    h('a.internal', { href: rowHref(doc, page) }, titleText(doc.t)),
    h('span.lat-embed-loc', `${r - doc.r0 + 1}/${doc.n} · ${(score / 255).toFixed(2)}`),
  ])
}

/** The explorer's `?focus=N` links open /embeddings at that row from inside a note. */
export function rewriteFocusLinks(tree: Element, href: string): void {
  visit(tree, 'element', node => {
    const target = node.properties?.href
    if (node.tagName === 'a' && typeof target === 'string' && target.startsWith('?focus='))
      node.properties.href = `${href}${target}`
  })
}

type Meta = NonNullable<Runtime['bench']>['meta'] & {}

/** The view's bench, or the default one: features level when there are dictionaries. */
export function benchOf(saved: SharedBench | null, meta: Meta, hasFeatures: boolean): SharedBench {
  const pcs = [1, 2, 3].map(n => meta.axes.findIndex(axis => axisBase(axis.name) === `pc${n}`))
  const axes = pcs.every(i => i >= 0) ? pcs : [0, 1, 2]
  if (saved)
    return {
      ...saved,
      axes: saved.axes.every(i => i < meta.axes.length) ? saved.axes : axes,
      level: saved.level === 'feats' && !hasFeatures ? 'docs' : saved.level,
    }
  return {
    yaw: 0.6,
    pitch: 0.5,
    zoom: 1,
    panX: 0,
    panY: 0,
    axes,
    level: hasFeatures ? 'feats' : 'docs',
    hidden: 0,
    featsOn: hasFeatures,
    picked: false,
    feature: null,
  }
}

async function drawBench(
  rt: Runtime,
  st: FeatureState,
  engine: ReturnType<typeof nodeLatent>['engine'],
  bench: SharedBench,
  meta: Meta,
  dicts: FeatureDict[],
  sidecar: Sidecar,
  focus: number,
): Promise<{
  poster: Element
  status: string
  data: Record<string, string>
  asset: { name: string; bytes: Uint8Array } | null
}> {
  const dirs = new Float32Array(3 * 256)
  bench.axes.forEach((axis, slot) => dirs.set(meta.axes[axis].dir, slot * 256))
  const axisNames = bench.axes.map(a => axisBase(meta.axes[a].name)) as [string, string, string]
  let coords: Float32Array
  let extent: number
  let kinds: Uint8Array
  let lvl: Uint8Array | undefined
  let mark = -1
  let edges: { near: number[]; parent: number; children: number[] } | undefined
  let heat: Uint8Array | null = null
  let atlas: Atlas | null = null
  const level = bench.level
  // the feature the bench marks: its own pick, or the panel's selected feature at the atlas
  const feature = bench.feature ?? (level === 'feats' ? st.selected : null)
  if (level === 'feats') {
    const usable = dicts.filter(dict => dict.meta.dims === meta.mean.length)
    const decs = await Promise.all(usable.map(dict => weights(rt, dict, 'dec')))
    atlas = atlasFrom(usable, decs, meta.mean)
    ;({ coords, extent } = projectAtlas(atlas, dirs))
    kinds = atlas.kind
    lvl = atlas.lvl
    if (feature) {
      mark = atlas.at.get(`${feature.dict}:${feature.f}`) ?? -1
      const dict = usable.find(d => d.meta.name === feature.dict)
      const row = dict?.table[feature.f]
      if (dict && row && mark >= 0) {
        const index = (g: number) => atlas!.at.get(`${dict.meta.name}:${g}`) ?? -1
        edges = {
          near: row.near.map(([g]) => index(g)),
          parent: row.parent !== undefined && row.parent >= 0 ? index(row.parent) : -1,
          children: (dict.children.get(row.f) ?? []).slice(0, 8).map(index),
        }
      }
    }
  } else {
    const rows = level === 'docs' ? meta.docs : meta.rows
    const url = `${sidecar.base}/vec/${level}-256-int8.bin`
    await engine.load(url, rows, 256)
    ;({ coords, extent } = engine.project(url, dirs, new Float32Array(meta.mean), meta.scale))
    extent ||= 1
    kinds = new Uint8Array(rows)
    if (level === 'docs') for (const doc of sidecar.docs) kinds[doc.d] = KIND_INDEX[doc.s]
    else for (const doc of sidecar.docs) kinds.fill(KIND_INDEX[doc.s], doc.r0, doc.r0 + doc.n)
    mark = level === 'docs' ? docOfRow(sidecar, focus).d : focus
    if (bench.feature) {
      const dict = dicts.find(d => d.meta.name === bench.feature!.dict)
      if (dict?.table[bench.feature.f])
        heat = await featureHeat(rt, dict, bench.feature.f, level, sidecar)
    }
  }
  const poster = benchPoster({
    view: bench,
    level,
    coords,
    extent,
    kinds,
    lvl,
    hidden: bench.hidden,
    heat,
    mark,
    edges,
    axes: axisNames,
  })
  const n = coords.length / 3
  const unit = level === 'docs' ? 'documents' : level === 'rows' ? 'chunks' : 'features'
  const sampled = posterCount(level, n)
  const hiddenKinds = (Object.keys(KIND_INDEX) as LatentKind[]).filter(
    k => (bench.hidden >> KIND_INDEX[k]) & 1,
  )
  const status = [
    sampled < n
      ? `${sampled.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} ${unit}`
      : `${n.toLocaleString('en-US')} ${unit}`,
    axisNames.join(' '),
    hiddenKinds.length ? `no ${hiddenKinds.map(k => KIND_LABEL[k]).join(', ')}` : null,
    feature && level === 'feats' && mark >= 0
      ? `# ${featureName(dicts.find(d => d.meta.name === feature.dict)!.table[feature.f])}`
      : null,
  ]
    .filter(Boolean)
    .join(' · ')

  // the coordinates the live bench turns (coords.ts); content-addressed per space, axes and heat
  const { bytes, reach } = encodeCoords(coords, kinds, lvl, heat)
  const name = createHash('sha256')
    .update(
      `${sidecar.base}\u0000${level}\u0000${bench.axes.join(',')}\u0000${heat ? `${bench.feature!.dict}:${bench.feature!.f}` : ''}`,
    )
    .digest('hex')
    .slice(0, 8)
  const data: Record<string, string> = {
    'data-latent-coords': `/static/latent-embed/${name}.bin`,
    'data-latent-level': level,
    'data-latent-points': String(n),
    'data-latent-reach': String(reach),
    'data-latent-extent': String(extent),
    'data-latent-mark': String(mark),
    'data-latent-axis-names': JSON.stringify(axisNames),
    'data-latent-bench': JSON.stringify({
      yaw: bench.yaw,
      pitch: bench.pitch,
      zoom: bench.zoom,
      panX: bench.panX,
      panY: bench.panY,
      axes: bench.axes,
      hidden: bench.hidden,
    }),
  }
  if (edges) data['data-latent-edges'] = JSON.stringify(edges)
  if (heat) data['data-latent-heat'] = ''
  if (atlas && feature) data['data-latent-feature'] = `${feature.dict}:${feature.f}`
  return { poster, status, data, asset: { name, bytes } }
}

/** Activation (0..255) per point of one feature: a document takes its strongest row. */
async function featureHeat(
  rt: Runtime,
  dict: FeatureDict,
  f: number,
  level: 'docs' | 'rows',
  sidecar: Sidecar,
): Promise<Uint8Array> {
  const { rows, codeBlock: size, k } = dict.meta
  const docs = level === 'docs'
  const heat = new Uint8Array(docs ? sidecar.docs.length : sidecar.index.rows)
  const docByRow = rt.docByRow
  for (let b = 0; b * size < rows; b++) {
    const { ids, acts } = await codeBlock(rt, dict, b)
    for (let j = 0; j < acts.length; j++) {
      if (ids[j] !== f || !acts[j]) continue
      const r = b * size + Math.floor(j / k)
      const p = docs ? docByRow[r]?.d : r
      if (p === undefined || p >= heat.length) continue
      if (acts[j] > heat[p]) heat[p] = acts[j]
    }
  }
  return heat
}

export type { SharedView }
