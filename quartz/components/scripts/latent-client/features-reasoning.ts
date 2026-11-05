import { type Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import {
  chunkLink,
  groupDocumentLinks,
  type DocumentLink,
} from '../../../util/latent/document-links'
import { mark } from '../../../util/latent/markup'
import { titleRuns } from '../../../util/latent/title'
import { type LatentKind } from '../../../util/latent/types'
import { linkItems } from './document-list'
import { weights, encodeVector, codesOf, loadDict } from './features-data'
import { chip } from './features-markup'
import { type FeatureDict, type FeatureState, type WhyLink } from './features-types'
import { type Runtime, knnOf, metaOf, requestVec, docOfRow } from './runtime'

function dictFor(dicts: FeatureDict[], kind: LatentKind | undefined): FeatureDict | undefined {
  return (
    dicts.find(d => kind && d.meta.kinds.includes(kind)) ?? dicts.find(d => d.meta.name === 'text')
  )
}

async function reconstruct(
  rt: Runtime,
  dict: FeatureDict,
  codes: [number, number][],
): Promise<{ codes: [number, number][]; dirs: Float32Array[] }> {
  const dec = await weights(rt, dict, 'dec')
  const { dims: D, decScale } = dict.meta
  return {
    codes: codes.map(([f, a]) => [f, a * dict.table[f].max]),
    dirs: codes.map(([f]) => {
      const v = new Float32Array(D)
      for (let d = 0; d < D; d++) v[d] = dec[f * D + d] * decScale[f]
      return v
    }),
  }
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0
  for (let d = 0; d < a.length; d++) s += a[d] * b[d]
  return s
}

/** The first five matching documents, with one explanation for each strongest chunk. */
async function explainLinks(
  rt: Runtime,
  st: FeatureState,
  dicts: FeatureDict[],
): Promise<{ group: DocumentLink; link: WhyLink }[]> {
  const focus = rt.focus
  const query = st.query
  const source = rt.query
  const steered = st.pinned && [...st.steer.values()].some(t => t.w !== 0)
  const items = steered
    ? linkItems(
        rt,
        st.pinned!.map(r => [r, 0]),
      )
    : rt.query
      ? rt.query.items
      : linkItems(
          rt,
          (await knnOf(rt, rt.focus)).neighbours.map(n => [n.r, n.score]),
        )
  const groups = groupDocumentLinks(items).slice(0, 5)
  if (rt.signal.aborted || rt.focus !== focus || st.query !== query || rt.query !== source)
    return []
  const links = await explainRows(
    rt,
    st,
    dicts,
    groups.map(group => group.best.preview.r),
  )
  return links.map((link, i) => ({ group: groups[i], link }))
}

/** Reasoning for the given link rows against the focus (or the typed query), cached per row. */
async function explainRows(
  rt: Runtime,
  st: FeatureState,
  dicts: FeatureDict[],
  links: number[],
): Promise<WhyLink[]> {
  const focus = rt.focus
  const query = st.query
  const stale = () => rt.signal.aborted || rt.focus !== focus || st.query !== query
  if (stale()) return []
  const key = query ? `q:${query.text}` : `f:${focus}`
  if (st.whyKey !== key) {
    st.whyKey = key
    st.whyCache.clear()
  }
  const cache = st.whyCache
  if (links.every(r => cache.has(r))) return links.map(r => cache.get(r)!)
  const meta = await metaOf(rt)
  if (stale()) return []
  const url = `${rt.base}/vec/rows-256-int8.bin`
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  if (stale()) return []
  const rowVec = async (r: number) => {
    const got = await requestVec(rt, { type: 'row', url, row: r, scale: meta.scale })
    return got.type === 'row' ? got.vector : new Float32Array(256)
  }
  const qDict = query
    ? dicts.find(d => d.meta.name === 'text')!
    : dictFor(dicts, docOfRow(rt, focus)?.s)!
  const q = query ? query.vector : await rowVec(focus)
  if (stale()) return []
  const qCodes = query ? await encodeVector(rt, qDict, q) : await codesOf(rt, qDict, focus)
  if (stale()) return []
  const qr = await reconstruct(rt, qDict, qCodes)
  if (stale()) return []
  const out: WhyLink[] = []
  for (const r of links) {
    const hit = cache.get(r)
    if (hit) {
      out.push(hit)
      continue
    }
    const xDict = dictFor(dicts, docOfRow(rt, r)?.s)!
    const [x, xCodes] = await Promise.all([rowVec(r), codesOf(rt, xDict, r)])
    if (stale()) return []
    const xr = await reconstruct(rt, xDict, xCodes)
    if (stale()) return []
    const cos = dot(q, x)
    const baseline =
      dot(qDict.meta.mean, x) + dot(q, xDict.meta.mean) - dot(qDict.meta.mean, xDict.meta.mean)
    const edges: WhyLink['edges'] = []
    let total = 0
    qr.codes.forEach(([f, a], i) =>
      xr.codes.forEach(([g, b], j) => {
        const v = qDict.meta.norm * xDict.meta.norm * a * b * dot(qr.dirs[i], xr.dirs[j])
        total += v
        edges.push({ q: [qDict, f], x: [xDict, g], v })
      }),
    )
    edges.sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    const above = cos - baseline
    const link: WhyLink = {
      r,
      cos,
      share: above > 1e-6 ? Math.max(0, Math.min(1, total / above)) : 0,
      edges: edges.slice(0, 3),
    }
    if (st.whyKey === key) cache.set(r, link)
    out.push(link)
  }
  return out
}

/** A link's top three feature-pair edges, each a pair of chips (one filled chip when both rows
 * fire the same feature) and the edge's value. */
function edgeList(link: WhyLink): Element {
  return h(
    'ol.lat-why-edges',
    link.edges.map(({ q: [qd, f], x: [xd, g], v }) =>
      h('li', { dataBoth: qd === xd && f === g ? '' : undefined }, [
        h('span.lat-why-pair', [
          chip(qd, f, qd === xd && f === g ? 'both rows fire it' : 'fires in the focus'),
          qd === xd && f === g ? null : h('span.lat-why-to', '⟷'),
          qd === xd && f === g ? null : chip(xd, g, 'fires in the link'),
        ]),
        h('span.lat-why-v', `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`),
      ]),
    ),
  )
}

const shareTip = (link: WhyLink) =>
  `cosine ${link.cos.toFixed(3)}; feature pairs carry ${Math.round(link.share * 100)} % of what is above the shared baseline, the rest is the dictionaries' error`

/** Explain only the exact chunk whose preview is open. */
export async function annotateChunk(
  rt: Runtime,
  st: FeatureState,
  row: number,
  slot: HTMLElement,
): Promise<void> {
  const focus = rt.focus
  const query = st.query
  const stale = () =>
    rt.signal.aborted ||
    !st.explain ||
    !slot.isConnected ||
    Number(slot.dataset.row) !== row ||
    rt.focus !== focus ||
    st.query !== query
  if (stale()) return
  const dicts = await Promise.all(st.names.map(name => loadDict(rt, st, name)))
  if (stale()) return
  const [link] = await explainRows(rt, st, dicts, [row])
  if (!link || stale()) return
  slot.title = shareTip(link)
  slot.innerHTML = toHtml(edgeList(link) as never)
}

const RELEVANCE_NOTE =
  'the search results in the links pane, best first: each row is its cosine to the focus or query, split into the feature pairs that carry it'

export async function whySection(
  rt: Runtime,
  st: FeatureState,
  dicts: FeatureDict[],
  now = false,
): Promise<Element> {
  const focus = rt.focus
  const query = st.query
  const head = h('header', [
    h(
      'span.lat-feat-head',
      // the hover note (latent-tips.scss / tips.ts) says what the rows are
      { tabIndex: 0, dataTip: RELEVANCE_NOTE, 'aria-description': RELEVANCE_NOTE },
      'relevance',
    ),
    h(
      'button.lat-feat-clear',
      { type: 'button', dataWhy: '', ariaExpanded: st.explain ? 'true' : 'false' },
      st.explain ? 'hide' : 'show',
    ),
  ])
  if (!st.explain) return h('section.lat-feat-group.lat-why', [head])
  // the first time, the decoders (512 KB for text) load after the panel paints: a quiet line
  // holds the place and fillWhy swaps the section in on idle
  if (!now && !st.whyReady)
    return h('section.lat-feat-group.lat-why', { dataWhyPending: '' }, [
      head,
      h('p.lat-why-wait', 'reading the links…'),
    ])
  const links = await explainLinks(rt, st, dicts)
  if (rt.signal.aborted || rt.focus !== focus || st.query !== query)
    return h('section.lat-feat-group.lat-why', [head])
  return h('section.lat-feat-group.lat-why', [
    head,
    h(
      'ol.lat-why-links',
      links.map(({ group, link }) =>
        h('li', { dataDoc: String(group.doc.d) }, [
          h('p.lat-why-head', [
            mark(group.doc.s),
            h(
              'a',
              {
                href: `?focus=${link.r}`,
                dataLatFocus: String(link.r),
                dataLatChunk: String(link.r),
                dataNoPopover: true,
              },
              titleRuns(group.doc.t),
            ),
            h(
              'span.lat-why-chunks',
              { ariaLabel: 'matching chunks' },
              group.matches.map(item => chunkLink(item.preview, group.doc)),
            ),
            h(
              'span.lat-why-share',
              { title: shareTip(link) },
              `${link.cos.toFixed(3)} · ${Math.round(link.share * 100)} %`,
            ),
          ]),
          edgeList(link),
        ]),
      ),
    ),
  ])
}
