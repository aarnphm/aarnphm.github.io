import type { Runtime } from '../runtime'
import type { Aligned } from './text'
import { bridgeOf, bridgesOf, predicts } from '../bridges'
import { benchFeatures } from '../features'
import { labelOf, type Bake, type BakeRow, type Label } from './bake'
import { pool } from './read'

const STRONG = 0.75

// ---- markup -----------------------------------------------------------------------------------

export const esc = (s: string) =>
  s.replace(
    /[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!,
  )

export const ll = (l: number) => `L${String(l).padStart(2, '0')}`

function featureButton(
  l: number,
  f: number,
  a: number,
  label: Label | null,
  gloss?: string,
): string {
  const strong = (label?.score ?? 0) >= STRONG
  const name = label ? label.label : gloss || `f${f}`
  const score =
    label?.score !== undefined
      ? ` · detection ${label.score.toFixed(2)}`
      : gloss
        ? ' · fires on'
        : ''
  const kind = !label && gloss ? ' data-gloss' : ''
  return `<button type="button" class="lat-gm-f"${strong ? ' data-strong' : ''}${kind} data-l="${l}" data-f="${f}" style="--a:${a.toFixed(3)}" data-tip="${esc(`${ll(l)} f${f}${score}`)}"><span>${esc(name)}</span><i></i></button>`
}

export function panelHtml(
  bake: Bake,
  data: BakeRow,
  al: Aligned,
  picked: number[],
  labels: Record<string, Record<string, Label | string>>,
  glosses: Record<string, string>[],
  bridged: string,
): string {
  const { layers, layerMax } = pool(data, picked)
  const s = data.spans[2 * picked[0]]
  const e = data.spans[2 * picked[picked.length - 1] + 1]
  const quote = al.raw.slice(s, e).replace(/\s+/g, ' ').trim()
  const top: { l: number; f: number; x: number; a: number }[] = []
  const strip = layers
    .map((feats, l) => {
      const cells = feats
        .slice(0, 3)
        .map(({ f, a }) => {
          const x = layerMax[l] > 0 ? a / layerMax[l] : 0
          top.push({ l, f, x, a })
          return featureButton(l, f, x, labelOf(labels, l, f), glosses[l]?.[String(f)])
        })
        .join('')
      return `<li><span class="lat-gm-l">${ll(l)}</span><span class="lat-gm-cells">${cells}</span></li>`
    })
    .reverse()
    .join('')
  top.sort((p, q) => q.x - p.x)
  const strongest = top
    .filter(({ l, f }) => (labelOf(labels, l, f)?.score ?? 0) >= STRONG)
    .slice(0, 10)
    .map(({ l, f, x }) => {
      const lab = labelOf(labels, l, f)!
      return `<li class="lat-feat"><span class="lat-feat-bar"><i style="--a:${x.toFixed(3)}"></i></span><button type="button" class="lat-feat-name lat-gm-f" data-l="${l}" data-f="${f}">${esc(lab.label)}</button><span class="lat-feat-meta">${ll(l)} f${f}</span><span class="lat-feat-meta">${(lab.score ?? 0).toFixed(2)}</span></li>`
    })
    .join('')
  const attn = bake.meta.attnLayers
    .map((layer, i) => `<button type="button" class="lat-gm-a" data-a="${i}">${ll(layer)}</button>`)
    .join('')
  const cut = bake.meta.rows[String(al.row)]?.[1]
  // localhost only: plt_serve.py traces the token after the selection on request
  const trace = bake.base.startsWith('http://localhost')
    ? `<p class="lat-gm-note"><button type="button" class="lat-gm-trace" data-tip="${esc('attribution graph of the next token\n32 tokens of context · ~15 s\nshown in the circuit tab')}">trace the next token</button></p>`
    : ''
  return [
    `<section class="lat-feat-group lat-gm-head"><header><span class="lat-feat-head">“${esc(quote.length > 80 ? `${quote.slice(0, 79)}…` : quote)}”</span><span class="lat-count">${picked.length} tok</span></header>`,
    `${cut ? '<p class="lat-gm-note">past the 1024-token window: the tail is not baked</p>' : ''}${trace}</section>`,
    strongest
      ? `<section class="lat-feat-group"><header><span class="lat-feat-head">reads as</span><span class="lat-count">detection ≥ ${STRONG}</span></header><ul class="lat-feat-list">${strongest}</ul></section>`
      : '',
    bridged,
    `<section class="lat-feat-group"><header><span class="lat-feat-head">attends to</span><span class="lat-count">global layers</span></header><nav class="lat-gm-attn">${attn}</nav></section>`,
    `<section class="lat-feat-group"><header><span class="lat-feat-head">depth</span><span class="lat-count">top 3 a layer, L41 first</span></header><ol class="lat-gm-strip">${strip}</ol></section>`,
    `<section class="lat-feat-group lat-gm-where" hidden></section>`,
  ].join('')
}

/** the text dictionary's feature names, through the features tab's loaded dictionary */
export async function textNames(rt: Runtime): Promise<((f: number) => string) | null> {
  const feats = benchFeatures(rt)
  const dict = await feats?.dict('text').catch(() => null)
  if (!feats || !dict) return null
  return f => (dict.table[f] ? feats.name(dict.table[f]) : `#${f}`)
}

export function textChip(f: number, name: string, tip: string): string {
  return `<button type="button" class="lat-feat-chip" data-open-text="${f}" data-tip="${esc(tip)}" data-tip-at="up">${esc(name)}</button>`
}

/**
 * The passage-SAE features the selection's Gemma features predict: each text feature scores the
 * sum, over the selection's features that bridge to it, of bridge weight × activation (as a share
 * of its layer's max in the passage).
 */
export async function bridgeSection(rt: Runtime, data: BakeRow, picked: number[]): Promise<string> {
  const [br, name] = await Promise.all([bridgesOf(rt), textNames(rt)])
  if (!br || !name) return ''
  const { layers, layerMax } = pool(data, picked)
  const score = new Map<number, { s: number; via: number }>()
  layers.forEach((feats, l) => {
    for (const { f, a } of feats) {
      const x = layerMax[l] > 0 ? a / layerMax[l] : 0
      for (const [j, w] of predicts(br, l, f)) {
        const got = score.get(j) ?? { s: 0, via: 0 }
        got.s += w * x
        got.via += 1
        score.set(j, got)
      }
    }
  })
  const top = [...score].sort((p, q) => q[1].s - p[1].s).slice(0, 6)
  if (!top.length) return ''
  const max = top[0][1].s
  const items = top
    .map(([j, { s, via }]) => {
      const b = bridgeOf(br, j)!
      const auc = b.auc === null ? 'auc n/a' : `auc ${b.auc.toFixed(2)}`
      return `<li class="lat-feat"><span class="lat-feat-bar"><i style="--a:${(s / max).toFixed(3)}"></i></span>${textChip(j, name(j), `text #${j} · ${auc}\nvia ${via} gemma feature${via === 1 ? '' : 's'}`)}<span class="lat-feat-meta">#${j}</span><span class="lat-feat-meta">${via}</span></li>`
    })
    .join('')
  return `<section class="lat-feat-group lat-gm-bridges"><header><span class="lat-feat-head">bridges</span><span class="lat-count">text features</span></header><ul class="lat-feat-list">${items}</ul></section>`
}
