import { type LatentKind } from '../../../util/latent/types'

/**
 * Sparse features of the latent space (Prism's move): quartz/scripts/latent/features.py trains a
 * Matryoshka TopK sparse autoencoder per family of rows (text passages, images, base entries) and
 * writes `feat/<name>/`; label.py names the features with a local Gemma. The probe's features
 * tab lists the features the focused chunk fires, a click pins a feature's strongest rows to the
 * links pane and opens its card (ancestry, children, notes over time), ± steers the focus vector
 * along a feature's decoder direction and asks for its neighbours there, and a typed query is
 * encoded into features in the browser. The tag probes (`feat/tags.json`) add the tags a document
 * reads as but does not carry. Nothing here loads until the tab opens or the search box is used.
 *
 * Steering is score fusion (docs/latent-steering.md): a row ranks by its cosine to the base plus
 * Σ weight × channel, where a feature's channel is its detector (relu of the encoder
 * pre-activation over the max, which finds rows that fire the feature far better than the decoder
 * direction) and a tag's or an axis's is the projection on its difference-in-means direction. The
 * worker keeps the channels, so a slider drag is one pass over cached arrays. A term's unit comes
 * from `feat/steer.json`: the weight at which half the steered links carry it.
 */

export interface FeatureDictMeta {
  name: string
  kinds: LatentKind[]
  features: number
  k: number
  dims: number
  rows: number
  trained: number
  fvu: number
  /** Matryoshka prefixes: features [0, prefixes[0]) are level 0, and so on */
  prefixes?: number[]
  levels?: { prefix: number; fvu: number }[]
  /** rows per codes/ block */
  codeBlock: number
  /** dated notes per month (yyyymm) */
  months?: Record<string, number>
  norm: number
  mean: number[]
  bias: number[]
  encScale: number[]
  decScale: number[]
}

export interface FeatureRow {
  f: number
  /** a written name from latent/labels.json, when one exists */
  label?: string
  /** label.py's one-sentence description */
  about?: string
  /** detection score of the label: balanced accuracy on held-out rows, 0.5 = chance */
  score?: number
  /** Matryoshka level (0 = coarsest) and the coarser feature that fires with it most */
  level?: number
  parent?: number
  share?: number
  terms: string[]
  n: number
  /** firing rows by kind: note, pdf, image, base */
  kinds: [number, number, number, number]
  docs: number
  max: number
  near: [number, number][]
  /** [row, activation / max × 255] */
  top: [number, number][]
  cross: [number, number][]
  /** dated notes that fire it, by month: [yyyymm, notes] */
  time?: [number, number][]
}

export interface CodeBlock {
  ids: Uint16Array
  acts: Uint8Array
}

export interface FeatureDict {
  meta: FeatureDictMeta
  table: FeatureRow[]
  children: Map<number, number[]>
  blocks: Map<number, Promise<CodeBlock>>
  enc: Int8Array | null
  dec: Int8Array | null
}

export interface TagProbes {
  tags: { t: string; n: number; auc: number }[]
  /** [doc, [[tag index, p × 100]]] */
  pred: [number, [number, number][]][]
}

/** feat/steer.json (quartz/scripts/latent/steering.py) */
export interface SteerTables {
  topn: number
  grid: number
  /** per dictionary, per feature: [unit, share of links that fire it at one unit, weak] */
  dicts: Record<string, [number, number, number][]>
  tags: {
    t: string
    dir: number[]
    unit: number
    p: number
    weak: number
    /** the 5-feature sparse probe: [text feature, coefficient] */
    explain: [number, number][]
    auc5: number
    auc: number
  }[]
  axes: { id: string; name: string; unit: number }[]
}

/** feat/text/words.json (quartz/scripts/latent/words.py): words that make a feature fire */
export interface WordTable {
  features: Record<string, { w: [string, number][]; r: [number, [string, number][]][] }>
}

export type SteerTerm =
  | { kind: 'feature'; dict: string; f: number; w: number }
  | { kind: 'tag'; id: string; w: number }
  | { kind: 'axis'; id: string; w: number }

export interface FeatureState {
  refresh(): void
  pin(label: string, rows: [number, number][]): Promise<void>
  names: string[]
  dicts: Map<string, Promise<FeatureDict>>
  tags: Promise<Map<number, [string, number][]> | null> | null
  selected: { dict: string; f: number } | null
  query: { text: string; vector: Float32Array } | null
  needle: string
  /** the steering set: term key (`text:12`, `tag:philosophy`, `axis:fm1`) → signed units */
  steer: Map<string, SteerTerm>
  steerSeq: number
  tables: Promise<SteerTables | null> | null
  tablesNow: SteerTables | null
  words: Promise<WordTable | null> | null
  /** channel keys the worker holds for the rows matrix */
  channels: Set<string>
  /** the "reasoning" group is open (it is by default) */
  explain: boolean
  /** the decoders "reasoning" needs have loaded once, so it renders with the panel */
  whyReady: boolean
  /** bumps per panel render, so a late reasoning fill lands only in the panel it was made for */
  renderSeq: number
  /** reasoning per link row, for the current focus or query (whyKey) */
  whyCache: Map<number, WhyLink>
  whyKey: string
  /** the rows steering last pinned to the links pane, which "reasoning" explains */
  pinned: number[] | null
  dragFrame: number | null
  panel: HTMLElement
}

export type TermHit = { key: string; name: string; kind: 'tag' | 'axis'; score: number }

// ---- reasoning ------------------------------------------------------------------------------

/**
 * A link's cosine as an attribution graph. Each row is mean + norm · Σ a_f d_f + error in its
 * family's dictionary, so ⟨q, x⟩ splits into the dictionaries' mean terms (the space's shared
 * baseline), one edge per pair of active features (norm_q · norm_x · a_f · a_g · ⟨d_f, d_g⟩), and
 * error. Over the corpus the edges carry 86 % of what is above the baseline and the top three
 * edges 63 % of the edge mass, so three edges and the error share say why a link is there.
 */
export interface WhyLink {
  r: number
  cos: number
  share: number
  edges: { q: [FeatureDict, number]; x: [FeatureDict, number]; v: number }[]
}
