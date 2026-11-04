/** Versioned URL state. Parsing is bounded before any shared value reaches a data loader. */
export interface SharedFeature {
  dict: string
  f: number
}

export interface SharedFeatures {
  selected: SharedFeature | null
  steer: [string, number][]
  explain: boolean
  query: string
  pin: boolean
  cluster: boolean
}

export interface SharedBench {
  yaw: number
  pitch: number
  zoom: number
  /** Pan as a fraction of the viewport, so a link also works at a different screen size. */
  panX: number
  panY: number
  axes: number[]
  level: 'docs' | 'rows' | 'feats'
  hidden: number
  featsOn: boolean
  picked: boolean
  feature: SharedFeature | null
}

export interface SharedView {
  v: 1
  base: string
  focus: number
  tab: 'axes' | 'bench' | 'features'
  features: SharedFeatures | null
  bench: SharedBench | null
}

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
const integer = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const dictName = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-z][a-z0-9-]{0,63}$/.test(value)
const feature = (value: unknown): value is SharedFeature | null =>
  value === null || (record(value) && dictName(value.dict) && integer(value.f))

function features(value: unknown): value is SharedFeatures | null {
  if (value === null) return true
  return (
    record(value) &&
    feature(value.selected) &&
    typeof value.explain === 'boolean' &&
    typeof value.pin === 'boolean' &&
    typeof value.cluster === 'boolean' &&
    typeof value.query === 'string' &&
    value.query.length <= 4096 &&
    Array.isArray(value.steer) &&
    value.steer.length <= 128 &&
    value.steer.every(
      (term: unknown) =>
        Array.isArray(term) &&
        term.length === 2 &&
        typeof term[0] === 'string' &&
        term[0].length <= 256 &&
        /^(?:tag:.+|axis:.+|[a-z][a-z0-9-]*:\d+)$/.test(term[0]) &&
        finite(term[1]) &&
        Math.abs(term[1]) <= 12,
    )
  )
}

function bench(value: unknown): value is SharedBench | null {
  if (value === null) return true
  return (
    record(value) &&
    finite(value.yaw) &&
    finite(value.pitch) &&
    finite(value.zoom) &&
    value.zoom > 0 &&
    value.zoom <= 100 &&
    finite(value.panX) &&
    finite(value.panY) &&
    Array.isArray(value.axes) &&
    value.axes.length === 3 &&
    value.axes.every(integer) &&
    (value.level === 'docs' || value.level === 'rows' || value.level === 'feats') &&
    integer(value.hidden) &&
    value.hidden <= 15 &&
    typeof value.featsOn === 'boolean' &&
    typeof value.picked === 'boolean' &&
    feature(value.feature)
  )
}

export function readSharedView(hash: string): SharedView | null {
  if (!hash.startsWith('#view=') || hash.length > 65536) return null
  try {
    const value: unknown = JSON.parse(decodeURIComponent(hash.slice(6)))
    if (
      !record(value) ||
      value.v !== 1 ||
      typeof value.base !== 'string' ||
      value.base.length > 512 ||
      !integer(value.focus) ||
      !['axes', 'bench', 'features'].includes(String(value.tab)) ||
      !features(value.features) ||
      !bench(value.bench)
    )
      return null
    const tab = value.tab
    if (tab !== 'axes' && tab !== 'bench' && tab !== 'features') return null
    return {
      v: 1,
      base: value.base,
      focus: value.focus,
      tab,
      features: value.features,
      bench: value.bench,
    }
  } catch {
    return null
  }
}

export function sharedViewUrl(href: string, view: SharedView): string {
  const url = new URL(href)
  url.searchParams.set('focus', String(view.focus))
  url.hash = 'view=' + encodeURIComponent(JSON.stringify(view))
  return url.href
}
