// Mooncake's layer-wise prefill (Qin et al., 2024, §5.2) for LLaMA3-70B on one 8×A800 prefill node.
// Before layer ℓ's attention the node waits for ℓ's prefix KV to arrive from the pool and starts
// loading ℓ+1; after attention, Messenger streams ℓ's KV to the decode node's CPU DRAM. Loads come
// in and streams go out, so on full-duplex links they do not compete. KV sizes are exact; compute
// time is an estimate at the stated MFU.

export const LAYERS = 80
export const KV_HEADS = 8
export const HEAD_DIM = 128
export const D_MODEL = 8192
// 70.6B parameters less the input and output embeddings (2 × 128256 × 8192), spread over 80 layers.
export const PARAMS_PER_LAYER = 0.856e9
export const KV_BYTES_PER_TOKEN_LAYER = 2 * KV_HEADS * HEAD_DIM * 2
export const GPUS = 8
export const GPU_BF16_FLOPS = 312e12
export const MFU = 0.5
export const NODE_FLOPS = GPUS * GPU_BF16_FLOPS * MFU
// Transfer Engine splits any request over 64 KiB into slices that spread across the NICs.
export const SLICE_BYTES = 64 * 1024

export type StreamLink = { id: string; label: string; gbps: number; nics: number; note: string }

// Mooncake README: 40 GB moved at 87 GB/s on 4×200 Gbps and 190 GB/s on 8×400 Gbps RoCE, 2.4× and
// 4.6× TCP. The TCP rate is derived from the second ratio.
export const STREAM_LINKS: readonly StreamLink[] = [
  { id: 'tcp', label: 'TCP', gbps: 190 / 4.6, nics: 8, note: 'derived' },
  { id: 'roce4', label: '4×200G RoCE', gbps: 87, nics: 4, note: 'measured' },
  { id: 'roce8', label: '8×400G RoCE', gbps: 190, nics: 8, note: 'measured' },
]

export const STREAM_CONTEXTS = [8192, 32768, 131072] as const
export const STREAM_HITS = [0, 0.5, 0.9, 0.99] as const

export type Span = { start: number; end: number }
export type StreamPlan = {
  tokens: number
  prefix: number
  fresh: number
  link: StreamLink
  loadBytes: number
  streamBytes: number
  totalBytes: number
  tLoad: number
  tCompute: number
  tStream: number
  slices: number
  load: Span[]
  compute: Span[]
  stream: Span[]
  layered: number
  serial: number
  exposed: number
  bound: 'compute' | 'network'
}

export const streamLink = (id: string | undefined) =>
  STREAM_LINKS.find(link => link.id === id) ?? STREAM_LINKS[2]

export const streamContext = (value: string | undefined) => {
  const n = Number(value)
  return STREAM_CONTEXTS.find(c => c === n) ?? STREAM_CONTEXTS[1]
}

export const streamHit = (value: string | undefined) => {
  const h = Number(value)
  return STREAM_HITS.find(c => c === h) ?? STREAM_HITS[1]
}

// Times in ms.
export function streamPlan(tokens: number, hit: number, link: StreamLink): StreamPlan {
  const prefix = Math.round(tokens * hit)
  const fresh = tokens - prefix
  const loadBytes = prefix * KV_BYTES_PER_TOKEN_LAYER
  // Decode needs the whole context, so every layer ships all of its KV, cached prefix included.
  const streamBytes = tokens * KV_BYTES_PER_TOKEN_LAYER
  const bytesPerMs = link.gbps * 1e6
  // Projections and MLP: 2 FLOPs per parameter per token. Attention: QKᵀ and AV cost 4·d per
  // query-key pair, and each new token sees the prefix plus the causal part of its own chunk.
  const pairs = fresh * prefix + (fresh * (fresh + 1)) / 2
  const flops = 2 * PARAMS_PER_LAYER * fresh + 4 * D_MODEL * pairs
  const tLoad = loadBytes / bytesPerMs
  const tCompute = (flops / NODE_FLOPS) * 1e3
  const tStream = streamBytes / bytesPerMs

  const load: Span[] = []
  const compute: Span[] = []
  const stream: Span[] = []
  let loadFree = 0
  let computeFree = 0
  let streamFree = 0
  for (let l = 0; l < LAYERS; l++) {
    const ls = loadFree
    const le = ls + tLoad
    load.push({ start: ls, end: le })
    const cs = Math.max(computeFree, le)
    const ce = cs + tCompute
    compute.push({ start: cs, end: ce })
    // Layer ℓ+1's load is triggered once layer ℓ's has landed, which is when ℓ's attention starts.
    loadFree = Math.max(le, cs)
    computeFree = ce
    const ss = Math.max(ce, streamFree)
    stream.push({ start: ss, end: ss + tStream })
    streamFree = ss + tStream
  }
  const layered = streamFree
  const serial = LAYERS * (tLoad + tCompute + tStream)
  return {
    tokens,
    prefix,
    fresh,
    link,
    loadBytes,
    streamBytes,
    totalBytes: streamBytes * LAYERS,
    tLoad,
    tCompute,
    tStream,
    slices: Math.ceil(streamBytes / SLICE_BYTES),
    load,
    compute,
    stream,
    layered,
    serial,
    exposed: layered - compute[LAYERS - 1].end,
    bound: Math.max(tLoad, tStream) > tCompute ? 'network' : 'compute',
  }
}

export const fmtTime = (ms: number) =>
  ms >= 1000
    ? `${(ms / 1000).toFixed(2)} s`
    : ms >= 10
      ? `${ms.toFixed(0)} ms`
      : `${ms.toFixed(1)} ms`

export const fmtBytes = (b: number) =>
  b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GiB` : `${(b / 2 ** 20).toFixed(0)} MiB`

export const fmtTokens = (n: number) => `${n / 1024}k`

// Shared by the server render and the inline script so both draw the same bars. `wide` is the
// default drawing; `tall` keeps lanes legible in frames under 600px.
export const LANE_X = 12
export const LANE_H = 20
export const STREAM_VIEWS = [
  { id: 'wide', viewW: 480, laneW: 456 },
  { id: 'tall', viewW: 300, laneW: 276 },
] as const
const MIN_BAR = 1

export type Bar = { x: number; w: number }
// The layer-wise finish marks each overlapped lane and ticks the axis, leaving label rows clear.
export const markerPath = (x: number, laneYs: readonly number[], laneH: number, axisY: number) =>
  [...laneYs.map(y => `M${x.toFixed(2)},${y}v${laneH}`), `M${x.toFixed(2)},${axisY - 6}v12`].join(
    '',
  )

export function streamBars(plan: StreamPlan, laneW: number) {
  const scale = laneW / plan.serial
  const bar = (s: Span): Bar => ({
    x: LANE_X + s.start * scale,
    w: Math.max(MIN_BAR, (s.end - s.start) * scale),
  })
  const totals = [LAYERS * plan.tLoad, LAYERS * plan.tCompute, LAYERS * plan.tStream]
  let at = 0
  const serial = totals.map(t => {
    const span = { start: at, end: at + t }
    at += t
    return bar(span)
  })
  return {
    load: plan.load.map(bar),
    compute: plan.compute.map(bar),
    stream: plan.stream.map(bar),
    serial,
    marker: LANE_X + plan.layered * scale,
  }
}

export function streamReadout(plan: StreamPlan): Record<string, string> {
  return {
    'kv-layer': fmtBytes(plan.streamBytes),
    'kv-total': fmtBytes(plan.totalBytes),
    'prefix-layer': plan.prefix ? fmtBytes(plan.loadBytes) : '–',
    't-compute': fmtTime(plan.tCompute),
    't-load': plan.prefix ? fmtTime(plan.tLoad) : '–',
    't-stream': fmtTime(plan.tStream),
    slices: `${plan.slices} · ${Math.ceil(plan.slices / plan.link.nics)}/NIC`,
    layered: fmtTime(plan.layered),
    serial: fmtTime(plan.serial),
    exposed: fmtTime(plan.exposed),
    bound: plan.bound,
    marker: `layer-wise ${fmtTime(plan.layered)}`,
    'axis-end': fmtTime(plan.serial),
  }
}
