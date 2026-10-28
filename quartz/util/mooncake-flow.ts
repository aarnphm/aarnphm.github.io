// Mooncake's KVCache-centric path (Qin et al., 2024) for one request: the conductor routes the
// prompt to the prefill instance with the lowest estimated TTFT, prefill reuses the cached prefix
// and computes the rest while its messenger streams finished layers over RDMA into the decode
// node's CPU DRAM, and decode appends new blocks. A block is 512 tokens as in the paper; the
// instance counts are illustrative. Every state the figure draws comes from these functions.

export const PROMPT_BLOCKS = 8
export const QUEUE_SLOTS = 8
export const POOL_SLOTS = 16
export const DECODE_VRAM_SLOTS = 24
export const GENERATED_BLOCKS = 2
export const BLOCK_PREFILL_MS = 40

export type MooncakeStage = 'route' | 'prefill' | 'transfer' | 'decode'
export const MOONCAKE_STAGES: readonly MooncakeStage[] = ['route', 'prefill', 'transfer', 'decode']

export type CellState = 'empty' | 'other' | 'pending' | 'hit' | 'new' | 'kept'

type PrefillInstance = { id: string; label: string; cached: readonly string[]; queued: number }
type DecodeInstance = { id: string; label: string; resident: number }
export type MooncakeRequest = { id: string; label: string; blocks: readonly string[] }

const run = (prefix: string, n: number) => Array.from({ length: n }, (_, i) => `${prefix}${i + 1}`)

export const PREFILL_INSTANCES: readonly PrefillInstance[] = [
  { id: 'p0', label: 'prefill P0', cached: [...run('a', 6), ...run('x', 2)], queued: 6 },
  { id: 'p1', label: 'prefill P1', cached: [...run('s', 5), ...run('y', 2)], queued: 2 },
]

export const DECODE_INSTANCES: readonly DecodeInstance[] = [
  { id: 'd0', label: 'decode D0', resident: 14 },
  { id: 'd1', label: 'decode D1', resident: 9 },
]

export const MOONCAKE_REQUESTS: readonly MooncakeRequest[] = [
  { id: 'agent', label: 'agent step', blocks: [...run('s', 5), ...run('n', 3)] },
  { id: 'chat', label: 'chat follow-up', blocks: [...run('a', 6), ...run('n', 2)] },
  { id: 'doc', label: 'new document', blocks: run('n', 8) },
]

export type PrefillRoute = {
  id: string
  label: string
  hit: number
  miss: number
  queued: number
  ttftMs: number
}
export type DecodeRoute = {
  id: string
  label: string
  resident: number
  after: number
  fits: boolean
}
export type MooncakePlan = {
  request: MooncakeRequest
  prefill: PrefillRoute[]
  decode: DecodeRoute[]
  prefillId: string
  decodeId: string
}

const prefixHit = (blocks: readonly string[], cached: readonly string[]) => {
  const set = new Set(cached)
  let n = 0
  while (n < blocks.length && set.has(blocks[n])) n++
  return n
}

export const mooncakeRequest = (id: string | undefined): MooncakeRequest =>
  MOONCAKE_REQUESTS.find(request => request.id === id) ?? MOONCAKE_REQUESTS[0]

export function mooncakePlan(requestId: string | undefined): MooncakePlan {
  const request = mooncakeRequest(requestId)
  const prefill = PREFILL_INSTANCES.map(instance => {
    const hit = prefixHit(request.blocks, instance.cached)
    const miss = request.blocks.length - hit
    return {
      id: instance.id,
      label: instance.label,
      hit,
      miss,
      queued: instance.queued,
      // Queued prefill work and the uncached suffix both wait on the same GPU.
      ttftMs: (instance.queued + miss) * BLOCK_PREFILL_MS,
    }
  })
  const decode = DECODE_INSTANCES.map(instance => {
    const after = instance.resident + request.blocks.length + GENERATED_BLOCKS
    return {
      id: instance.id,
      label: instance.label,
      resident: instance.resident,
      after,
      fits: after <= DECODE_VRAM_SLOTS,
    }
  })
  const prefillPick = prefill.reduce((best, route) =>
    route.ttftMs < best.ttftMs || (route.ttftMs === best.ttftMs && route.queued < best.queued)
      ? route
      : best,
  )
  const decodePick = decode
    .filter(route => route.fits)
    .reduce((best, route) => (route.after < best.after ? route : best), decode[0])
  return { request, prefill, decode, prefillId: prefillPick.id, decodeId: decodePick.id }
}

// Arriving while prefill streams, landed at transfer, settled once decode reads it.
const DECODE_POOL_BY_STAGE: CellState[] = ['empty', 'pending', 'new', 'kept']

const STAGE_INDEX: Record<MooncakeStage, number> = { route: 0, prefill: 1, transfer: 2, decode: 3 }

export const mooncakeStage = (value: string | undefined): MooncakeStage =>
  MOONCAKE_STAGES.find(stage => stage === value) ?? 'route'

export type MooncakeView = { cells: Record<string, CellState>; active: Record<string, boolean> }

// Cell keys: `prompt-i`, `<inst>-queue-i`, `<inst>-vram-i`, `<inst>-pool-i`. Active keys mark the
// chosen instances, the scheduler doing the work, and the RDMA path in use.
export function mooncakeView(plan: MooncakePlan, stage: MooncakeStage): MooncakeView {
  const at = STAGE_INDEX[stage]
  const cells: Record<string, CellState> = {}
  const active: Record<string, boolean> = {}
  const chosen = plan.prefill.find(route => route.id === plan.prefillId)!
  const n = plan.request.blocks.length

  for (let i = 0; i < n; i++) {
    const cached = i < chosen.hit
    cells[`prompt-${i}`] = at >= 2 ? 'kept' : cached ? 'hit' : at === 1 ? 'new' : 'pending'
  }

  for (const instance of PREFILL_INSTANCES) {
    const picked = instance.id === plan.prefillId
    for (let i = 0; i < QUEUE_SLOTS; i++)
      cells[`${instance.id}-queue-${i}`] = i < instance.queued ? 'other' : 'empty'
    for (let i = 0; i < PROMPT_BLOCKS; i++) {
      let state: CellState = 'empty'
      if (picked && i < n && at === 1) state = i < chosen.hit ? 'hit' : 'new'
      else if (picked && i < n && at === 2) state = 'kept'
      cells[`${instance.id}-vram-${i}`] = state
    }
    for (let i = 0; i < POOL_SLOTS; i++) {
      let state: CellState = i < instance.cached.length ? 'other' : 'empty'
      if (picked && i < chosen.hit && at <= 1) state = 'hit'
      // Prefill writes the whole prompt's KV back to the pool, so the new suffix is reusable too.
      const stored = i >= instance.cached.length && i < instance.cached.length + chosen.miss
      if (picked && at >= 2 && (i < chosen.hit || stored)) state = 'kept'
      cells[`${instance.id}-pool-${i}`] = state
    }
  }

  for (const instance of DECODE_INSTANCES) {
    const picked = instance.id === plan.decodeId
    for (let i = 0; i < POOL_SLOTS; i++)
      cells[`${instance.id}-pool-${i}`] = picked && i < n ? DECODE_POOL_BY_STAGE[at] : 'empty'
    for (let i = 0; i < DECODE_VRAM_SLOTS; i++) {
      let state: CellState = i < instance.resident ? 'other' : 'empty'
      const own = i - instance.resident
      if (picked && at === 3 && own >= 0 && own < n) state = 'kept'
      if (picked && at === 3 && own >= n && own < n + GENERATED_BLOCKS) state = 'new'
      cells[`${instance.id}-vram-${i}`] = state
    }
  }

  // Messenger streams each finished layer during prefill, so the link is live in both stages.
  const streaming = at === 1 || at === 2
  active[plan.prefillId] = at <= 2
  active[plan.decodeId] = true
  active['sched-prefill'] = at === 0
  active['sched-decode'] = at === 0
  active['rdma'] = streaming
  active[`link-${plan.prefillId}`] = streaming
  active[`link-${plan.decodeId}`] = streaming
  return { cells, active }
}
