import type { QuartzDevEvent } from './dev-events'

export type QuartzManagerState = {
  currentEpoch: string | null
  quartz: 'stopped' | 'building' | 'ready' | 'failed'
  wrangler: 'stopped' | 'starting' | 'ready' | 'stopping' | 'backoff'
  publicAvailable: boolean
}

export type QuartzManagerAction =
  | { type: 'stop-wrangler'; reason: string }
  | { type: 'schedule-wrangler-start'; delayMs: number }

export function createQuartzManagerState(): QuartzManagerState {
  return { currentEpoch: null, quartz: 'stopped', wrangler: 'stopped', publicAvailable: false }
}

export function resetQuartzManagerState(state: QuartzManagerState): void {
  state.currentEpoch = null
  state.quartz = 'stopped'
  state.wrangler = 'stopped'
  state.publicAvailable = false
}

export function applyQuartzDevEvent(
  state: QuartzManagerState,
  event: QuartzDevEvent,
  startDelayMs: number,
): QuartzManagerAction[] {
  switch (event.type) {
    case 'build:start':
      state.currentEpoch = event.epoch
      state.quartz = 'building'
      return []
    case 'public:remove:start':
      if (event.epoch !== state.currentEpoch) return []
      state.publicAvailable = false
      state.quartz = 'building'
      return [{ type: 'stop-wrangler', reason: 'stopping wrangler while public is regenerated' }]
    case 'build:ready':
      if (event.epoch !== state.currentEpoch) return []
      state.publicAvailable = true
      state.quartz = 'ready'
      return [{ type: 'schedule-wrangler-start', delayMs: startDelayMs }]
    case 'build:error':
      if (event.epoch !== state.currentEpoch) return []
      state.quartz = 'failed'
      return state.publicAvailable
        ? []
        : [{ type: 'stop-wrangler', reason: 'stopping wrangler after failed rebuild' }]
  }
}
