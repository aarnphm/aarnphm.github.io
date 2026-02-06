import type { Cmd } from './cmd'

export type Dispatch<Event> = (event: Event) => void

export type EffectContext<Model, Event> = { dispatch: Dispatch<Event>; retrieve: () => Model }

export type EffectRunner<Model, Event, Effect> = (
  effect: Effect,
  ctx: EffectContext<Model, Event>,
) => void | (() => void)

export type Program<Model, Event, Effect> = {
  init: () => { model: Model; effects: Cmd<Effect> }
  reduce: (model: Model, event: Event) => { model: Model; effects: Cmd<Effect> }
  effects: EffectRunner<Model, Event, Effect>
  subscriptions?: (ctx: EffectContext<Model, Event>) => void | (() => void)
}

export const start = <Model, Event, Effect>(program: Program<Model, Event, Effect>) => {
  let model: Model
  let running = true
  const cleanups: Array<() => void> = []

  const retrieve = () => model

  const runEffects = (effects: Cmd<Effect>, ctx: EffectContext<Model, Event>) => {
    for (const effect of effects) {
      const cleanup = program.effects(effect, ctx)
      if (cleanup) {
        cleanups.push(cleanup)
      }
    }
  }

  const dispatch: Dispatch<Event> = event => {
    if (!running) return
    const result = program.reduce(model, event)
    model = result.model
    runEffects(result.effects, { dispatch, retrieve })
  }

  const result = program.init()
  model = result.model

  if (program.subscriptions) {
    const cleanup = program.subscriptions({ dispatch, retrieve })
    if (cleanup) {
      cleanups.push(cleanup)
    }
  }

  runEffects(result.effects, { dispatch, retrieve })

  const stop = () => {
    if (!running) return
    running = false
    const pending = cleanups.slice()
    cleanups.length = 0
    for (const cleanup of pending) {
      cleanup()
    }
  }

  return { dispatch, retrieve, stop }
}
