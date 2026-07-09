import type { ArenaEffect, ArenaModel } from '../arena/model'
import { Cmd, none, start } from '../../functional'
import { mountArena, runArenaEffect, reduce } from '../arena'
import { currentNavSignal } from './nav-lifecycle'

let activeArenaSignal: AbortSignal | undefined

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeArenaSignal === signal) return
  activeArenaSignal = signal

  const program = start({
    init: (): { model: ArenaModel; effects: Cmd<ArenaEffect> } => ({
      model: { ready: false },
      effects: none(),
    }),
    reduce,
    effects: effect => runArenaEffect(effect),
  })

  const cleanup = mountArena(program.dispatch)
  program.dispatch({ type: 'nav.ready' })

  let stopped = false
  const stop = () => {
    if (stopped) return
    stopped = true
    if (cleanup) cleanup()
    program.stop()
    if (activeArenaSignal === signal) activeArenaSignal = undefined
  }

  window.addCleanup(stop)
})
