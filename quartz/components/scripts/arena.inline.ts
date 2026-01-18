import { Cmd, none, start } from "../../functional"
import { mountArena, runArenaEffect, reduce } from "../arena"
import type { ArenaEffect, ArenaModel } from "../arena/model"

document.addEventListener("nav", () => {
  const program = start({
    init: (): { model: ArenaModel; effects: Cmd<ArenaEffect> } => ({
      model: { ready: false },
      effects: none(),
    }),
    reduce,
    effects: (effect) => runArenaEffect(effect),
  })

  const cleanup = mountArena(program.dispatch)
  program.dispatch({ type: "nav.ready" })

  const stop = () => {
    if (cleanup) cleanup()
    program.stop()
  }

  window.addCleanup(stop)
})
