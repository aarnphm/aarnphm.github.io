import {
  createMossGrid,
  damageMoss,
  decodeMossModel,
  type MossGrid,
  type MossModel,
  mossRandom,
  type MossRequest,
  type MossResponse,
  mossSize,
  mossTonesOf,
  seedMoss,
  stepMoss,
} from './moss-nca'

// Six updates a second: a clump grows in over about ten seconds and heals a boot print in
// twenty to forty on a wet, mild day. The weather paces healing: a healing clump steps at the
// growth rate, so dry moss heals slowly and frozen or buried moss keeps its prints until it
// thaws, and the wind regrows a print from its downwind rim first. A clump's first growth ignores
// the weather; the moss was there before the visit. Only clumps that are awake step; a grown,
// untouched clump costs nothing.
// Clumps wake in turn, a few hundred milliseconds apart, so the opening never steps them all.
const interval = 1000 / 6
const growSteps = 72
const settleSteps = 12
// The most steps one waking may run, so an unsettled clump cannot keep the worker busy.
const budget = 360
const stagger = 350

type Clump = {
  grid: MossGrid
  tones: Uint8Array
  // Steps left before the clump may sleep, and steps since its tones last changed.
  awake: number
  still: number
  quiet: boolean
  wakeAt: number
  spent: number
  // Healing after a boot print, and the fraction of a step owed at the current growth rate.
  healing: boolean
  owed: number
}

let model: MossModel | null = null
let random = mossRandom(1)
const clumps = new Map<string, Clump>()
let timer: ReturnType<typeof setTimeout> | null = null
let paused = false
let growth = 1
let lean = 0

const post = (message: MossResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer)

const tick = () => {
  timer = null
  if (!model || paused) return
  const started = performance.now()
  const paints: { id: string; tones: Uint8Array }[] = []
  const transfer: Transferable[] = []
  let awake = 0
  for (const [id, clump] of clumps) {
    if (clump.awake <= 0) continue
    if (clump.wakeAt > started) {
      awake++
      continue
    }
    if (clump.healing) {
      // Dormant moss waits for a climate message rather than keeping the timer alive.
      if (growth <= 0) continue
      clump.owed += growth
      if (clump.owed < 1) {
        awake++
        continue
      }
      clump.owed -= 1
    }
    stepMoss(clump.grid, model, random, clump.healing ? lean : 0)
    clump.awake--
    clump.spent++
    const changed = mossTonesOf(clump.grid, clump.tones)
    clump.still = changed ? 0 : clump.still + 1
    // A clump that is still changing as its allowance runs out keeps going a little longer.
    if (clump.awake <= 0 && clump.still < settleSteps && clump.spent < budget)
      clump.awake = settleSteps
    const asleep = clump.awake <= 0
    if (asleep) clump.healing = false
    else awake++
    let paint = changed
    if (clump.quiet) {
      paint = asleep
      if (asleep) clump.quiet = false
    }
    if (paint) {
      const tones = clump.tones.slice()
      paints.push({ id, tones })
      transfer.push(tones.buffer)
    }
  }
  if (paints.length || !awake)
    post({ type: 'paint', paints, cost: performance.now() - started, awake }, transfer)
  if (awake) schedule()
}
const schedule = () => {
  if (timer !== null || paused) return
  timer = setTimeout(tick, interval)
}

self.addEventListener('message', (event: MessageEvent<MossRequest>) => {
  const message = event.data
  switch (message.type) {
    case 'init':
      model = decodeMossModel(message.weights)
      random = mossRandom(message.seed)
      break
    case 'plant':
      for (const [index, id] of message.ids.entries()) {
        const grid = createMossGrid()
        seedMoss(grid)
        clumps.set(id, {
          grid,
          tones: new Uint8Array(mossSize * mossSize),
          awake: growSteps,
          still: 0,
          // Reduced motion grows the clump unseen and paints it once, fully grown.
          quiet: message.grown,
          wakeAt: message.grown ? 0 : performance.now() + index * stagger,
          spent: 0,
          healing: false,
          owed: 0,
        })
      }
      schedule()
      break
    case 'damage': {
      // The print shows at once, even on dormant moss that will not heal it for a while.
      const paints: { id: string; tones: Uint8Array }[] = []
      for (const { id, x, y, radius } of message.hits) {
        const clump = clumps.get(id)
        if (!clump || !damageMoss(clump.grid, x, y, radius)) continue
        clump.awake = Math.max(clump.awake, growSteps)
        clump.still = 0
        clump.spent = 0
        clump.healing = true
        if (!clump.quiet && mossTonesOf(clump.grid, clump.tones))
          paints.push({ id, tones: clump.tones.slice() })
      }
      if (paints.length)
        post(
          {
            type: 'paint',
            paints,
            cost: 0,
            awake: [...clumps.values()].filter(clump => clump.awake > 0).length,
          },
          paints.map(paint => paint.tones.buffer),
        )
      schedule()
      break
    }
    case 'clear':
      clumps.clear()
      break
    case 'pause':
      paused = true
      if (timer !== null) clearTimeout(timer)
      timer = null
      break
    case 'resume':
      paused = false
      schedule()
      break
    case 'climate':
      growth = Math.max(0, Math.min(1, message.growth))
      lean = Math.max(-1, Math.min(1, message.lean))
      schedule()
      break
  }
})
