import type { TreeKind } from './scenery'
import { trailRandom } from './random'

// One forest year passes each real week, counted from the census, when the planner's trees were the
// standing crop. Each tree slot keeps its identity (and its shared watering) while its occupants
// grow, die, stand as snags, fall, and give way to a seedling the old trunk nurses.
const census = Date.UTC(2025, 9, 15)
const week = 7 * 86_400_000

type Habit = {
  // Typical lifespan and the age at which the crown reaches nine tenths of its height, in years.
  span: number
  mature: number
  // Shade tolerance from 0 (aspen) to 1 (hemlock): what decides who fills a gap.
  shade: number
}
const habits: Record<TreeKind, Habit> = {
  pine: { span: 220, mature: 45, shade: 0.35 },
  spruce: { span: 250, mature: 50, shade: 0.7 },
  fir: { span: 160, mature: 40, shade: 0.85 },
  hemlock: { span: 400, mature: 70, shade: 0.95 },
  cypress: { span: 300, mature: 55, shade: 0.6 },
  larch: { span: 160, mature: 35, shade: 0.15 },
  birch: { span: 80, mature: 22, shade: 0.2 },
  aspen: { span: 70, mature: 18, shade: 0.1 },
  oak: { span: 320, mature: 60, shade: 0.45 },
  maple: { span: 260, mature: 45, shade: 0.9 },
  willow: { span: 60, mature: 14, shade: 0.2 },
  sequoia: { span: 1500, mature: 120, shade: 0.4 },
  snag: { span: 0, mature: 1, shade: 0 },
}
const needled = new Set<TreeKind>(['pine', 'spruce', 'fir', 'hemlock', 'cypress', 'larch'])

export type Occupant = {
  kind: TreeKind
  treeSeed: number
  height: number
  // Share of the full height reached, from 0 for a seedling to 1.
  growth: number
  age: number
  generation: number
  // The seed of the fallen trunk this tree grows from, while its log lasts.
  log: number | null
}

export const shadeTolerance = (kind: TreeKind) => habits[kind].shade

export const forestYear = (now: number) => (now - census) / week

// A storm flattens a grove about once a century. Each grove keeps its own record.
export function groveStorms(grove: number, year: number) {
  const storms: number[] = []
  for (let at = 0; at <= year; at++)
    if (trailRandom(Math.imul(grove + 1, 0x2545f491) ^ Math.imul(at + 1, 0x9e3779b1))() < 1 / 110)
      storms.push(at)
  return storms
}

// Horn's replacement rule: a gap one tree wide goes to the saplings already waiting in the shade;
// open ground after a storm goes to the pioneers. A stand list names each kind once per share.
function successor(pool: readonly TreeKind[], previous: TreeKind, opened: boolean, roll: number) {
  const weights = pool.map(kind => {
    const { shade } = habits[kind]
    const light = opened ? 1.3 - shade : 0.25 + shade
    return light * (kind === previous && !opened ? shade : 1)
  })
  let pick = roll * weights.reduce((sum, weight) => sum + weight, 0)
  for (let index = 0; index < pool.length; index++) {
    pick -= weights[index]
    if (pick < 0) return pool[index]
  }
  return pool[0]
}

// The occupant of one slot in a given forest year. The founder is the planner's tree, of an age
// drawn from its identity; a founding snag is a trunk that died before the census.
export function slotOccupant(
  founder: { kind: TreeKind; treeSeed: number; height: number },
  slot: number,
  pool: readonly TreeKind[],
  storms: readonly number[],
  year: number,
  // A lone tree stands in full sun, so its gap is always open ground.
  open = false,
): Occupant & { stage: 'living' | 'snag' } {
  let { kind, treeSeed, height } = founder
  const draw = trailRandom(treeSeed ^ 0x51ed2705)
  let previous: TreeKind = kind === 'snag' ? pool[Math.floor(draw() * pool.length)] : kind
  let born = kind === 'snag' ? -draw() * 40 : -(0.12 + 0.98 * draw() ** 0.8) * habits[previous].span
  let log: number | null = null
  let fallen = -Infinity
  for (let generation = 0; generation < 64; generation++) {
    const random = trailRandom(treeSeed ^ Math.imul(generation + 1, 0x85ebca6b))
    const habit = habits[previous]
    const natural = kind === 'snag' ? born : born + habit.span * (0.75 + random() * 0.5)
    // A storm takes most standing trees of its grove; seedlings bend and survive.
    const storm = storms.find(
      at =>
        at > born + habit.mature * 0.4 &&
        at < natural &&
        trailRandom(Math.imul(slot + 1, 0x27d4eb2f) ^ at)() < 0.75,
    )
    const death = storm ?? natural
    const broadleaf = !needled.has(previous)
    const fall = storm !== undefined ? death : death + (broadleaf ? 5 : 12) + random() * 18
    const rot = (broadleaf ? 12 : 22) + random() * 20
    const growth = (age: number) => 1 - Math.exp((-2.3 * Math.max(0, age)) / habit.mature)
    const current = log !== null && year < fallen ? log : null
    if (year < death)
      return {
        kind,
        treeSeed,
        height: 16 + (height - 16) * growth(year - born),
        growth: growth(year - born),
        age: year - born,
        generation,
        log: current,
        stage: 'living',
      }
    if (year < fall)
      return {
        kind: 'snag',
        treeSeed,
        height: 16 + (height - 16) * growth(death - born),
        growth: 1,
        age: year - born,
        generation,
        log: current,
        stage: 'snag',
      }
    const next = successor(pool, previous, open || storm !== undefined, random())
    log = treeSeed
    fallen = fall + rot
    previous = next
    kind = next
    treeSeed = Math.floor(random() * 4294967296)
    height = 30 + random() * 26
    born = fall
  }
  return { kind, treeSeed, height, growth: 1, age: 0, generation: 64, log: null, stage: 'living' }
}
