// A neural cellular automaton (Mordvintsev et al., "Growing Neural Cellular Automata", 2020)
// that grows one moss clump from a seed and regrows it after damage. Every cell holds sixteen
// channels: three premultiplied colour channels, alpha, and twelve hidden ones. Each step a cell
// perceives its 3x3 neighbourhood through fixed identity and Sobel filters, runs a two-layer
// perceptron, and adds the result with a 50 % chance. A cell whose neighbourhood has no alpha
// above 0.1 is dead and stays zero, so the update only visits living cells and their rim.

export const mossChannels = 16
export const mossSize = 32
export const mossTones = 3

export type MossModel = {
  hidden: number
  // Row-major: hidden x 48 and channels x hidden.
  w1: Float32Array
  b1: Float32Array
  w2: Float32Array
}
export type MossWeights = {
  hidden: number
  // Int8 weights with one scale per output row, base64-encoded.
  w1: string
  w1Scale: number[]
  b1: number[]
  w2: string
  w2Scale: number[]
}

export function decodeMossModel(weights: MossWeights): MossModel {
  const rows = (text: string, scales: number[]) => {
    const bytes = Uint8Array.from(atob(text), char => char.charCodeAt(0))
    const columns = bytes.length / scales.length
    const out = new Float32Array(bytes.length)
    for (let row = 0; row < scales.length; row++)
      for (let column = 0; column < columns; column++) {
        const index = row * columns + column
        out[index] = ((bytes[index] << 24) >> 24) * scales[row]
      }
    return out
  }
  return {
    hidden: weights.hidden,
    w1: rows(weights.w1, weights.w1Scale),
    b1: Float32Array.from(weights.b1),
    w2: rows(weights.w2, weights.w2Scale),
  }
}

// Target tone colours the clump was trained toward, unpremultiplied. Painting snaps every living
// cell to the nearest one, so the clump stays pixel art and takes the theme's colour for the tone.
const toneTargets = [
  [0x3d / 255, 0x4c / 255, 0x07 / 255],
  [0x66 / 255, 0x80 / 255, 0x0b / 255],
  [0xcd / 255, 0xd5 / 255, 0x97 / 255],
]

export type MossGrid = {
  state: Float32Array
  next: Float32Array
  alive: Uint8Array
  perception: Float32Array
  hidden: Float32Array
}

export function createMossGrid(): MossGrid {
  const cells = mossSize * mossSize
  return {
    state: new Float32Array(cells * mossChannels),
    next: new Float32Array(cells * mossChannels),
    alive: new Uint8Array(cells),
    perception: new Float32Array(mossChannels * 3),
    hidden: new Float32Array(0),
  }
}

export function seedMoss(grid: MossGrid, x = mossSize >> 1, y = mossSize >> 1) {
  grid.state.fill(0)
  grid.next.fill(0)
  const base = (y * mossSize + x) * mossChannels
  for (let channel = 3; channel < mossChannels; channel++) grid.state[base + channel] = 1
}

// Kills every cell within `radius` of (x, y), in cells.
export function damageMoss(grid: MossGrid, x: number, y: number, radius: number) {
  const { state } = grid
  const r2 = radius * radius
  let hit = false
  for (let cy = Math.max(0, Math.floor(y - radius)); cy <= Math.min(mossSize - 1, y + radius); cy++)
    for (
      let cx = Math.max(0, Math.floor(x - radius));
      cx <= Math.min(mossSize - 1, x + radius);
      cx++
    ) {
      if ((cx - x) ** 2 + (cy - y) ** 2 > r2) continue
      const base = (cy * mossSize + cx) * mossChannels
      if (state[base + 3] > 0.1) hit = true
      state.fill(0, base, base + mossChannels)
    }
  return hit
}

// Marks every cell whose 3x3 neighbourhood holds an alpha above 0.1.
function liveness(source: Float32Array, alive: Uint8Array) {
  const S = mossSize
  const C = mossChannels
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      let max = 0
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy
        if (ny < 0 || ny >= S) continue
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          if (nx < 0 || nx >= S) continue
          const alpha = source[(ny * S + nx) * C + 3]
          if (alpha > max) max = alpha
        }
      }
      alive[y * S + x] = max > 0.1 ? 1 : 0
    }
}

// One asynchronous update. Returns the number of cells visited. `lean` (-1..1) tilts the update
// chance across the grid, 0.1 on the upwind edge to 0.9 downwind in a gale: wind carries spores
// and spray to the lee side, so a print regrows from there first. The rule was trained at 0.5 and
// still heals every print in the same ~25 steps at these rates (sky-world-palette wind-heal.ts).
export function stepMoss(grid: MossGrid, model: MossModel, random: () => number, lean = 0) {
  const S = mossSize
  const C = mossChannels
  const P = C * 3
  const H = model.hidden
  const { state, next, alive, perception } = grid
  if (grid.hidden.length !== H) grid.hidden = new Float32Array(H)
  const hidden = grid.hidden
  const { w1, b1, w2 } = model
  liveness(state, alive)
  next.fill(0)
  let visited = 0
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      const cell = y * S + x
      if (!alive[cell]) continue
      visited++
      const base = cell * C
      if (random() >= 0.5 + 0.4 * lean * ((2 * x) / (S - 1) - 1)) {
        for (let channel = 0; channel < C; channel++) next[base + channel] = state[base + channel]
        continue
      }
      const up = y > 0 ? base - S * C : -1
      const down = y < S - 1 ? base + S * C : -1
      const left = x > 0
      const right = x < S - 1
      for (let channel = 0; channel < C; channel++) {
        const here = base + channel
        const l = left ? state[here - C] : 0
        const r = right ? state[here + C] : 0
        const t = up >= 0 ? state[up + channel] : 0
        const b = down >= 0 ? state[down + channel] : 0
        const tl = up >= 0 && left ? state[up + channel - C] : 0
        const tr = up >= 0 && right ? state[up + channel + C] : 0
        const bl = down >= 0 && left ? state[down + channel - C] : 0
        const br = down >= 0 && right ? state[down + channel + C] : 0
        perception[channel] = state[here]
        perception[C + channel] = (2 * (r - l) + (tr - tl) + (br - bl)) / 8
        perception[2 * C + channel] = (2 * (b - t) + (bl - tl) + (br - tr)) / 8
      }
      for (let h = 0; h < H; h++) {
        let sum = b1[h]
        const row = h * P
        for (let p = 0; p < P; p++) sum += w1[row + p] * perception[p]
        hidden[h] = sum > 0 ? sum : 0
      }
      for (let channel = 0; channel < C; channel++) {
        let sum = 0
        const row = channel * H
        for (let h = 0; h < H; h++) sum += w2[row + h] * hidden[h]
        next[base + channel] = state[base + channel] + sum
      }
    }
  // A cell lives on only when its neighbourhood stays alive after the update too.
  liveness(next, alive)
  for (let cell = 0; cell < S * S; cell++) if (!alive[cell]) next.fill(0, cell * C, cell * C + C)
  grid.state = next
  grid.next = state
  return visited
}

// Snaps each cell to a tone index: 0 is bare ground, 1 to 3 are the tones.
export function mossTonesOf(grid: MossGrid, tones: Uint8Array) {
  const { state } = grid
  const C = mossChannels
  let changed = false
  for (let cell = 0; cell < mossSize * mossSize; cell++) {
    const base = cell * C
    const alpha = state[base + 3]
    let tone = 0
    if (alpha > 0.1) {
      const scale = 1 / Math.max(alpha, 0.1)
      const r = state[base] * scale
      const g = state[base + 1] * scale
      const b = state[base + 2] * scale
      let best = Infinity
      for (let index = 0; index < toneTargets.length; index++) {
        const [tr, tg, tb] = toneTargets[index]
        const distance = (r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2
        if (distance < best) {
          best = distance
          tone = index + 1
        }
      }
    }
    if (tones[cell] !== tone) {
      tones[cell] = tone
      changed = true
    }
  }
  return changed
}

export function mossRandom(seed: number) {
  let state = seed >>> 0 || 0x9e3779b9
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

export type MossRequest =
  | { type: 'init'; weights: MossWeights; seed: number }
  | { type: 'plant'; ids: string[]; grown: boolean }
  | { type: 'damage'; hits: { id: string; x: number; y: number; radius: number }[] }
  | { type: 'clear' }
  | { type: 'pause' }
  | { type: 'resume' }
  // Healing speed from the weather, 0 dormant .. 1 a wet, mild day (world-schedule.ts), and the
  // wind across the clump, -1 blowing left .. 1 blowing right.
  | { type: 'climate'; growth: number; lean: number }
export type MossResponse =
  | { type: 'paint'; paints: { id: string; tones: Uint8Array }[]; cost: number; awake: number }
  | { type: 'asleep' }
