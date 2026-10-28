import { col, dot, row, type Grid, type RGB } from '../404-pixel'
import { BARN, TROUGH, type Beast, type Herd } from './herd-state'

export type HerdInk = {
  // Coats (white, black, red), the muzzle, legs and hooves, and a ewe's fleece and face.
  coats: [RGB, RGB, RGB]
  muzzle: RGB
  leg: RGB
  fleece: RGB
  face: RGB
  // The barn's boards, its tin roof, its painted doors, and the dark inside it.
  boards: RGB
  roof: RGB
  trim: RGB
  dark: RGB
  // The farmer's straw hat, face and hands, and the bib and legs of the overalls.
  hat: RGB
  skin: RGB
  bib: RGB
  trousers: RGB
  // The pail and the feed in it.
  pail: RGB
  feed: RGB
}

// The barn, end on: boards under a tin roof pitched at forty-five degrees, a hay door up in the
// gable, and the big doors: shut, two painted leaves under a header; open, the dark inside with the
// leaves swung back flat against the wall either side, the left one first.
export function paintBarn(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd) {
  const i = col(g, BARN[0])
  const j = row(g, BARN[1])
  for (let dj = 1; dj <= 9; dj++) {
    const eave = dj < 4 ? -1 : dj === 4 ? 5 : 9 - dj
    const wall = dj <= 4 ? 4 : eave - 1
    for (let di = -wall; di <= wall; di++) dot(d, g, i + di, j - dj, ink.boards)
    if (eave >= 0) {
      dot(d, g, i - eave, j - dj, ink.roof)
      dot(d, g, i + eave, j - dj, ink.roof)
    }
  }
  for (let dj = 1; dj <= 3; dj++) {
    for (let di = -1; di <= 1; di++) {
      const ajar = h.doors === 2 || (h.doors === 1 && di < 1)
      dot(d, g, i + di, j - dj, ajar ? ink.dark : di === 0 && dj < 3 ? ink.boards : ink.trim)
    }
    if (h.doors > 0) dot(d, g, i - 2, j - dj, ink.trim)
    if (h.doors > 1) dot(d, g, i + 2, j - dj, ink.trim)
  }
  dot(d, g, i, j - 6, ink.dark)
}

// The trough, galvanised like the roof, with the feed heaped in it while there is any.
export function paintTrough(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd) {
  const i = col(g, TROUGH[0])
  const j = row(g, TROUGH[1])
  const heap = Math.ceil(h.trough * 3)
  for (let di = -2; di <= 2; di++) {
    const inside = Math.abs(di) < 2
    const fed = inside && Math.abs(di) * 2 < heap
    dot(d, g, i + di, j - 2, inside ? (fed ? ink.feed : ink.dark) : ink.roof)
    dot(d, g, i + di, j - 1, ink.roof)
  }
}

// The farmer in a straw hat and denim overalls, legs apart in a stride; at the doors, an arm out to
// the leaf in hand; carrying the pail of feed out to the trough, and tipping it in.
export function paintFarmer(d: Uint8ClampedArray, g: Grid, ink: HerdInk, h: Herd, time: number) {
  const f = h.farmer
  if (f.chore === 'home' || f.chore === 'make') return
  const i = col(g, f.x)
  const j = row(g, f.y)
  if (f.chore === 'walk' && Math.floor(time * 2.5) % 2) {
    dot(d, g, i - 1, j - 1, ink.trousers)
    dot(d, g, i + 1, j - 1, ink.trousers)
  } else dot(d, g, i, j - 1, ink.trousers)
  dot(d, g, i, j - 2, ink.bib)
  dot(d, g, i, j - 3, ink.skin)
  for (let di = -1; di <= 1; di++) dot(d, g, i + di, j - 4, ink.hat)
  if (f.chore === 'doors') {
    // Opening, the left leaf and then the right; shutting, the right and then the left.
    const left = h.out ? h.doors === 0 : h.doors === 1
    dot(d, g, i + (left ? -1 : 1), j - 2, ink.skin)
  } else if (f.chore === 'walk' && f.after === 'pour') dot(d, g, i + f.face, j - 1, ink.pail)
  else if (f.chore === 'pour') {
    dot(d, g, i + 1, j - 3, ink.pail)
    if (Math.floor(time * 6) % 2) dot(d, g, i + 1, j - 2, ink.feed)
  }
}

// A cow is six pixels long with her head, standing three and a half: two legs, two rows of body
// with her markings, the head carried level and the muzzle below it, down in the grass to graze or
// lifted to look, and the tail swinging at the flies; lying, the body sits on the grass. A calf is
// a row of body on two legs; a ewe a woolly row and a half on black legs with her black face.
export function paintBeast(
  d: Uint8ClampedArray,
  g: Grid,
  ink: HerdInk,
  b: Beast,
  time: number,
  n: number,
) {
  if (b.in) return
  const i = col(g, b.x)
  const j = row(g, b.y)
  const f = b.face
  const coat = ink.coats[b.coat]
  const head = ink.coats[b.head]
  const low = b.doing === 'lie' ? 1 : 0
  const stride =
    (b.doing === 'walk' || b.doing === 'run') &&
    Math.floor(time * (b.doing === 'run' ? 6 : 2.5) + n) % 2
  const up = b.doing === 'look' || b.doing === 'walk' || b.doing === 'run'
  if (b.kind === 'cow') {
    if (!low) for (const di of stride ? [-1, 0] : [-2, 1]) dot(d, g, i + f * di, j - 1, ink.leg)
    for (let k = 0; k < 8; k++) {
      const di = (k % 4) - 2
      const dj = k < 4 ? -3 : -2
      dot(d, g, i + f * di, j + dj + low, b.spots & (1 << k) ? ink.coats[1] : coat)
    }
    if (low) {
      dot(d, g, i + 2 * f, j - 2, head)
      dot(d, g, i + 3 * f, j - 2, ink.muzzle)
    } else if (b.doing === 'graze' || b.doing === 'eat') {
      dot(d, g, i + 2 * f, j - 2, head)
      dot(d, g, i + 3 * f, j - 1, ink.muzzle)
    } else if (b.doing === 'look') {
      dot(d, g, i + 2 * f, j - 3, coat)
      dot(d, g, i + 2 * f, j - 4, head)
      dot(d, g, i + 3 * f, j - 4, ink.muzzle)
    } else {
      dot(d, g, i + 2 * f, j - 3, head)
      dot(d, g, i + 3 * f, j - 2, ink.muzzle)
    }
    const swish = Math.sin(time * 1.7 + n * 2.3) > 0.85 ? 1 : 0
    dot(d, g, i - 3 * f, j - 2 + low - swish, b.coat === 0 ? ink.coats[1] : coat)
  } else if (b.kind === 'calf') {
    if (!low) for (const di of stride ? [0] : [-1, 1]) dot(d, g, i + f * di, j - 1, ink.leg)
    for (let di = -1; di <= 1; di++)
      dot(d, g, i + f * di, j - 2 + low, b.spots & (1 << (di + 1)) ? ink.coats[1] : coat)
    if (up && !low) {
      dot(d, g, i + 2 * f, j - 3, head)
      dot(d, g, i + 3 * f, j - 3, ink.muzzle)
    } else {
      dot(d, g, i + 2 * f, j - 2, head)
      if (!low) dot(d, g, i + 3 * f, j - 1, ink.muzzle)
    }
  } else {
    if (!low) for (const di of stride ? [0] : [-1, 1]) dot(d, g, i + f * di, j - 1, ink.face)
    for (let di = -1; di <= 1; di++) dot(d, g, i + f * di, j - 2 + low, ink.fleece)
    dot(d, g, i, j - 3 + low, ink.fleece)
    dot(d, g, i - f, j - 3 + low, ink.fleece)
    dot(d, g, i + 2 * f, j - (up && !low ? 3 : 2), ink.face)
  }
}
