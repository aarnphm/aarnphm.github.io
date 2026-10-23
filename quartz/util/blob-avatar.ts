import { mulberry } from '../components/scripts/404-pixel'
import {
  curve,
  filled,
  hand,
  INK,
  loop,
  PAPER,
  pressure,
  ring,
  stroke,
  type Pt,
} from './mascot/pen'
import { beret, bucketHat, sprout, sweatband } from './mascot/small-props'

// Commenter avatars: a blob from the pet's family, drawn with the same pen. A name picks every trait
// and the outline's wobble, so the same handle is the same blob wherever it shows; a commenter can
// pin traits from the palette, saved as a short code ("b1.s2.f4.e1.m0.h3.g1").

export const BLOB_TRAITS = {
  shape: ['round', 'squat', 'egg', 'boxy', 'drop', 'cloud'],
  fill: ['#fdb2a2', '#cdd597', '#f9cf9e', '#f2df9a', '#b9d6e6', '#cdc3e8', '#a9dccd', '#f4b9cf'],
  eyes: ['dash', 'pill', 'dot', 'happy', 'wink', 'wide'],
  mouth: ['none', 'o', 'flat'],
  hat: ['none', 'sprout', 'beret', 'band', 'bucket', 'antenna'],
  gaze: ['left', 'centre', 'right'],
} as const

export type BlobTrait = keyof typeof BLOB_TRAITS
export type BlobTraits = Record<BlobTrait, number>

const letters: Record<BlobTrait, string> = {
  shape: 's',
  fill: 'f',
  eyes: 'e',
  mouth: 'm',
  hat: 'h',
  gaze: 'g',
}
const traitNames = Object.keys(BLOB_TRAITS) as BlobTrait[]

// FNV-1a: a stable 32-bit seed from a name.
export function blobSeed(name: string): number {
  let h = 0x811c9dc5
  for (const ch of name.trim().toLowerCase()) {
    h ^= ch.codePointAt(0)!
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

export function blobTraits(name: string): BlobTraits {
  const rand = mulberry(blobSeed(name))
  const pick = (t: BlobTrait) => Math.floor(rand() * BLOB_TRAITS[t].length)
  // Most blobs go bare-headed and open-eyed; hats and odd eyes stay a treat.
  const traits = Object.fromEntries(traitNames.map(t => [t, pick(t)])) as BlobTraits
  if (rand() < 0.55) traits.hat = 0
  if (rand() < 0.4) traits.eyes = rand() < 0.5 ? 0 : 1
  return traits
}

export function encodeBlob(traits: BlobTraits): string {
  return ['b1', ...traitNames.map(t => `${letters[t]}${traits[t]}`)].join('.')
}

// The traits a code pins, ignoring anything out of range so an old code never breaks a blob.
export function decodeBlob(code: string): Partial<BlobTraits> {
  const [version, ...fields] = code.split('.')
  if (version !== 'b1') return {}
  const out: Partial<BlobTraits> = {}
  for (const field of fields) {
    const trait = traitNames.find(t => letters[t] === field[0])
    const value = Number(field.slice(1))
    if (trait && Number.isInteger(value) && value >= 0 && value < BLOB_TRAITS[trait].length)
      out[trait] = value
  }
  return out
}

// The body's outline: the pet's hand circle, bent into the trait's shape.
function outline(shape: number, seed: number): Pt[] {
  return loop(seed % 997, 0.05, 0.03, 160).map(([x, y]) => {
    const a = Math.atan2(y, x)
    const r = Math.hypot(x, y)
    switch (BLOB_TRAITS.shape[shape]) {
      case 'squat':
        return [x * 1.12, y * 0.86 + 0.1] as const
      case 'egg':
        return [x * (0.9 + 0.12 * y), y * 1.04] as const
      case 'boxy': {
        const k = 0.94 * (Math.abs(Math.cos(a)) ** 4 + Math.abs(Math.sin(a)) ** 4) ** -0.25
        return [x * k, y * k * 0.94 + 0.04] as const
      }
      case 'drop': {
        const pinch = 1 - 0.38 * Math.max(0, -y) ** 2
        return [x * pinch * 1.04, y * 1.04] as const
      }
      case 'cloud': {
        const lobe = 0.93 + 0.09 * Math.abs(Math.sin(2.5 * (a + Math.PI / 2)))
        return [r * lobe * Math.cos(a) * 1.04, r * lobe * Math.sin(a) * 0.96] as const
      }
      default:
        return [x, y] as const
    }
  })
}

function eyes(kind: number, gaze: number, w: number, seed: number): string {
  const dx = (gaze - 1) * 0.1
  const xs = [-0.42 + dx, 0.24 + dx]
  const y = -0.16
  const dash = (x: number) =>
    stroke(
      hand(
        [
          [x, y - 0.2],
          [x + 0.02, y + 0.16],
        ],
        0.04,
        0.008,
        seed + x * 10,
      ),
      t => w * (1.05 + 0.35 * Math.sin(Math.PI * t)),
      INK,
    )
  const shut = (x: number, up: boolean) =>
    stroke(
      curve([x - 0.13, y], [x, y + (up ? -0.17 : 0.15)], [x + 0.13, y], 10),
      t => w * (0.8 + 0.5 * Math.sin(Math.PI * t)),
      INK,
    )
  switch (BLOB_TRAITS.eyes[kind]) {
    case 'pill':
      return xs
        .map(
          (x, i) =>
            `<path d="M${x - 0.065} ${y - 0.13}a0.065 0.065 0 0 1 0.13 0v0.26a0.065 0.065 0 0 1-0.13 0z" transform="rotate(${i ? 6 : -4} ${x} ${y})" style="fill:${INK}"/>`,
        )
        .join('')
    case 'dot':
      return xs.map(x => `<circle cx="${x}" cy="${y}" r="0.1" style="fill:${INK}"/>`).join('')
    // Blobatar's happy: no mouth; the eyes squash into short thick bars, raised and spread a
    // little, both leaning the same way.
    case 'happy':
      return xs
        .map((x, i) => {
          const cx = x + (i ? 0.04 : -0.04)
          const a = (-8 * Math.PI) / 180
          const [ux, uy] = [0.1 * Math.cos(a), 0.1 * Math.sin(a)]
          return stroke(
            [
              [cx - ux, y - 0.05 - uy],
              [cx + ux, y - 0.05 + uy],
            ],
            () => 0.15,
            INK,
          )
        })
        .join('')
    case 'wink':
      return dash(xs[0]) + shut(xs[1], true)
    case 'wide':
      return xs
        .map(
          (x, i) =>
            filled(ring(seed + i, x, y, 0.17), PAPER, () => w * 0.8, INK) +
            `<circle cx="${x + (gaze - 1) * 0.06}" cy="${y + 0.02}" r="0.075" style="fill:${INK}"/>`,
        )
        .join('')
    default:
      return xs.map(dash).join('')
  }
}

function mouth(kind: number, gaze: number, w: number, seed: number): string {
  const x = -0.1 + (gaze - 1) * 0.1
  const y = 0.3
  const pen = (pts: Pt[]) => stroke(hand(pts, 0.04, 0.006, seed), pressure(seed, w, 0.15), INK)
  switch (BLOB_TRAITS.mouth[kind]) {
    case 'o':
      return filled(ring(seed, x, y + 0.04, 0.08), INK, () => w * 0.6, INK)
    case 'flat':
      return pen([
        [x - 0.12, y + 0.04],
        [x + 0.12, y + 0.05],
      ])
    default:
      return ''
  }
}

function hat(kind: number, w: number, fill: string): string {
  switch (BLOB_TRAITS.hat[kind]) {
    case 'sprout':
      return sprout(w * 0.8)
    case 'beret':
      return beret(w * 0.8)
    case 'band':
      return sweatband(w * 0.8)
    case 'bucket':
      return bucketHat(w * 0.3, PAPER)
    case 'antenna':
      return (
        stroke(
          hand(curve([0.05, -0.92], [0.2, -1.2], [0.08, -1.42], 10), 0.03, 0.008, 61),
          t => w * (1 - 0.3 * t),
          INK,
        ) +
        filled(
          ring(62, 0.08, -1.5, 0.11),
          fill === '#fdb2a2' ? '#cdd597' : '#fdb2a2',
          () => w * 0.7,
          INK,
        )
      )
    default:
      return ''
  }
}

// The blob as a standalone SVG. The view leaves headroom for a hat, so blobs with and without one
// stay the same size side by side.
export function blobSvg(name: string, pinned: Partial<BlobTraits> = {}): string {
  const t = { ...blobTraits(name), ...pinned }
  const seed = blobSeed(name)
  const w = 0.15
  const fill = BLOB_TRAITS.fill[t.fill]
  const body = outline(t.shape, seed)
  // Hats are drawn for the pet's round head, whose top is y = -1; this sets them on this body's.
  const top = Math.min(...body.map(([, y]) => y))
  // A paper rim, as on the pet, keeps the ink outline readable on a dark page.
  const rim = `<filter id="blob-rim" filterUnits="userSpaceOnUse" x="-1.62" y="-1.78" width="3.24" height="3.24"><feMorphology in="SourceAlpha" operator="dilate" radius="0.07" result="grown"/><feFlood flood-color="${PAPER}"/><feComposite operator="in" in2="grown"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>`
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.62 -1.78 3.24 3.24"><defs>${rim}</defs><g filter="url(#blob-rim)">${filled(body, fill, pressure(seed % 89, w, 0.3), INK)}${eyes(t.eyes, t.gaze, w, seed % 53)}${mouth(t.mouth, t.gaze, w, seed % 61)}<g transform="translate(0 ${(top + 1).toFixed(3)})">${hat(t.hat, w, fill)}</g></g></svg>`
}

export const blobDataUrl = (name: string, pinned: Partial<BlobTraits> = {}) =>
  `data:image/svg+xml;charset=utf-8,${encodeURIComponent(blobSvg(name, pinned))}`
