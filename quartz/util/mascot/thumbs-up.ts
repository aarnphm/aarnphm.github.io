import { curve, densify, filled, hand, part, pressure, rounded, type PetPen, type Pt } from './pen'

// Turns `pts` by `deg` about (cx, cy).
const turn = (pts: Pt[], cx: number, cy: number, deg: number): Pt[] => {
  const r = (deg * Math.PI) / 180
  const [c, s] = [Math.cos(r), Math.sin(r)]
  return pts.map(
    ([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c] as const,
  )
}

// A thumbs up from the hand on its right (the far side as it faces left), in blobatar's chunky
// round hand: one curve for the arm, a soft paper fist, and the thumb as an outlined paper capsule
// leaning out from its top, with two short strokes for the curled fingers. The thumb is drawn first
// so the fist covers its root. In the snow the mitten lies over the fist with its own thumb up.
export function thumbsUp(s: PetPen) {
  const { L, w, paper, ink, pen, mitt } = s
  const arm = curve([0.92, 0.06], [1.24, 0.12], [1.3, -0.2], 14)
  const thumb = turn(hand(rounded(1.3, -0.58, 0.2, 0.34, 0.1), 0.02, 0.003, 64), 1.3, -0.44, 14)
  const fist = hand(rounded(1.38, -0.34, 0.38, 0.3, 0.14), 0.02, 0.004, 61)
  const curls = [-0.37, -0.28].map(y =>
    pen(curve([1.43, y], [1.5, y - 0.025], [1.56, y + 0.005], 8), () => w * 0.6),
  )
  const palm: Pt[] = [
    [1.0, -0.36],
    [1.22, -0.36],
  ]
  return part(
    'arm-thumb',
    pen(densify(arm, 0.05), pressure(62, L * 1.1, 0.2)) +
      filled(thumb, paper, pressure(64, L, 0.15), ink) +
      filled(fist, paper, pressure(63, L * 1.05, 0.2), ink) +
      curls.join('') +
      mitt(palm, -1, 1010),
  )
}
