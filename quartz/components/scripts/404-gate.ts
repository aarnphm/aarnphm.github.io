// Geometry of the clock gate, shared by the SVG ruin and the scripts that paint around it: a ring of
// twelve voussoirs, one per hour, standing on the steps with its opening onto the rift. Screen y
// runs down, so angles grow clockwise, the way the hands turn.
export const GATE = { x: 800, y: 440, r: 168, ring: 46, depth: 14 }
export const STEP_TOP = 690
// The eight o'clock stone has fallen out of the ring and lies on the bank.
export const MISSING = 8

export const hourAngle = (h: number) => ((h - 3) * Math.PI) / 6

// Quarter stones stand proud of the ring, the keystone further still.
export const stoneOuter = (h: number) =>
  GATE.r + GATE.ring + (h % 12 === 0 ? 38 : h % 3 === 0 ? 24 : 0)

// Stones the roots have worked loose: pushed out along the ring (`out`), dropped (`drop`), and
// turned about their own middles (`turn`, degrees). The keystone has sagged into the opening.
const SLIP: Record<number, { out: number; drop: number; turn: number }> = {
  0: { out: 0, drop: 9, turn: -3 },
  4: { out: 3, drop: 4, turn: -4 },
  7: { out: 0, drop: 6, turn: 3 },
  10: { out: 10, drop: 0, turn: 5 },
}

// The middle of a voussoir, which it turns about when it slips.
export function stoneMiddle(h: number): [number, number] {
  const r = (GATE.r + stoneOuter(h)) / 2
  return [GATE.x + Math.cos(hourAngle(h)) * r, GATE.y + Math.sin(hourAngle(h)) * r]
}

// The SVG transform that carries a stone from its place in the ring to where it has slipped.
export function slipTransform(h: number) {
  const s = SLIP[h]
  if (!s) return undefined
  const [cx, cy] = stoneMiddle(h)
  const a = hourAngle(h)
  const dx = Math.cos(a) * s.out
  const dy = Math.sin(a) * s.out + s.drop
  return `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) rotate(${s.turn} ${cx.toFixed(2)} ${cy.toFixed(2)})`
}

// Where a point on stone `h` has ended up, matching slipTransform.
export function slipPoint(h: number, x: number, y: number): [number, number] {
  const s = SLIP[h]
  if (!s) return [x, y]
  const [cx, cy] = stoneMiddle(h)
  const t = (s.turn * Math.PI) / 180
  const a = hourAngle(h)
  const rx = cx + (x - cx) * Math.cos(t) - (y - cy) * Math.sin(t)
  const ry = cy + (x - cx) * Math.sin(t) + (y - cy) * Math.cos(t)
  return [rx + Math.cos(a) * s.out, ry + Math.sin(a) * s.out + s.drop]
}

// A point on the ring in polar form about the gate's centre, carried along with its stone.
export function onStone(h: number, r: number, a: number): [number, number] {
  return slipPoint(h, GATE.x + Math.cos(a) * r, GATE.y + Math.sin(a) * r)
}
