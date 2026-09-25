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
