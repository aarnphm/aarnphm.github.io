export type VisitorPrecipitation = { kind: 'rain' | 'snow' | null; strength: number }

const rainCodes = new Map([
  [51, 0.3],
  [53, 0.45],
  [55, 0.55],
  [56, 0.35],
  [57, 0.55],
  [61, 0.55],
  [63, 0.75],
  [65, 0.95],
  [66, 0.6],
  [67, 0.9],
  [80, 0.6],
  [81, 0.8],
  [82, 0.98],
  [95, 0.8],
  [96, 0.9],
  [97, 0.95],
  [99, 0.95],
])
const snowCodes = new Map([
  [71, 0.55],
  [73, 0.75],
  [75, 0.95],
  [77, 0.35],
  [85, 0.55],
  [86, 0.95],
])

export function visitorPrecipitation(code: number | null | undefined): VisitorPrecipitation {
  if (code === null || code === undefined) return { kind: null, strength: 0 }
  const rain = rainCodes.get(code)
  if (rain !== undefined) return { kind: 'rain', strength: rain }
  const snow = snowCodes.get(code)
  return snow === undefined ? { kind: null, strength: 0 } : { kind: 'snow', strength: snow }
}
