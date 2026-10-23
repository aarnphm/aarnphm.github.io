// Geometry and numbers shared by the SSR figure and multi-head-attention.inline.ts.
export const HEAD_OPTIONS = [1, 2, 4, 8] as const
export const DEFAULT_HEADS = 4
export const DEFAULT_DM = 512
export const SEQ_LEN = 4

export const VIEW_W = 760
export const VIEW_H = 560
export const PAD_L = 18
export const PAD_R = 18

export const COL_X = { input: 60, proj: 200, score: 332, head: 472, concat: 612, wo: 700 } as const
export const HEAD_BAND_TOP = 36
export const HEAD_BAND_BOT = VIEW_H - 36
export const HEAD_BAND_H = HEAD_BAND_BOT - HEAD_BAND_TOP

export const PROJ_W = 52
export const PROJ_H = 26
export const PROJ_GAP = 4
export const SCORE_BOX = 52
export const HEAD_OUT_W = 56
export const HEAD_OUT_H = 22

export const SCORE_PATTERNS: number[][][] = [
  [
    [0.85, 0.06, 0.04, 0.05],
    [0.05, 0.78, 0.12, 0.05],
    [0.04, 0.1, 0.82, 0.04],
    [0.05, 0.04, 0.06, 0.85],
  ],
  [
    [0.05, 0.85, 0.05, 0.05],
    [0.04, 0.04, 0.88, 0.04],
    [0.04, 0.05, 0.04, 0.87],
    [0.83, 0.06, 0.06, 0.05],
  ],
  [
    [0.62, 0.18, 0.12, 0.08],
    [0.16, 0.5, 0.22, 0.12],
    [0.1, 0.2, 0.46, 0.24],
    [0.06, 0.12, 0.28, 0.54],
  ],
  [
    [0.25, 0.25, 0.25, 0.25],
    [0.2, 0.3, 0.3, 0.2],
    [0.18, 0.32, 0.32, 0.18],
    [0.3, 0.2, 0.2, 0.3],
  ],
  [
    [0.7, 0.2, 0.06, 0.04],
    [0.5, 0.3, 0.15, 0.05],
    [0.3, 0.35, 0.25, 0.1],
    [0.1, 0.2, 0.35, 0.35],
  ],
  [
    [0.1, 0.2, 0.3, 0.4],
    [0.15, 0.25, 0.35, 0.25],
    [0.25, 0.35, 0.25, 0.15],
    [0.4, 0.3, 0.2, 0.1],
  ],
  [
    [0.88, 0.04, 0.04, 0.04],
    [0.06, 0.05, 0.84, 0.05],
    [0.78, 0.08, 0.08, 0.06],
    [0.06, 0.82, 0.06, 0.06],
  ],
  [
    [0.4, 0.1, 0.4, 0.1],
    [0.1, 0.4, 0.1, 0.4],
    [0.42, 0.08, 0.42, 0.08],
    [0.08, 0.42, 0.08, 0.42],
  ],
]

export const headBandY = (i: number, h: number): { top: number; mid: number; bot: number } => {
  const slot = HEAD_BAND_H / h
  const top = HEAD_BAND_TOP + slot * i
  return { top, mid: top + slot / 2, bot: top + slot }
}

export const fmtBytes = (n: number): string => {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${n} B`
}

export const fmtCompact = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

// The three projection boxes of a head fill its band up to PROJ_H, so a 12px label fits them at
// the inline width with up to four heads.
export const projLayout = (h: number): { height: number; gap: number } => {
  const slot = HEAD_BAND_H / h
  const gap = slot < 56 ? 2 : PROJ_GAP
  return { gap, height: Math.min(PROJ_H, Math.max(9, (slot - 8 - 2 * gap) / 3)) }
}
