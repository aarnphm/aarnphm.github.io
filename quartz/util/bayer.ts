// The 4 × 4 ordered-dither screen every pixel scene shares (the homepage trail and the 404), as
// thresholds at the cell centres, 0..1. A tone covers a cell when its level exceeds the threshold.
export const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)

export const bayerThreshold = (column: number, row: number) => BAYER[(row & 3) * 4 + (column & 3)]
