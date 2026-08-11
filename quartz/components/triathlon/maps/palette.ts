export const rampGradient = (colors: readonly string[]): string =>
  `linear-gradient(to right, ${colors[0]}, ${colors[3]}, ${colors[6]})`

export const HEAT_RAMP = [
  '#997c6d',
  '#a9745b',
  '#b96c4a',
  '#ca6538',
  '#da5d27',
  '#ea5515',
  '#fc4c02',
]

export const ramp7 = (from: string, to: string): string[] => {
  const rgb = (hex: string): number[] => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ]
  const start = rgb(from)
  const end = rgb(to)
  return Array.from({ length: 7 }, (_, index) => {
    const fraction = index / 6
    return `#${start
      .map((value, channel) =>
        Math.round(value + (end[channel] - value) * fraction)
          .toString(16)
          .padStart(2, '0'),
      )
      .join('')}`
  })
}

export const HR_RAMP = ramp7('#9c7f7a', '#af3029')
export const CAD_RAMP = ramp7('#8a8197', '#5e409d')
export const SPD_RAMP = ramp7('#7d8a96', '#205ea6')
export const ELEV_RAMP = ramp7('#868a72', '#66800b')
export const RESP_RAMP = ramp7('#74898a', '#16878a')
export const STRIDE_RAMP = ramp7('#819078', '#3f7d57')
