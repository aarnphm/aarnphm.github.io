import { oklabChroma, oklabDistance, type Rgb } from './okhsv'

const SIDE = 64
const toHex = (rgb: Rgb) => `#${rgb.map(c => Math.round(c).toString(16).padStart(2, '0')).join('')}`

/**
 * The card's main colours, for the picker's second chip row. Downscales the composed canvas to
 * 64px, buckets pixels at 5 bits a channel, then keeps the heaviest buckets that sit apart in
 * Oklab. Chroma weighs in so the photo and map beat the paper; noir and blanc have their own
 * segments, so near-black and near-white are left out. A tainted canvas yields nothing.
 */
export function cardColors(source: HTMLCanvasElement, count = 6): string[] {
  if (!source.width || !source.height) return []
  const scale = SIDE / Math.max(source.width, source.height)
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(source.width * scale))
  canvas.height = Math.max(1, Math.round(source.height * scale))
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) return []
  context.drawImage(source, 0, 0, canvas.width, canvas.height)
  let data: Uint8ClampedArray
  try {
    data = context.getImageData(0, 0, canvas.width, canvas.height).data
  } catch {
    return []
  }

  const buckets = new Map<number, { n: number; r: number; g: number; b: number }>()
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) continue
    const key = ((data[i] >> 3) << 10) | ((data[i + 1] >> 3) << 5) | (data[i + 2] >> 3)
    const bucket = buckets.get(key) ?? { n: 0, r: 0, g: 0, b: 0 }
    bucket.n++
    bucket.r += data[i]
    bucket.g += data[i + 1]
    bucket.b += data[i + 2]
    buckets.set(key, bucket)
  }

  const ranked = [...buckets.values()]
    .map(({ n, r, g, b }) => {
      const rgb: Rgb = [r / n, g / n, b / n]
      return { rgb, score: n * (0.2 + Math.min(1, oklabChroma(rgb) / 0.08)) }
    })
    .sort((x, y) => y.score - x.score)

  const picked: Rgb[] = []
  for (const { rgb } of ranked) {
    if (picked.length >= count) break
    if (oklabDistance(rgb, [0, 0, 0]) < 0.12 || oklabDistance(rgb, [255, 255, 255]) < 0.06) continue
    if (picked.some(other => oklabDistance(rgb, other) < 0.09)) continue
    picked.push(rgb)
  }
  return picked.map(toHex)
}
