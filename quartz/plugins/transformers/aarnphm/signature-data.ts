import sharp from 'sharp'
import { firstBanks } from './signature-banks-first'
import { lastBanks } from './signature-banks-last'

export const banks = { ...firstBanks, ...lastBanks } as const

export const glyphs = {
  '.': {
    d: 'M2 32.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z',
    opts: { viewBox: '0 4 1 51', width: '13', height: '51' },
    name: 'dot',
  },
  '-': { d: 'M1 32.5h10', opts: { viewBox: '0 4 12 51', width: '13', height: '51' }, name: 'dash' },
} as const

export type EmailSignatureAsset = {
  contentType: 'image/png'
  content: string
  width: number
  height: number
  text: string
}

const parseNumber = (value: string): number => {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const parseViewBox = (value: string): [number, number, number, number] | null => {
  const segments = value
    .trim()
    .split(/[\s,]+/)
    .map(part => Number.parseFloat(part))
  if (segments.length !== 4 || segments.some(segment => Number.isNaN(segment))) return null
  return [segments[0], segments[1], segments[2], segments[3]]
}

const readSignatureStroke = (value: string): string => value.replace(/["<>]/g, '')

const getSignatureVector = (
  char: string,
): {
  d: string
  viewBox: [number, number, number, number]
  width: number
  height: number
} | null => {
  const glyph = glyphs[char as keyof typeof glyphs]
  if (glyph) {
    const viewBox = parseViewBox(glyph.opts.viewBox)
    if (!viewBox) return null
    return {
      d: glyph.d,
      viewBox,
      width: parseNumber(glyph.opts.width),
      height: parseNumber(glyph.opts.height),
    }
  }
  const key = char.toLowerCase() as keyof typeof banks
  const bank = banks[key]
  if (!bank) return null
  const shape = char === char.toUpperCase() ? bank.upper : bank.lower
  const viewBox = parseViewBox(shape.opts.viewBox)
  if (!viewBox) return null
  return {
    d: shape.d,
    viewBox,
    width: parseNumber(shape.opts.width),
    height: parseNumber(shape.opts.height),
  }
}

const buildSignatureSvg = (
  text: string,
  stroke: string,
): { svg: string; width: number; height: number } | null => {
  const groups: string[] = []
  let cursorX = 0
  let maxHeight = 51
  const normalizedStroke = readSignatureStroke(stroke)

  for (const char of text) {
    if (char === ' ') {
      cursorX += 9.6
      continue
    }
    const vector = getSignatureVector(char)
    if (!vector) {
      cursorX += 8
      continue
    }
    const [minX, minY, viewWidth, viewHeight] = vector.viewBox
    if (viewWidth <= 0 || viewHeight <= 0 || vector.width <= 0 || vector.height <= 0) continue
    const scaleX = vector.width / viewWidth
    const scaleY = vector.height / viewHeight
    groups.push(
      `<g transform="translate(${cursorX} 0) scale(${scaleX} ${scaleY}) translate(${-minX} ${-minY})"><path d="${vector.d}" fill="none" stroke="${normalizedStroke}" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/></g>`,
    )
    cursorX += vector.width
    maxHeight = Math.max(maxHeight, vector.height)
  }

  if (groups.length === 0) return null
  const width = Math.max(1, Math.ceil(cursorX))
  const height = Math.max(1, Math.ceil(maxHeight))
  return {
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" fill="none">${groups.join('')}</svg>`,
    width,
    height,
  }
}

export const buildEmailSignatureAsset = async (
  text: string,
  stroke: string,
): Promise<EmailSignatureAsset | null> => {
  const rendered = buildSignatureSvg(text, stroke)
  if (!rendered) return null
  try {
    const png = await sharp(Buffer.from(rendered.svg)).png().toBuffer()
    return {
      contentType: 'image/png',
      content: png.toString('base64'),
      width: rendered.width,
      height: rendered.height,
      text,
    }
  } catch {
    return null
  }
}
