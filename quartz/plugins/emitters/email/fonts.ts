import { promises as fs } from 'node:fs'
import path from 'path'
import { QUARTZ } from '../../../util/path'

export const DEFAULT_SANS_SERIF =
  'system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"'
export const DEFAULT_MONO = 'ui-monospace, SFMono-Regular, SF Mono, Menlo, monospace'

type FontFaceEntry = {
  family: string
  filename: string
  format: string
  style: string
  weight: string
  path: string
  unicodeRange?: string
}

const normalizeFontFormat = (format: string): string => {
  const normalized = format.toLowerCase()
  if (normalized === 'ttf') return 'truetype'
  if (normalized === 'otf') return 'opentype'
  return normalized
}

const splitFontFaceArgs = (value: string): string[] => {
  const args: string[] = []
  let current = ''
  let inString = false
  let quote = ''
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index]
    if (inString) {
      current += char
      if (char === quote && value[index - 1] !== '\\') {
        inString = false
        quote = ''
      }
      continue
    }
    if (char === '"' || char === "'") {
      inString = true
      quote = char
      current += char
      continue
    }
    if (char === ',') {
      args.push(current.trim())
      current = ''
      continue
    }
    current += char
  }
  const trimmed = current.trim()
  if (trimmed.length > 0) args.push(trimmed)
  return args
}

const unquote = (value: string): string => value.replace(/^["']|["']$/g, '')

const parseFontFaceIncludes = (scss: string): FontFaceEntry[] => {
  const entries: FontFaceEntry[] = []
  const regex = /@include font-face\(([\s\S]*?)\);/g
  let match = regex.exec(scss)
  while (match) {
    const args = splitFontFaceArgs(match[1] ?? '')
    if (args.length >= 3) {
      const family = unquote(args[0] ?? '')
      const filename = unquote(args[1] ?? '')
      const format = unquote(args[2] ?? '')
      const style = unquote(args[3] ?? 'normal')
      const weight = unquote(args[4] ?? 'normal')
      const pathArg = unquote(args[5] ?? '/fonts/')
      const unicodeRange = args[6] ? unquote(args[6]) : undefined
      if (family && filename && format) {
        entries.push({ family, filename, format, style, weight, path: pathArg, unicodeRange })
      }
    }
    match = regex.exec(scss)
  }
  return entries
}

const normalizeFontPath = (value: string): string => {
  let normalized = value.trim()
  if (!normalized.startsWith('/')) normalized = `/${normalized}`
  if (!normalized.endsWith('/')) normalized = `${normalized}/`
  return normalized
}

export const buildFontFaceCss = (
  entries: FontFaceEntry[],
  baseUrl: string | undefined,
  families: Set<string>,
): string => {
  const base = baseUrl ? (baseUrl.startsWith('http') ? baseUrl : `https://${baseUrl}`) : ''
  return entries
    .filter(entry => families.has(entry.family))
    .map(entry => {
      const pathPrefix = normalizeFontPath(entry.path)
      const url = base ? `${base}${pathPrefix}${entry.filename}` : `${pathPrefix}${entry.filename}`
      const format = normalizeFontFormat(entry.format)
      const range = entry.unicodeRange ? `unicode-range:${entry.unicodeRange};` : ''
      return `@font-face{font-family:"${entry.family}";font-style:${entry.style};font-weight:${entry.weight};font-display:swap;${range}src:url("${url}") format("${format}")}`
    })
    .join('')
}

export const buildFontStack = (fontName: string, fallback: string): string =>
  `"${fontName}", ${fallback}`

export const loadFontFaceEntries = async (): Promise<FontFaceEntry[]> => {
  const fontsPath = path.join(QUARTZ, 'styles', 'fonts.scss')
  try {
    const scss = await fs.readFile(fontsPath, 'utf8')
    return parseFontFaceIncludes(scss)
  } catch {
    return []
  }
}
