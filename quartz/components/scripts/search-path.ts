import { tokenizeTerm } from '../../util/search-text'

const pending = new Map<HTMLElement, string>()
const fitted = new WeakMap<HTMLElement, string>()
let frame: number | undefined
let context: CanvasRenderingContext2D | null | undefined

function fitPath(
  path: string,
  width: number,
  pattern: RegExp | undefined,
  font: string,
  boldFont: string,
  letterSpacing: number,
  canvas: CanvasRenderingContext2D,
): string {
  const measure = (value: string) => {
    const parts = pattern ? value.split(pattern) : [value]
    let measured = 0
    for (const [index, part] of parts.entries()) {
      canvas.font = index % 2 === 0 ? font : boldFont
      measured += canvas.measureText(part).width
    }
    return measured + Math.max(0, Array.from(value).length - 1) * letterSpacing
  }
  if (measure(path) <= width) return path

  const characters = Array.from(path)
  const filenameLength = Array.from(path.slice(path.lastIndexOf('/') + 1)).length
  const shorten = (keep: number) => {
    const tail = Math.min(keep - 1, Math.max(Math.ceil(keep / 2), filenameLength))
    return `${characters.slice(0, keep - tail).join('')}…${characters.slice(-tail).join('')}`
  }

  let low = 2
  let high = characters.length - 1
  let value = '…'
  while (low <= high) {
    const keep = Math.floor((low + high) / 2)
    const candidate = shorten(keep)
    if (measure(candidate) <= width) {
      value = candidate
      low = keep + 1
    } else {
      high = keep - 1
    }
  }
  return value
}

function pathPattern(term: string): RegExp | undefined {
  const terms = tokenizeTerm(term).map(value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  return terms.length ? new RegExp(`(${terms.join('|')})`, 'gi') : undefined
}

function flushPaths(): void {
  frame = undefined
  const batch = [...pending]
  pending.clear()
  context ??= document.createElement('canvas').getContext('2d')
  const canvas = context
  if (!canvas) return

  // Read every label before changing any: DOM measurement inside the binary search
  // forces a full search/preview layout for each candidate and blocks typing.
  const measurements = batch.flatMap(([element, term]) => {
    if (!element.isConnected) return []
    const path = element.title
    const width = element.clientWidth
    if (!path || width === 0) return []
    const style = getComputedStyle(element)
    const font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
    const boldFont = `${style.fontStyle} 700 ${style.fontSize} ${style.fontFamily}`
    const letterSpacing = Number.parseFloat(style.letterSpacing) || 0
    const key = JSON.stringify([
      path,
      term,
      width,
      font,
      letterSpacing,
      document.fonts.status,
      document.fonts.check(font),
      document.fonts.check(boldFont),
    ])
    if (fitted.get(element) === key) return []
    return [{ element, path, width, font, boldFont, letterSpacing, key, term }]
  })

  for (const { element, path, width, font, boldFont, letterSpacing, key, term } of measurements) {
    const pattern = pathPattern(term)
    const value = fitPath(path, width - 1, pattern, font, boldFont, letterSpacing, canvas)
    const parts = pattern ? value.split(pattern) : [value]
    element.replaceChildren(
      ...parts.map((part, index) => {
        if (index % 2 === 0) return document.createTextNode(part)
        const match = document.createElement('span')
        match.className = 'highlight'
        match.textContent = part
        return match
      }),
    )
    fitted.set(element, key)
  }
}

export function fitSearchPath(element: HTMLElement, term: string): void {
  pending.set(element, term)
  frame ??= requestAnimationFrame(flushPaths)
}
