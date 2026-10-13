import katex from 'katex'
import type { Run } from './press-kit-text'
import { customMacros } from '../../cfg'

const EM = 100

export interface MathImage {
  image: HTMLImageElement
  width: number
  ascent: number
  descent: number
  colors: Map<string, HTMLCanvasElement>
}

export type MathImages = Map<string, MathImage>

export const mathKey = (run: Run) => `${run.display ? 'display' : 'inline'}:${run.text}`

let stylesheet: Promise<{ css: string; url: string }> | undefined
const fontData = new Map<string, Promise<string>>()
const images = new Map<string, Promise<MathImage>>()

function katexStyles() {
  stylesheet ??= (async () => {
    const link = document.querySelector<HTMLLinkElement>('link[href*="katex"][href$=".css"]')
    if (!link) throw new Error('KaTeX stylesheet is unavailable')
    const response = await fetch(link.href, { signal: AbortSignal.timeout(15_000) })
    if (!response.ok) throw new Error('Unable to load KaTeX styles')
    return { css: await response.text(), url: link.href }
  })().catch(error => {
    stylesheet = undefined
    throw error
  })
  return stylesheet
}

function embeddedFont(url: string): Promise<string> {
  let pending = fontData.get(url)
  if (!pending) {
    pending = (async () => {
      const response = await fetch(url, { signal: AbortSignal.timeout(15_000) })
      if (!response.ok) throw new Error('Unable to load a KaTeX font')
      const blob = await response.blob()
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          if (typeof reader.result === 'string') resolve(reader.result)
          else reject(new Error('Unable to embed a KaTeX font'))
        }
        reader.onerror = () => reject(reader.error)
        reader.readAsDataURL(blob)
      })
    })().catch(error => {
      fontData.delete(url)
      throw error
    })
    fontData.set(url, pending)
  }
  return pending
}

async function mathImage(run: Run, pixelsPerEm: number): Promise<MathImage> {
  const { css, url } = await katexStyles()
  const host = document.createElement('div')
  host.setAttribute('aria-hidden', 'true')
  host.style.cssText = `position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none;white-space:nowrap;width:max-content;font-size:${EM}px;line-height:normal;color:#000;`
  const formula = document.createElement('span')
  katex.render(run.text, formula, {
    output: 'html',
    displayMode: !!run.display,
    throwOnError: false,
    strict: 'ignore',
    trust: false,
    macros: { ...customMacros },
    maxSize: 20,
    maxExpand: 1000,
  })
  // Display math uses KaTeX's larger operators, while its position is owned by the card.
  const display = formula.querySelector<HTMLElement>('.katex-display')
  if (display) display.style.cssText = 'display:inline;margin:0;text-align:left;'
  const katexRoot = formula.querySelector<HTMLElement>('.katex')
  if (katexRoot) katexRoot.style.cssText = `font-size:${EM}px;display:inline-block;text-align:left;`
  const marker = document.createElement('span')
  marker.style.cssText = 'display:inline-block;width:0;height:0;'
  host.append(formula, marker)
  document.body.append(host)
  try {
    const usedFonts = new Set<string>()
    const loads = new Map<string, string>()
    for (const element of formula.querySelectorAll<HTMLElement>('span')) {
      const style = getComputedStyle(element)
      const family = style.fontFamily.split(',')[0].replace(/["']/g, '').trim()
      if (!family.startsWith('KaTeX_')) continue
      usedFonts.add(family)
      const font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
      loads.set(font, (loads.get(font) ?? '') + (element.textContent ?? ''))
    }
    await Promise.all(Array.from(loads, ([font, text]) => document.fonts.load(font, text)))

    const faces = Array.from(css.matchAll(/@font-face\s*\{[^}]*\}/g), match => match[0])
    const embedded = await Promise.all(
      faces
        .filter(face => {
          const family = face
            .match(/font-family:([^;}]*)/)?.[1]
            .replace(/["']/g, '')
            .trim()
          return family !== undefined && usedFonts.has(family)
        })
        .map(async face => {
          const source = face.match(/url\(([^)]+\.woff2)\)/)?.[1].replace(/["']/g, '')
          if (!source) throw new Error('KaTeX font source is unavailable')
          const data = await embeddedFont(new URL(source, url).href)
          return face.replace(/src:[^;}]*/, `src:url("${data}") format("woff2")`)
        }),
    )
    const rect = host.getBoundingClientRect()
    const baseline = marker.getBoundingClientRect().top - rect.top
    const width = Math.ceil(rect.width) + 4
    const height = Math.ceil(rect.height) + 4
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('width', String(Math.ceil((width * pixelsPerEm) / EM)))
    svg.setAttribute('height', String(Math.ceil((height * pixelsPerEm) / EM)))
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
    const foreign = document.createElementNS(svg.namespaceURI, 'foreignObject')
    foreign.setAttribute('width', String(width))
    foreign.setAttribute('height', String(height))
    const content = document.createElement('div')
    content.style.cssText = `padding:2px;white-space:nowrap;width:max-content;font-size:${EM}px;line-height:normal;color:#000;`
    const style = document.createElement('style')
    style.textContent = css.replace(/@font-face\s*\{[^}]*\}/g, '') + embedded.join('')
    content.append(style, formula.cloneNode(true))
    foreign.append(content)
    svg.append(foreign)
    const image = new Image()
    // A self-contained data URL keeps foreignObject and its fonts exportable from canvas.
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(svg))}`
    await image.decode()
    return {
      image,
      width: width / EM,
      ascent: (baseline + 2) / EM,
      descent: (height - baseline - 2) / EM,
      colors: new Map(),
    }
  } finally {
    host.remove()
  }
}

export async function prepareMath(runs: Run[], pixelsPerEm: number): Promise<MathImages> {
  const keys = new Set<string>()
  const ready: MathImages = new Map()
  const pendingImages: Promise<void>[] = []
  for (const run of runs) {
    if (run.style !== 'math') continue
    const key = mathKey(run)
    if (keys.has(key)) continue
    keys.add(key)
    const cacheKey = `${pixelsPerEm}:${key}`
    let pending = images.get(cacheKey)
    if (!pending) {
      if (images.size >= 64) images.clear()
      pending = mathImage(run, pixelsPerEm).catch(error => {
        images.delete(cacheKey)
        throw error
      })
      images.set(cacheKey, pending)
    }
    pendingImages.push(
      pending.then(image => {
        ready.set(key, image)
      }),
    )
  }
  await Promise.all(pendingImages)
  return ready
}

export function drawMath(
  ctx: CanvasRenderingContext2D,
  math: MathImage,
  x: number,
  baseline: number,
  size: number,
  color: string,
) {
  let canvas = math.colors.get(color)
  if (!canvas) {
    canvas = document.createElement('canvas')
    canvas.width = math.image.naturalWidth
    canvas.height = math.image.naturalHeight
    const tint = canvas.getContext('2d')
    if (!tint) return
    tint.drawImage(math.image, 0, 0)
    tint.globalCompositeOperation = 'source-in'
    tint.fillStyle = color
    tint.fillRect(0, 0, canvas.width, canvas.height)
    math.colors.set(color, canvas)
  }
  ctx.drawImage(
    canvas,
    x,
    baseline - math.ascent * size,
    math.width * size,
    (math.ascent + math.descent) * size,
  )
}
