import type { Element, ElementContent, Properties, Root } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import fs from 'node:fs/promises'
import path from 'node:path'
import { SKIP, visit } from 'unist-util-visit'
import type { FilePath } from '../path'

const SVG_DATA_URI = 'data:image/svg+xml;base64,'
const localSvgImagePattern = /(<img\b[^>]*?\bsrc\s*=\s*)(?:"([^"]+?\.svg)"|'([^']+?\.svg)')/gi

// Course diagrams (graphviz, WaveDrom, drawio, nltk) paint with literal black and white. Mapping
// those to theme tokens keeps them legible in the dark theme; other colours carry meaning.
const paintTokens = new Map<string, string>([
  ['black', 'var(--dark)'],
  ['#000', 'var(--dark)'],
  ['#000000', 'var(--dark)'],
  ['rgb(0,0,0)', 'var(--dark)'],
  ['white', 'var(--light)'],
  ['#fff', 'var(--light)'],
  ['#ffffff', 'var(--light)'],
  ['rgb(255,255,255)', 'var(--light)'],
  ['#666', 'var(--darkgray)'],
  ['#666666', 'var(--darkgray)'],
  ['rgb(102,102,102)', 'var(--darkgray)'],
  ['#f5f5f5', 'var(--lightgray)'],
  ['rgb(245,245,245)', 'var(--lightgray)'],
])
const paintKeys = ['fill', 'stroke', 'color', 'stopColor'] as const
const stylePaintPattern = /(^|;)\s*(fill|stroke|color|stop-color)\s*:\s*([^;]+)/gi

function themedPaint(value: string): string | undefined {
  return paintTokens.get(value.replace(/\s+/g, '').toLowerCase())
}

// notebookToMarkdownChunks already rewrote local sources relative to the content root.
function contentSvgPath(raw: string): FilePath | undefined {
  if (raw.startsWith('/') || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(raw)) return undefined
  let decoded = raw
  try {
    decoded = decodeURI(raw)
  } catch {}
  const resolved = path.posix.normalize(decoded)
  return resolved.startsWith('../') ? undefined : (resolved as FilePath)
}

/** Rewrites local `<img src="….svg">` in notebook markdown to data URIs, so one hast pass inlines both kinds. */
export async function embedLocalSvgImages(
  chunks: readonly string[],
  directory: string,
): Promise<string[]> {
  const files = new Map<FilePath, Promise<string | undefined>>()
  const read = (fp: FilePath) => {
    let pending = files.get(fp)
    if (!pending) {
      pending = fs.readFile(path.join(directory, fp)).then(
        bytes => bytes.toString('base64'),
        () => undefined,
      )
      files.set(fp, pending)
    }
    return pending
  }
  return Promise.all(
    chunks.map(async chunk => {
      const matches = [...chunk.matchAll(localSvgImagePattern)]
      if (matches.length === 0) return chunk
      const encoded = await Promise.all(
        matches.map(match => {
          const fp = contentSvgPath(match[2] ?? match[3])
          return fp ? read(fp) : undefined
        }),
      )
      let index = 0
      return chunk.replace(localSvgImagePattern, (match, prefix: string) => {
        const base64 = encoded[index++]
        return base64 ? `${prefix}"${SVG_DATA_URI}${base64}"` : match
      })
    }),
  )
}

function parseSvg(base64: string): Element | undefined {
  let svg: Element | undefined
  const tree = fromHtml(Buffer.from(base64, 'base64').toString('utf8'), { fragment: true })
  visit(tree, 'element', node => {
    if (node.tagName !== 'svg') return
    svg = node
    return SKIP
  })
  return svg
}

function themeStyle(style: string): string {
  return style.replace(stylePaintPattern, (match, lead: string, key: string, value: string) => {
    const token = themedPaint(value)
    return token ? `${lead}${key}:${token}` : match
  })
}

function appendStyle(properties: Properties, declaration: string) {
  const style = typeof properties.style === 'string' ? properties.style.trim() : ''
  properties.style = style ? `${style.replace(/;$/, '')};${declaration}` : declaration
}

// Presentation attributes cannot hold var(), so themed paint moves into `style`.
function themeElement(node: Element) {
  const properties = (node.properties ??= {})
  for (const key of paintKeys) {
    const value = properties[key]
    if (typeof value !== 'string') continue
    const token = themedPaint(value)
    if (!token) continue
    delete properties[key]
    appendStyle(properties, `${key === 'stopColor' ? 'stop-color' : key}:${token}`)
  }
  if (typeof properties.style === 'string') properties.style = themeStyle(properties.style)
}

function themeSvg(svg: Element) {
  visit(svg, (node, index, parent) => {
    if (node.type === 'comment' && parent && index !== undefined) {
      parent.children.splice(index, 1)
      return [SKIP, index]
    }
    if (node.type !== 'element') return
    // drawio exports wrap the real drawing in a nested data-URI image.
    const href = node.properties?.href ?? node.properties?.xLinkHref
    if (node.tagName === 'image' && typeof href === 'string' && href.startsWith(SVG_DATA_URI)) {
      const nested = parseSvg(href.slice(SVG_DATA_URI.length))
      if (nested && parent && index !== undefined) {
        const { x, y, width, height } = node.properties
        nested.properties = { ...nested.properties, x, y, width, height }
        parent.children[index] = nested
        return index
      }
    }
    // graphviz paints a white page rectangle first; the notebook panel is the page.
    if (
      node.tagName === 'polygon' &&
      parent?.type === 'element' &&
      Array.isArray(parent.properties?.className) &&
      parent.properties.className.includes('graph') &&
      parent.children.find(child => child.type === 'element' && child.tagName === 'polygon') ===
        node &&
      typeof node.properties.fill === 'string' &&
      themedPaint(node.properties.fill) === 'var(--light)'
    ) {
      node.properties.fill = 'none'
    }
    themeElement(node)
  })
  // The SVG default paint is black; inherit ink from the page instead.
  const properties = (svg.properties ??= {})
  if (properties.fill === undefined && !/(^|;)\s*fill\s*:/.test(String(properties.style ?? ''))) {
    appendStyle(properties, 'fill:var(--dark)')
  }
}

function inlineSvgImage(img: Element): Element | undefined {
  const src = img.properties?.src
  if (typeof src !== 'string' || !src.startsWith(SVG_DATA_URI)) return undefined
  const svg = parseSvg(src.slice(SVG_DATA_URI.length))
  if (!svg) return undefined
  const { alt, style, width } = img.properties
  const wrapper: Properties = { className: ['svg-diagram'], role: 'img' }
  if (typeof alt === 'string' && alt) wrapper.ariaLabel = alt
  if (typeof style === 'string' && style) wrapper.style = style
  if (width !== undefined && width !== null) {
    const value = String(width)
    appendStyle(wrapper, `width:${/^\d+(\.\d+)?$/.test(value) ? `${value}px` : value}`)
  }
  if (typeof wrapper.style === 'string' && /(^|;)\s*width\s*:/.test(wrapper.style)) {
    // The wrapper owns the size; the viewBox keeps the aspect ratio.
    svg.properties.width = '100%'
    delete svg.properties.height
  }
  return { type: 'element', tagName: 'span', properties: wrapper, children: [svg] }
}

/** Inlines SVG images and themes SVG outputs in a notebook page, so diagrams follow the site theme and are not image-popup targets. */
export function themeNotebookSvgs(tree: Root): void {
  visit(tree, 'element', (node, index, parent) => {
    if (node.tagName === 'img' && parent && index !== undefined) {
      const inlined = inlineSvgImage(node)
      if (!inlined) return
      themeSvg(inlined.children[0] as Element)
      parent.children[index] = inlined as ElementContent
      return SKIP
    }
    // Only rendered outputs: toolbar and copy icons style their own paint.
    const className = node.properties?.className
    if (!Array.isArray(className) || !className.includes('notebook-output-svg')) return
    for (const child of node.children) {
      if (child.type === 'element' && child.tagName === 'svg') themeSvg(child)
    }
    return SKIP
  })
}
