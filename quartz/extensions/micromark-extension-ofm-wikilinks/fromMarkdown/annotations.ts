import type { Wikilink, WikilinkData } from '../fromMarkdown'
import { slugAnchor } from '../../../util/path'

/**
 * annotate regular link with hast properties.
 * converts `[[target]]`, `[[target|alias]]`, `[[target#anchor]]` to <a> elements.
 */
export function annotateRegularLink(
  node: Wikilink,
  wikilink: WikilinkData,
  url: string,
  targetSlug: string,
): void {
  const { alias, target, metadataParsed, metadata, anchorSegments, anchorIsBlockRef } = wikilink

  const fp = target.trim()
  let displayText = alias ?? fp

  if (!node.data) node.data = { wikilink }

  const nested = !anchorIsBlockRef && anchorSegments !== undefined && anchorSegments.length >= 2

  node.data.hName = 'a'
  node.data.hProperties = {
    href: url,
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
    ...(nested
      ? {
          'data-anchor-path': JSON.stringify(anchorSegments.map(s => slugAnchor(s.trim()))),
          ...(targetSlug ? { 'data-anchor-target': targetSlug } : {}),
        }
      : {}),
  }
  node.data.hChildren = [{ type: 'text', value: displayText }]
}

/**
 * annotate image embed with hast properties.
 * converts `![[image.png]]`, `![[image.png|alt]]`, `![[image.png|100x200]]` to <img> elements.
 * supports figure/figcaption: `![[image.png|caption text|300x200]]`.
 */
export function annotateImageEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { alias, metadataParsed, metadata } = wikilink

  const style = metadata?.slice(1, -1).trim()

  const parts = (alias ?? '').split('|').map(s => s.trim())
  let caption = ''
  let width = 'auto'
  let height = 'auto'

  if (parts.length > 0) {
    const lastPart = parts[parts.length - 1]
    const dimMatch = /^(\d+)(?:x(\d+))?$/.exec(lastPart)

    if (dimMatch) {
      width = dimMatch[1]
      height = dimMatch[2] ?? 'auto'
      caption = parts.slice(0, -1).join('|').trim()
    } else {
      caption = parts.join('|').trim()
    }
  }

  if (!node.data) node.data = { wikilink }

  // backward compatible
  node.data.hName = 'img'
  node.data.hProperties = {
    src: url,
    width,
    height,
    alt: caption,
    ...(style ? { style } : {}),
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }
}

/**
 * annotate video embed with hast properties.
 * converts `![[video.mp4]]` to <video> elements with controls.
 */
export function annotateVideoEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { metadataParsed, metadata } = wikilink

  if (!node.data) node.data = { wikilink }

  node.data.hName = 'video'
  node.data.hProperties = {
    src: url,
    controls: true,
    loop: true,
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }
}

export function annotateAudioEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { alias, metadataParsed, metadata } = wikilink
  const caption = alias?.trim()

  if (!node.data) node.data = { wikilink }

  node.data.hName = 'audio'
  node.data.hProperties = {
    src: url,
    controls: true,
    ...(caption ? { 'data-caption': caption } : {}),
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }
}

/**
 * annotate external URL embed with hast properties.
 * converts `![[https://...]]` to an <img> marked `external-embed`;
 * the ofm.ts html plugin rewrites it to a sized <iframe>.
 */
export function annotateExternalEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { alias, metadataParsed, metadata } = wikilink

  let width: string | undefined
  let height: string | undefined
  const parts = (alias ?? '').split('|').map(s => s.trim())
  const dimMatch = /^(\d+%?)(?:x(\d+%?))?$/.exec(parts[parts.length - 1] ?? '')
  if (dimMatch) {
    width = dimMatch[1]
    height = dimMatch[2]
  }

  if (!node.data) node.data = { wikilink }

  node.data.hName = 'img'
  node.data.hProperties = {
    src: url,
    className: ['external-embed'],
    ...(width ? { width } : {}),
    ...(height ? { height } : {}),
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }
}

function pdfMetadataValue(metadata: Record<string, unknown> | undefined, key: string): unknown {
  return metadata ? metadata[key] : undefined
}

function readPdfNumber(value: unknown): string | undefined {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return String(value)
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (!/^\d+(?:\.\d+)?%?$/.test(trimmed)) return undefined
  return trimmed.endsWith('%') ? String(Number.parseFloat(trimmed) / 100) : trimmed
}

function readPdfInteger(value: unknown): string | undefined {
  const token = readPdfNumber(value)
  if (!token) return undefined
  const parsed = Number.parseInt(token, 10)
  return Number.isInteger(parsed) && parsed > 0 ? String(parsed) : undefined
}

function readPdfFit(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim().toLowerCase()
  if (normalized === 'width' || normalized === 'fit-width') return 'width'
  if (normalized === 'page' || normalized === 'fit-page') return 'page'
  if (normalized === 'actual' || normalized === 'native' || normalized === '100%') return 'actual'
  return undefined
}

function readPdfTitle(wikilink: WikilinkData): string {
  const alias = wikilink.alias?.trim()
  if (alias) {
    const parts = alias.split('|').map(part => part.trim())
    const titleParts = /^\d+(?:x\d+)?$/.test(parts[parts.length - 1] ?? '')
      ? parts.slice(0, -1)
      : parts
    const title = titleParts.join('|').trim()
    if (title) return title
  }

  const segments = wikilink.target.split('/').filter(Boolean)
  return segments[segments.length - 1] || wikilink.target || 'document.pdf'
}

export function pdfPageFromAnchor(wikilink: WikilinkData): string | undefined {
  const anchor = wikilink.anchorText?.trim() ?? ''
  const match = /^page=(\d+)$/i.exec(anchor)
  return match ? match[1] : undefined
}

export function annotatePdfEmbed(node: Wikilink, wikilink: WikilinkData, url: string): void {
  const { metadataParsed, metadata } = wikilink
  const page =
    readPdfInteger(pdfMetadataValue(metadataParsed, 'page')) ?? pdfPageFromAnchor(wikilink)
  const fit = readPdfFit(
    pdfMetadataValue(metadataParsed, 'fit') ?? pdfMetadataValue(metadataParsed, 'mode'),
  )
  const scale = readPdfNumber(pdfMetadataValue(metadataParsed, 'scale'))
  const height = readPdfInteger(pdfMetadataValue(metadataParsed, 'height'))
  const overscan = readPdfInteger(pdfMetadataValue(metadataParsed, 'overscan'))

  if (!node.data) node.data = { wikilink }

  node.data.hName = 'div'
  node.data.hProperties = {
    class: 'internal-embed pdf-embed',
    'data-pdf-src': url,
    'data-pdf-title': readPdfTitle(wikilink),
    tabindex: 0,
    ...(page ? { 'data-pdf-page': page } : {}),
    ...(fit ? { 'data-pdf-fit': fit } : {}),
    ...(scale ? { 'data-pdf-scale': scale } : {}),
    ...(height ? { 'data-pdf-height': height } : {}),
    ...(overscan ? { 'data-pdf-overscan': overscan } : {}),
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }
  node.data.hChildren = [
    {
      type: 'element',
      tagName: 'span',
      properties: { class: 'pdf-embed-loading' },
      children: [{ type: 'text', value: 'Loading PDF' }],
    },
  ]
}

/**
 * annotate block transclude with hast properties.
 * converts `![[page#heading]]`, `![[page#^block]]` to <blockquote> with nested <a>.
 */
export function annotateTransclude(
  node: Wikilink,
  wikilink: WikilinkData,
  url: string,
  displayAnchor: string,
): void {
  const { alias, target, metadataParsed, metadata, anchorSegments, anchorIsBlockRef } = wikilink
  const block = displayAnchor
  const nested = !anchorIsBlockRef && anchorSegments !== undefined && anchorSegments.length >= 2

  if (!node.data) node.data = { wikilink }

  node.data.hName = 'blockquote'
  node.data.hProperties = {
    class: 'transclude',
    transclude: true,
    'data-url': url,
    'data-block': block,
    'data-embed-alias': alias ?? '',
    ...(nested
      ? {
          'data-anchor-path': JSON.stringify(
            anchorSegments.map(segment => {
              const value = segment.trim()
              return target.trim() === 'triathlon' &&
                (value.startsWith('filter=') ||
                  value.startsWith('settings=') ||
                  value.startsWith('display='))
                ? value
                : slugAnchor(value)
            }),
          ),
        }
      : {}),
    ...(metadataParsed
      ? { 'data-metadata': JSON.stringify(metadataParsed) }
      : metadata
        ? { 'data-metadata': metadata }
        : {}),
  }

  node.data.hChildren = [
    {
      type: 'element',
      tagName: 'a',
      properties: { href: url + displayAnchor, class: 'transclude-inner' },
      children: [{ type: 'text', value: `Transclude of ${url} ${block}` }],
    },
  ]
}
