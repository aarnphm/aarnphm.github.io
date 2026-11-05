import type { Element } from 'hast'
import { h } from 'hastscript'
import type { ReferenceDependency } from './dependencies'
import { interpretPdfFragment } from './target'

function positiveNumber(value: unknown): string | undefined {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return String(value)
  if (typeof value !== 'string' || !/^\d+(?:\.\d+)?%?$/.test(value.trim())) return
  const number = Number.parseFloat(value)
  return number > 0 ? String(value.trim().endsWith('%') ? number / 100 : number) : undefined
}

function positiveInteger(value: unknown): string | undefined {
  const number = positiveNumber(value)
  if (!number) return
  const integer = Number.parseInt(number, 10)
  return integer > 0 ? String(integer) : undefined
}

function pdfFit(value: unknown): string | undefined {
  if (typeof value !== 'string') return
  const fit = value.trim().toLowerCase()
  if (fit === 'width' || fit === 'fit-width') return 'width'
  if (fit === 'page' || fit === 'fit-page') return 'page'
  if (['actual', 'native', '100%'].includes(fit)) return 'actual'
}

export function pdfReferenceElement(record: ReferenceDependency, url: string): Element {
  const settings = record.resolvedSettings?.kind === 'object' ? record.resolvedSettings.value : {}
  const fragment = interpretPdfFragment(record.fragments.join('#'))
  const page = positiveInteger(settings.page) ?? fragment.page
  const height = positiveInteger(settings.height) ?? fragment.height
  const fit = pdfFit(settings.fit ?? settings.mode)
  const scale = positiveNumber(settings.scale)
  const overscan = positiveInteger(settings.overscan)
  const parts = (record.label ?? record.implicitLabel ?? '').split('|').map(part => part.trim())
  if (/^\d+(?:x\d+)?$/.test(parts.at(-1) ?? '')) parts.pop()
  const title =
    parts.join('|').trim() || record.target.split('/').filter(Boolean).at(-1) || 'document.pdf'
  return h(
    'div.internal-embed.pdf-embed',
    {
      'data-pdf-src': url,
      'data-pdf-title': title,
      tabindex: 0,
      ...(page ? { 'data-pdf-page': page } : {}),
      ...(height ? { 'data-pdf-height': height } : {}),
      ...(fit ? { 'data-pdf-fit': fit } : {}),
      ...(scale ? { 'data-pdf-scale': scale } : {}),
      ...(overscan ? { 'data-pdf-overscan': overscan } : {}),
    },
    [h('span.pdf-embed-loading', 'Loading PDF')],
  )
}

export function imageReferenceElement(record: ReferenceDependency, url: string): Element {
  const parts = (record.label ?? record.implicitLabel ?? '').split('|').map(part => part.trim())
  const dimensions = /^(\d+)(?:x(\d+))?$/.exec(parts.at(-1) ?? '')
  if (dimensions) parts.pop()
  const style = record.resolvedSettings?.kind === 'css' ? record.resolvedSettings.raw : undefined
  return h('img', {
    src: url,
    width: dimensions?.[1] ?? 'auto',
    height: dimensions?.[2] ?? 'auto',
    alt: parts.join('|').trim(),
    ...(style ? { style } : {}),
  })
}
