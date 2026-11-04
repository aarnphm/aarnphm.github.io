import type { Element } from 'hast'
import { h, s } from 'hastscript'
import { extractArxivId } from '../../plugins/stores/citations'

/**
 * The site's pdf viewer inside the latent reader. A pdf passage carries a page glyph in its
 * locus gutter: the page drawn small with the passage's box on it. A click opens the folio, the
 * page itself in the site's pdf embed (pdf.inline.ts) under the passage with that box marked
 * (latent-client/folio.ts). An arXiv link in a passage gets the note pages' arXiv preview.
 */

// a type, not an interface: hastscript's properties want the index signature only aliases get
type LinkProperties = {
  href: string
  className: string[]
  rel?: string
  target?: string
  dataArxivId?: string
}

/** Link properties for a passage link; an arXiv address previews its pdf as on note pages. */
export function linkProperties(href: string): LinkProperties {
  if (href.startsWith('/') || href.startsWith('#')) return { href, className: ['internal'] }
  const arxiv = extractArxivId(href)
  // links.ts `handleArxiv`: the popover treats an internal link with an arXiv id as its pdf
  if (arxiv)
    return {
      href,
      className: ['internal', 'lat-arxiv'],
      dataArxivId: arxiv,
      rel: 'noopener noreferrer',
      target: '_blank',
    }
  return { href, className: ['external'], rel: 'noopener noreferrer', target: '_blank' }
}

/** The pdf file a document id names, as the transclude plugin points an embed at it. */
export function pdfSource(id: string): string {
  return encodeURI(`/${id.replace(/^\/+/, '')}`)
}

// a page glyph of 10 × 13 units: a letter page inside a 0.5 unit margin
const GLYPH_W = 9
const GLYPH_H = 12

/**
 * The gutter button that opens a passage's page. Its glyph is the page with the passage's box
 * (permille of the page) filled, so the button already says where on the page the text sits.
 */
export function folioToggle(
  src: string,
  page: number,
  box: [number, number, number, number] | undefined,
  label: string,
): Element {
  const band = box
    ? (() => {
        const [x0, y0, x1, y1] = box
        const x = 0.5 + (x0 / 1000) * GLYPH_W
        const y = 0.5 + (y0 / 1000) * GLYPH_H
        const w = Math.max(1.5, ((x1 - x0) / 1000) * GLYPH_W)
        const hgt = Math.max(0.9, ((y1 - y0) / 1000) * GLYPH_H)
        return s('rect.lat-folio-band', {
          x: x.toFixed(2),
          y: y.toFixed(2),
          width: w.toFixed(2),
          height: Math.min(hgt, 12.5 - y).toFixed(2),
        })
      })()
    : null
  return h(
    'button.lat-folio-toggle',
    {
      type: 'button',
      dataFolio: src,
      dataFolioPage: String(page),
      ariaExpanded: 'false',
      ariaLabel: `show ${label} of the pdf`,
      title: `${label}: the page, this passage marked`,
    },
    [
      s('svg', { viewBox: '0 0 10 13', ariaHidden: 'true', focusable: 'false' }, [
        s('rect.lat-folio-sheet', { x: '0.5', y: '0.5', width: String(GLYPH_W), height: '12' }),
        band,
      ]),
    ],
  )
}
