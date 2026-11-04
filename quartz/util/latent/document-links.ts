import type { Element } from 'hast'
import { h } from 'hastscript'
import type { LatentDoc, LatentKind, LatentPreview } from './types'
import { titleRuns, titleText } from './title'

export interface LinkItem {
  preview: LatentPreview
  score: number
  doc: LatentDoc
  inScope: boolean
}

export interface DocumentLink {
  doc: LatentDoc
  matches: LinkItem[]
  best: LinkItem
}

/** Preserve the ranking of each document's strongest match and the exact matched rows. */
export function groupDocumentLinks(items: LinkItem[]): DocumentLink[] {
  const groups = new Map<number, DocumentLink>()
  for (const item of items) {
    const group = groups.get(item.doc.d)
    if (!group) groups.set(item.doc.d, { doc: item.doc, matches: [item], best: item })
    else {
      if (!group.matches.some(match => match.preview.r === item.preview.r)) group.matches.push(item)
      if (item.score > group.best.score) group.best = item
    }
  }
  return [...groups.values()].map(group => ({
    ...group,
    matches: group.matches.sort((a, b) => a.preview.r - b.preview.r),
  }))
}

export function chunkLink(preview: LatentPreview, doc: LatentDoc): Element {
  const number = preview.r - doc.r0 + 1
  const page = preview.s === 'pdf' && preview.p ? `, page ${preview.l || preview.p}` : ''
  return h(
    'a.lat-chunk-link',
    {
      href: `?focus=${preview.r}`,
      dataLatFocus: String(preview.r),
      dataLatChunk: String(preview.r),
      dataNoPopover: true,
      ariaLabel: `${titleText(doc.t)}, chunk ${number}${page}`,
    },
    String(number),
  )
}

export function documentLink(group: DocumentLink, focus: number): Element {
  const { doc, matches, best } = group
  return h(
    'li.lat-link.lat-link-document',
    {
      dataRow: String(best.preview.r),
      dataDoc: String(doc.d),
      dataOut: best.inScope ? undefined : '',
      dataSame: doc.d === focus ? '' : undefined,
    },
    [
      h(
        'a.lat-link-title',
        {
          href: `?focus=${best.preview.r}`,
          dataLatFocus: String(best.preview.r),
          dataLatChunk: String(best.preview.r),
          dataNoPopover: true,
        },
        titleRuns(doc.t),
      ),
      h(
        'span.lat-document-chunks',
        { ariaLabel: 'matching chunks' },
        matches.map(item => chunkLink(item.preview, doc)),
      ),
      h('span.lat-score', { title: 'strongest chunk cosine' }, (best.score / 255).toFixed(3)),
    ],
  )
}

export function linksPane(
  focus: LatentPreview,
  items: LinkItem[],
  kinds: LatentKind[],
  label = 'links',
): Element {
  const documents = groupDocumentLinks(items)
  return h('section.lat-links', { dataFocus: String(focus.r) }, [
    h('header.lat-pane-head', [
      h('span', label),
      h('span.lat-count', `${documents.length} documents · ${items.length} chunks`),
    ]),
    ...kinds.flatMap(kind => {
      const own = documents.filter(group => group.doc.s === kind)
      if (!own.length) return []
      return [
        h('h3.lat-group', [
          h('span.lat-mark', { dataKind: kind, title: kind }, kind),
          h('span', String(own.length)),
        ]),
        h(
          'ol.lat-list.lat-document-list',
          { dataKind: kind },
          own.map(group => documentLink(group, focus.d)),
        ),
      ]
    }),
  ])
}
