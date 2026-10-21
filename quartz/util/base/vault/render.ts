import type { Element, ElementContent } from 'hast'
import { h, s } from 'hastscript'
import { packVault } from './layout'
import { type LabelKind, type SceneCan, vaultScene } from './scene'

/** One placed copy of a film, already resolved from the Base by `../render.ts`. */
export interface VaultEntry {
  slug: string
  /** Path that seeds the can's size jitter. */
  path: string
  href: string
  title: string
  directors: string[]
  year?: number
  rating?: number
  status?: string
  studio?: string
  watched?: string
  runtime?: number
  description?: string
  /** Page-relative poster URL; the cover emitter swaps in a thumbnail through `coverSlug`. */
  cover?: string
  coverSlug?: string
  /** groupBy label, or null when the view has no groupBy. */
  group: string | null
  /** The view's `order` properties, rendered. */
  rows: { key: string; label: string; value: (ElementContent | string)[] }[]
}

export interface VaultView {
  name: string
  bayWidth?: unknown
  shelfRows?: unknown
  dividers?: unknown
  eraCuts?: unknown
  rackMaterial?: unknown
  showTags?: unknown
  crank?: unknown
}

/** Fields the vault card shows in its own rows; `order` adds the rest. */
const CARD_KEYS = new Set([
  'title',
  'director',
  'year',
  'rating',
  'status',
  'studio',
  'watched',
  'runtime',
  'posters',
  'description',
])

const KIND: Record<string, LabelKind> = { recurring: 'tape', finished: 'viewed' }

const surname = (name: string) => name.trim().split(/\s+/).pop() ?? name

const decade = (year?: number) =>
  year === undefined ? 'undated' : `${Math.floor(year / 10) * 10}s`

const controlIcon = (paths: string[]) =>
  s(
    'svg',
    { viewBox: '0 0 24 24', 'aria-hidden': 'true' },
    paths.map(d => s('path', { d })),
  )

const button = (attrs: Record<string, string>, label: string, paths: string[]) =>
  h('button', { type: 'button', ...attrs, 'aria-label': label, title: label }, controlIcon(paths))

export function buildVault(entries: VaultEntry[], view: VaultView): Element {
  const material = view.rackMaterial === 'black-lacquer' ? 'black-lacquer' : 'rack-steel'
  const showTags = view.showTags !== false
  const crank = view.crank !== false
  const sections = entries.map(entry => entry.group ?? decade(entry.year))
  const layout = packVault(
    entries.map((entry, i) => ({
      path: entry.path,
      year: entry.year,
      section: sections[i],
      item: {
        title: entry.title,
        surname: entry.directors.map(surname).join(' & '),
        year: entry.year,
        rating: entry.rating,
        kind: KIND[entry.status ?? ''] ?? 'vinyl',
        watched: entry.watched,
        studio: entry.studio,
      } satisfies SceneCan,
    })),
    view,
  )
  const total = entries.length
  const scene = vaultScene(layout, {
    material,
    showTags,
    crank,
    label: `${view.name}: ${total} film cans on ${layout.shelves} shelves in ${layout.bays} bays`,
  })

  const detail = (entry: VaultEntry, index: number): Element[] => {
    const can = layout.cans[index]
    const label = [
      entry.title,
      can.item.surname && `dir. ${can.item.surname}`,
      entry.year,
      '35 mm',
      `${can.footage} ft`,
    ]
      .filter(Boolean)
      .join(' · ')
      .toUpperCase()
    const prop = (key: string, value: (ElementContent | string)[] | string) =>
      h('div.vault-prop', [h('dt', key), h('dd', typeof value === 'string' ? value : value)])
    const reels =
      entry.runtime !== undefined
        ? ` · ${Math.ceil(entry.runtime / (can.footage === 2000 ? 22 : 11))} reels (est.)`
        : ''
    const rows = [
      prop(
        entry.directors.length > 1 ? 'directors' : 'director',
        entry.directors.join(', ') || '–',
      ),
      prop('studio', entry.studio ?? '–'),
      ...entry.rows.filter(row => !CARD_KEYS.has(row.key)).map(row => prop(row.label, row.value)),
      prop('rating', entry.rating === undefined ? '–' : `${entry.rating} / 10`),
      prop('status', entry.status ?? '–'),
      prop('watched', entry.watched ?? '–'),
      prop('runtime', entry.runtime === undefined ? '–' : `${entry.runtime} min${reels}`),
    ]
    return [
      h('p.vault-label-line', label),
      h('dl.vault-props', rows),
      ...(entry.description ? [h('p.vault-description', entry.description)] : []),
    ]
  }

  const groups: { label: string; rows: Element[] }[] = []
  entries.forEach((entry, index) => {
    if (groups[groups.length - 1]?.label !== sections[index])
      groups.push({ label: sections[index], rows: [] })
    groups[groups.length - 1].rows.push(
      h(
        'li.vault-entry',
        {
          'data-index': index,
          'data-base-slug': entry.slug,
          'data-href': entry.href,
          'data-cover': entry.cover,
          'data-full-cover': entry.cover,
          'data-bookcase-cover': entry.coverSlug,
        },
        [
          h('button.vault-entry-select', { type: 'button' }, [
            h('span.vault-entry-title', entry.title),
            h(
              'span.vault-entry-byline',
              [entry.directors.join(', '), entry.year].filter(Boolean).join(' · '),
            ),
          ]),
          h('div.vault-entry-detail', detail(entry, index)),
        ],
      ),
    )
  })

  return h('div.vault', { 'data-total': total }, [
    h('div.vault-stage', [
      h(
        'div.vault-viewport',
        {
          tabindex: 0,
          role: 'group',
          'aria-label':
            'Film vault drawing. Arrow keys move along the rack, Enter opens the selected can, Escape returns it, plus and minus zoom.',
        },
        [h('template.vault-source', [scene])],
      ),
      h('div.vault-controls', { role: 'group', 'aria-label': 'Vault camera' }, [
        button({ 'data-camera': 'in' }, 'Zoom in', ['M12 5v14M5 12h14']),
        button({ 'data-camera': 'out' }, 'Zoom out', ['M5 12h14']),
        button({ 'data-camera': 'fit' }, 'Fit rack', ['M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5']),
        h('span.vault-status', { role: 'status', 'data-state': 'loading' }, [
          h('span.vault-status-spinner', controlIcon(['M12 3a9 9 0 1 1-9 9'])),
          h('span.vault-status-ready', controlIcon(['M5 12l4 4L19 6'])),
          h('span.vault-status-text.visually-hidden', 'preparing vault'),
        ]),
      ]),
      h('span.vault-corner', { 'data-corner': 'title', 'data-vault-section': '' }, ''),
      h('span.vault-corner', { 'data-corner': 'hint' }, 'drag a can out or in · drag to pan'),
      h(
        'span.vault-corner',
        { 'data-corner': 'readout', 'data-vault-readout': '' },
        `– / ${total}`,
      ),
    ]),
    h(
      'div.vault-index',
      { 'aria-label': 'Film index' },
      groups.map(({ label, rows }) =>
        h('details.base-list-group.vault-index-group', { open: true }, [
          h('summary.base-list-group-header', [
            h('span.vault-index-label', label),
            h('span.vault-index-count', String(rows.length)),
          ]),
          h('ul.base-list.vault-index-list', rows),
        ]),
      ),
    ),
    h('dialog.vault-card', { 'aria-label': 'Vault card' }, [
      h('div.vault-card-toolbar', [
        h('span.vault-card-position'),
        button({ 'data-card': 'close', autofocus: '' }, 'Close card', ['M6 6l12 12M18 6 6 18']),
      ]),
      h('div.vault-card-content', [
        h('figure.vault-card-poster', [h('img', { alt: '', decoding: 'async' })]),
        h('div.vault-card-record', [
          h('h2', [h('a.vault-card-title')]),
          h('div.vault-card-detail'),
        ]),
      ]),
      h('div.vault-card-navigation', [
        button({ 'data-card': 'previous' }, 'Previous can', ['M15 5l-7 7 7 7']),
        button({ 'data-card': 'next' }, 'Next can', ['M9 5l7 7-7 7']),
      ]),
    ]),
  ])
}
