import { h, s } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug, resolveRelative } from '../../path'
import {
  BOOKCASE_DEFAULTS,
  BOOKCASE_LIMITS,
  clampOption,
  fallbackSpine,
  packShelves,
  spineInk,
} from '../bookcase/geometry'
import { bookcaseScene, type SceneBook } from '../bookcase/scene'
import {
  BOOKCASE_STYLES,
  hasTopicTag,
  placeShelves,
  SHELF_MATERIALS,
  stylePackOptions,
  topicKeys,
  topicLabel,
  topicShelf,
  topicShortcuts,
  type TopicShelf,
} from '../bookcase/styles'
import { BaseView, BOOKCASE_PROPERTY_DEFAULTS, PropertyConfig } from '../types'
import {
  groupFiles,
  groupLabel,
  resolveValueWithFormulas,
  type EvalContextFactory,
  type PropertyExprGetter,
} from './property-evaluation'
import {
  fallbackNameFromSlug,
  getFileDisplayName,
  getPropertyDisplayName,
  hasRenderableValue,
  listValueToPlainText,
  plainTextDescription,
  renderPropertyValueNodes,
  resolveCardImage,
  type RenderElement,
} from './property-presentation'

const SPINE_HEX = /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i

function expandHex(hex: string): string {
  return hex.length === 4 ? `#${Array.from(hex.slice(1), c => c + c).join('')}` : hex
}

// Spines carry the surname, as a shelf does; the index keeps the full credit.
function spineSurname(author: string): string {
  const first = author.split(/,|&| and /)[0].trim()
  return first.split(/\s+/).pop() ?? first
}

export function buildBookcase(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const bookHeight = clampOption(
    view.bookHeight,
    BOOKCASE_DEFAULTS.bookHeight,
    BOOKCASE_LIMITS.bookHeight,
  )
  const shelfWidth = clampOption(
    view.shelfWidth,
    BOOKCASE_DEFAULTS.shelfWidth,
    BOOKCASE_LIMITS.shelfWidth,
  )
  const shelfRows = clampOption(
    view.shelfRows,
    BOOKCASE_DEFAULTS.shelfRows,
    BOOKCASE_LIMITS.shelfRows,
  )
  const scrollStep = clampOption(
    view.scrollStep,
    BOOKCASE_DEFAULTS.scrollStep,
    BOOKCASE_LIMITS.scrollStep,
  )
  const [imageField, titleField, authorField, colorField] = BOOKCASE_PROPERTY_DEFAULTS.map(
    ([key, fallback]) => {
      const value = view[key]
      return typeof value === 'string' ? value : fallback
    },
  )
  const pageContent = view.pageContent !== false
  // `author` in `order` and `note.author` in `spineAuthor` name the same property.
  const bare = (field: string | undefined) => field?.replace(/^note\./, '')
  const shown = new Set([titleField, authorField, imageField].map(bare))
  const order = (view.order ?? []).filter(field => !shown.has(bare(field)))
  const resolve = (file: QuartzPluginData, field: string | undefined) =>
    field ? resolveValueWithFormulas(file, field, getContext, getPropertyExpr) : undefined

  const grouped: [string | null, QuartzPluginData[]][] = view.groupBy
    ? [...groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)].map(
        ([value, members]) => [value === null ? 'none' : groupLabel(value), members],
      )
    : [[null, files]]
  const groups = grouped.filter(([, members]) => members.length > 0)

  type Entry = SceneBook & { file: QuartzPluginData; fullTitle: string; byline: string }
  const toEntry = (file: QuartzPluginData): Entry => {
    const slug = (file.slug || '') as FullSlug
    const fullTitle =
      listValueToPlainText(resolve(file, titleField)) ??
      getFileDisplayName(file) ??
      fallbackNameFromSlug(slug)
    const author = listValueToPlainText(resolve(file, authorField))
    const cover = resolveCardImage(resolve(file, imageField), currentSlug)
    const explicit = listValueToPlainText(resolve(file, colorField))
    const path = String(file.relativePath ?? file.filePath ?? slug)
    const spine = explicit && SPINE_HEX.test(explicit) ? expandHex(explicit) : fallbackSpine(path)
    const lead = order[0] ? listValueToPlainText(resolve(file, order[0])) : undefined
    return {
      file,
      slug,
      fullTitle,
      title: fullTitle.split(':')[0].trim(),
      author: author ? spineSurname(author) : undefined,
      byline: [author, lead].filter(Boolean).join(' · '),
      spine,
      ink: spineInk(spine),
      cover: cover?.url,
      coverSlug: cover?.slug,
      sampleSpine: !(explicit && SPINE_HEX.test(explicit)),
    }
  }

  const entryGroups = groups.map(([topic, members]) => ({
    topic,
    label: topic === null ? null : topicLabel(topic, view.topicShelves),
    entries: members.map(toEntry),
  }))
  const style = BOOKCASE_STYLES.find(name => name === view.shelfStyle) ?? 'straight'
  const material = SHELF_MATERIALS.find(name => name === view.shelfMaterial)
  const count = entryGroups.reduce((n, group) => n + group.entries.length, 0)
  // A topic is a group, or a topicShelves key that names a tag. It holds each book once.
  const topicGroupBy = typeof view.topicGroupBy === 'string' ? view.topicGroupBy : 'subcategory'
  const declared = topicKeys(view.topicShelves)
  const members = new Map<string, Entry[]>()
  const memberSlugs = new Map<string, Set<string>>()
  const join = (topic: string, entry: Entry) => {
    let seen = memberSlugs.get(topic)
    if (!seen) {
      seen = new Set()
      memberSlugs.set(topic, seen)
    }
    if (seen.has(entry.slug)) return
    seen.add(entry.slug)
    const list = members.get(topic)
    if (list) list.push(entry)
    else members.set(topic, [entry])
  }
  for (const { topic, entries } of entryGroups)
    for (const entry of entries) {
      if (topic !== null) join(topic, entry)
      for (const key of declared)
        if (key !== topic && hasTopicTag(entry.file.frontmatter?.tags, key)) join(key, entry)
    }
  // Topics with one book or fewer share an "other" tab.
  for (const [topic, list] of Array.from(members))
    if (list.length <= 1) {
      members.delete(topic)
      memberSlugs.delete(topic)
      for (const entry of list) join('other', entry)
    }
  const topicsOf = new Map<Entry, string[]>()
  for (const [topic, list] of members)
    for (const entry of list) {
      const existing = topicsOf.get(entry)
      if (existing) existing.push(topic)
      else topicsOf.set(entry, [topic])
    }
  const topicShelves: Record<string, TopicShelf> = {}
  for (const topic of members.keys()) {
    const shelf = topicShelf(topic, view.topicShelves)
    if (shelf) topicShelves[topic] = shelf
  }
  // Collapsed tabs show a shortcut; the chosen tab opens to its name, as the triathlon map tabs do.
  const tabs = [
    { key: '', label: 'all', books: count },
    ...[...members]
      .filter(([topic]) => topic !== 'other')
      .sort(([, a], [, b]) => b.length - a.length)
      .map(([topic, list]) => ({
        key: topic,
        label: topicLabel(topic, view.topicShelves),
        books: list.length,
      })),
    ...(members.has('other')
      ? [{ key: 'other', label: 'other', books: members.get('other')!.length }]
      : []),
  ]
  const shortcuts = topicShortcuts(tabs.map(tab => tab.label))
  shortcuts[0] = 'ALL'
  const topicBar =
    members.size > 1
      ? h(
          'div.bookcase-topics',
          { role: 'tablist', 'aria-label': 'Topics', 'aria-orientation': 'vertical' },
          tabs.map(({ key, label, books }, i) => {
            const name = `${label} ${books}`
            return h(
              'button.bookcase-topic',
              {
                type: 'button',
                role: 'tab',
                'data-topic': key,
                'data-shortcut': shortcuts[i][0].toLowerCase(),
                'aria-selected': String(i === 0),
                'aria-label': `${label}, ${books} books`,
                tabindex: i === 0 ? 0 : -1,
                style: `--bc-tab-short:${shortcuts[i].length}ch;--bc-tab-label:${name.length}ch`,
              },
              [
                h('span.bookcase-topic-shortcut', { 'aria-hidden': 'true' }, shortcuts[i]),
                h('span.bookcase-topic-label', { 'aria-hidden': 'true' }, name),
              ],
            )
          }),
        )
      : null
  const layout = placeShelves(
    packShelves(
      entryGroups.map(({ label, entries }) => ({
        label,
        books: entries.map(entry => ({
          path: String(entry.file.relativePath ?? entry.file.filePath ?? entry.slug),
          item: entry,
        })),
      })),
      stylePackOptions(style, { bookHeight, shelfWidth }, count, shelfRows),
    ),
    style,
    material,
    { showTags: view.showTags !== false, rows: shelfRows },
  )
  const total = layout.books.length

  const detailOf = (entry: Entry): RenderElement[] => {
    const rows: RenderElement[] = []
    for (const field of order) {
      const value = resolve(entry.file, field)
      if (!hasRenderableValue(value)) continue
      const rendered = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (rendered.length === 0) continue
      rows.push(
        h('div.bookcase-prop', [
          h('dt', getPropertyDisplayName(field, properties)),
          h('dd', rendered),
        ]),
      )
    }
    const nodes: RenderElement[] = []
    if (rows.length > 0) nodes.push(h('dl.bookcase-props', rows))
    const description = entry.file.frontmatter?.description
    if (pageContent && typeof description === 'string' && description.trim()) {
      nodes.push(
        h('p.bookcase-description', plainTextDescription(description, entry.slug as FullSlug)),
      )
    }
    return nodes
  }

  let index = 0
  const indexGroups = entryGroups.map(({ label, entries }) => {
    const rows = entries.map(entry => {
      const href = resolveRelative(currentSlug, entry.slug as FullSlug)
      return h(
        'li.bookcase-entry',
        {
          'data-base-slug': entry.slug,
          'data-book-index': index++,
          'data-cover': entry.cover,
          'data-bookcase-cover': entry.coverSlug,
          // A book with several values sits on the shelf of its first one.
          'data-topic-shelf': listValueToPlainText([resolve(entry.file, topicGroupBy)].flat()[0]),
          'data-topics': topicsOf.has(entry) ? JSON.stringify(topicsOf.get(entry)) : undefined,
        },
        [
          h('a.bookcase-entry-link', { href, 'data-slug': entry.slug }, [
            h('span.bookcase-entry-title', entry.fullTitle),
            entry.byline ? h('span.bookcase-entry-byline', entry.byline) : '',
          ]),
          h('div.bookcase-entry-detail', detailOf(entry)),
        ],
      )
    })
    const list = h('ul.base-list.bookcase-index-list', rows)
    return label === null
      ? list
      : h('details.base-list-group.bookcase-index-group', { open: true }, [
          h('summary.base-list-group-header', [
            h('span.bookcase-index-label', label),
            h('span.bookcase-index-count', String(entries.length)),
          ]),
          list,
        ])
  })

  const controlIcon = (paths: string[]) =>
    s(
      'svg',
      { viewBox: '0 0 24 24', 'aria-hidden': 'true' },
      paths.map(d => s('path', { d })),
    )
  return h(
    'div.bookcase',
    {
      style: `--bookcase-step: ${scrollStep}px;`,
      'data-scroll-step': scrollStep,
      'data-book-height': bookHeight,
      'data-shelf-width': shelfWidth,
      'data-shelf-rows': shelfRows,
      'data-shelf-style': style,
      'data-shelf-material': material,
      'data-show-tags': view.showTags === false ? 'false' : undefined,
      'data-topic-shelves': topicBar ? JSON.stringify(topicShelves) : undefined,
    },
    [
      h('div.bookcase-stage', [
        h('div.bookcase-plate', [
          h('span.bookcase-corner', { 'data-corner': 'title', 'data-bookcase-shelf': '' }, ''),
          h(
            'div.bookcase-viewport',
            {
              tabindex: 0,
              role: 'group',
              'aria-label':
                'Bookcase drawing. Arrow keys pan, plus and minus zoom, Home fits all books, Enter previews the selected book.',
            },
            [
              h('div.bookcase-world', [
                bookcaseScene(
                  layout,
                  `${view.name}: ${total} books on ${layout.shelves.length} shelves`,
                ),
              ]),
            ],
          ),
          h('div.bookcase-camera-controls', { role: 'group', 'aria-label': 'Bookcase controls' }, [
            h(
              'button',
              { type: 'button', 'data-camera': 'in', 'aria-label': 'Zoom in', title: 'Zoom in' },
              controlIcon(['M12 5v14M5 12h14']),
            ),
            h(
              'button',
              { type: 'button', 'data-camera': 'out', 'aria-label': 'Zoom out', title: 'Zoom out' },
              controlIcon(['M5 12h14']),
            ),
            h(
              'button',
              {
                type: 'button',
                'data-camera': 'fit',
                'aria-label': 'Fit bookcase',
                title: 'Fit bookcase',
              },
              controlIcon(['M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5']),
            ),
            ...(topicBar ? [topicBar] : []),
          ]),
          h('p.bookcase-no-results', 'no books match this search'),
          h(
            'span.bookcase-corner',
            { 'data-corner': 'hint' },
            'drag to explore · click to preview',
          ),
          h(
            'span.bookcase-corner',
            { 'data-corner': 'readout', 'data-bookcase-readout': '' },
            `– / ${total}`,
          ),
        ]),
      ]),
      h('div.bookcase-index', indexGroups),
      h('dialog.bookcase-preview', { 'aria-label': 'Book preview' }, [
        h('div.bookcase-preview-toolbar', [
          h('span.bookcase-preview-position'),
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'close',
              'aria-label': 'Close preview',
              title: 'Close preview',
              autofocus: true,
            },
            controlIcon(['M6 6l12 12M18 6 6 18']),
          ),
        ]),
        h('div.bookcase-preview-content', [
          h('div.bookcase-preview-art', { 'aria-hidden': 'true' }),
          h('div.bookcase-preview-body', [
            h('h2', [h('a.bookcase-preview-title.bookcase-preview-link')]),
            h('p.bookcase-preview-byline'),
            h('div.bookcase-preview-detail'),
          ]),
        ]),
        h('div.bookcase-preview-navigation', [
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'previous',
              'aria-label': 'Previous book',
              title: 'Previous book',
            },
            controlIcon(['M15 5l-7 7 7 7']),
          ),
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'next',
              'aria-label': 'Next book',
              title: 'Next book',
            },
            controlIcon(['M9 5l7 7-7 7']),
          ),
        ]),
      ]),
    ],
  )
}
