import type { Element, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { visit } from 'unist-util-visit'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { FullSlug } from '../path'
import type { RenderedBaseView } from './render'
import {
  DOCUMENT_PREVIEW_KIND,
  previewLabel,
  type BasePreview,
  type BasePreviewSample,
} from '../document-preview'

const CARD_SAMPLE = 8
const TABLE_COLUMNS = 4
const TABLE_ROWS = 7
const LIST_ITEMS = 8

const classes = (node: Element): string[] => {
  const value = node.properties?.className
  return Array.isArray(value) ? value.map(String) : []
}

const hasClass = (node: Element, name: string) => classes(node).includes(name)

function findAll(root: Root | Element, test: (node: Element) => boolean, limit: number): Element[] {
  const found: Element[] = []
  visit(root, 'element', node => {
    if (found.length >= limit) return false
    if (test(node)) found.push(node)
  })
  return found
}

const find = (root: Root | Element, test: (node: Element) => boolean): Element | undefined =>
  findAll(root, test, 1)[0]

const text = (node: Element | undefined): string => (node ? previewLabel(toString(node)) : '')

/** Card images are relative to the view page; the preview resolves them to site paths. */
function cardImage(card: Element, viewSlug: FullSlug): string | null {
  const link = find(card, node => hasClass(node, 'base-card-image-link'))
  const img = link ? find(link, node => node.tagName === 'img') : undefined
  const style = link?.properties?.style
  const raw =
    typeof img?.properties?.src === 'string'
      ? img.properties.src
      : typeof style === 'string'
        ? /url\((['"]?)([^'")]+)\1\)/.exec(style)?.[2]
        : undefined
  if (!raw) return null
  const url = new URL(raw, `https://garden.invalid/${viewSlug}`)
  return url.origin === 'https://garden.invalid' ? `${url.pathname}${url.search}` : url.toString()
}

function sampleView(rendered: RenderedBaseView): BasePreviewSample | null {
  const { tree, view, slug } = rendered
  switch (view.type) {
    case 'card':
    case 'cards':
    case 'gallery':
    case 'board': {
      const cards = findAll(tree, node => hasClass(node, 'base-card'), CARD_SAMPLE)
      return cards.length > 0
        ? {
            type: 'cards',
            items: cards.map(card => ({
              title: text(find(card, node => hasClass(node, 'base-card-title'))),
              image: cardImage(card, slug),
            })),
          }
        : null
    }
    case 'table': {
      const table = find(tree, node => hasClass(node, 'base-table'))
      if (!table) return null
      const head = find(table, node => node.tagName === 'thead')
      const body = find(table, node => node.tagName === 'tbody')
      const columns = head
        ? findAll(head, node => node.tagName === 'th', TABLE_COLUMNS).map(text)
        : []
      const rows = body
        ? body.children
            .filter(
              (row): row is Element =>
                row.type === 'element' &&
                row.tagName === 'tr' &&
                !hasClass(row, 'base-group-header'),
            )
            .slice(0, TABLE_ROWS)
            .map(row =>
              row.children
                .filter((cell): cell is Element => cell.type === 'element' && cell.tagName === 'td')
                .slice(0, TABLE_COLUMNS)
                .map(text),
            )
        : []
      return columns.length > 0 || rows.length > 0 ? { type: 'table', columns, rows } : null
    }
    case 'list':
    case 'calendar': {
      const list = find(tree, node => hasClass(node, 'base-list'))
      const items = list
        ? list.children
            .filter((item): item is Element => item.type === 'element' && item.tagName === 'li')
            .slice(0, LIST_ITEMS)
            .map(item => text(find(item, node => node.tagName === 'a') ?? item))
        : []
      return items.length > 0 ? { type: 'list', items } : null
    }
    default:
      return null
  }
}

export function buildBasePreview(
  baseSlug: FullSlug,
  baseData: QuartzPluginData,
  views: readonly RenderedBaseView[],
): BasePreview {
  const frontmatterTitle = baseData.frontmatter?.title
  const title =
    typeof frontmatterTitle === 'string' && frontmatterTitle.length > 0
      ? frontmatterTitle
      : (baseSlug.split('/').pop() ?? baseSlug)
  const description =
    typeof baseData.description === 'string' && baseData.description.trim().length > 0
      ? previewLabel(baseData.description, 160)
      : null

  let sample: BasePreview['sample'] = null
  for (const rendered of views) {
    const found = sampleView(rendered)
    if (found) {
      sample = { ...found, view: rendered.view.name }
      break
    }
  }

  return {
    kind: DOCUMENT_PREVIEW_KIND,
    type: 'base',
    slug: baseSlug,
    title,
    description,
    views: views.map(rendered => ({
      name: rendered.view.name,
      type: rendered.view.type,
      count: rendered.resultCount,
    })),
    sample,
  }
}
