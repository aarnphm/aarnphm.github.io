import { h } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug, resolveRelative } from '../../path'
import { BaseView, PropertyConfig } from '../types'
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
  renderPropertyValueNodes,
  resolveCardImage,
  type RenderElement,
} from './property-presentation'

export function buildCards(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const isKanban = view.type === 'kanban'
  if (isKanban && !view.groupBy) {
    return h('p.base-kanban-empty', 'Choose a groupBy property to organize this view into columns.')
  }
  const imageField = view.image || (isKanban ? undefined : 'image')
  const order = view.order ?? []
  const titleField = isKanban ? (order[0] ?? 'file.name') : undefined

  const renderCard = (file: QuartzPluginData): RenderElement => {
    const slug = (file.slug || '') as FullSlug
    const titleValue = titleField
      ? resolveValueWithFormulas(file, titleField, getContext, getPropertyExpr)
      : undefined
    const title =
      listValueToPlainText(titleValue) ?? getFileDisplayName(file) ?? fallbackNameFromSlug(slug)
    const href = resolveRelative(currentSlug, slug)

    const imageValue = imageField
      ? resolveValueWithFormulas(file, imageField, getContext, getPropertyExpr)
      : undefined
    const image = resolveCardImage(imageValue, currentSlug)

    const metadataItems: RenderElement[] = []
    const metadataFields = (isKanban ? order.slice(1) : order).filter(
      (field): field is string =>
        typeof field === 'string' &&
        (isKanban || (field !== 'title' && field !== 'file.title' && field !== 'note.title')) &&
        field !== imageField,
    )

    for (const field of metadataFields) {
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (!hasRenderableValue(value)) continue
      const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (renderedValue.length === 0) continue
      const label = getPropertyDisplayName(field, properties)
      metadataItems.push(
        h('div.base-card-meta-item', [
          h('span.base-card-meta-label', label),
          h('span.base-card-meta-value', renderedValue),
        ]),
      )
    }

    const cardChildren: RenderElement[] = []
    if (image) {
      cardChildren.push(
        h('a.base-card-image-link', { href, 'data-slug': slug, 'aria-label': title }, [
          h('img.base-card-image', {
            src: image.url,
            'data-bookcase-cover': image.slug,
            'data-full-cover': image.slug ? image.url : undefined,
            alt: '',
            loading: 'lazy',
            decoding: 'async',
          }),
        ]),
      )
    }

    const contentChildren: RenderElement[] = [
      h('a.base-card-title-link', { href, 'data-slug': slug }, [h('h3.base-card-title', title)]),
    ]
    if (metadataItems.length > 0) {
      contentChildren.push(h('div.base-card-meta', metadataItems))
    }

    cardChildren.push(h('div.base-card-content', contentChildren))

    return h(isKanban ? 'li.base-card' : 'div.base-card', { 'data-base-slug': slug }, cardChildren)
  }

  const styleParts: string[] = []
  if (typeof view.cardSize === 'number' && view.cardSize > 0) {
    styleParts.push(`--base-card-min: ${view.cardSize}px;`)
  }
  // Obsidian stores height / width; CSS aspect-ratio uses width / height.
  if (view.imageAspectRatio !== undefined) {
    styleParts.push(`--base-card-aspect: 1 / ${view.imageAspectRatio};`)
  } else if (typeof view.cardAspect === 'number' && view.cardAspect > 0) {
    styleParts.push(`--base-card-aspect: ${view.cardAspect};`)
  }
  if (view.imageFit) styleParts.push(`--base-card-image-fit: ${view.imageFit};`)
  if (isKanban) styleParts.push(`--base-kanban-width: ${view.columnWidth ?? 280}px;`)
  const varStyle = styleParts.length > 0 ? styleParts.join(' ') : undefined

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)
    const groupElements: RenderElement[] = []

    const groupSizes = view.groupSizes
    const groupAspects = view.groupAspects

    for (const [groupName, groupFiles] of groups) {
      const cards = groupFiles.map(file => renderCard(file))
      const label = groupLabel(groupName)
      if (isKanban) {
        groupElements.push(
          h('section.base-kanban-column', { 'aria-label': groupName === null ? 'None' : label }, [
            h('div.base-kanban-column-header', [
              h('h3.base-kanban-column-title', groupName === null ? 'None' : label),
              h(
                'span.base-kanban-count',
                { 'aria-label': `${cards.length} results` },
                String(cards.length),
              ),
            ]),
            h('ul.base-kanban-cards', cards),
            h('p.base-kanban-column-empty', { hidden: cards.length > 0 }, 'No results'),
          ]),
        )
        continue
      }
      const parts: string[] = []
      const size = groupSizes?.[label]
      if (typeof size === 'number' && size > 0) {
        parts.push(`--base-card-min: ${size}px;`)
      }
      const aspect = groupAspects?.[label]
      if (typeof aspect === 'number' && aspect > 0) {
        parts.push(`--base-card-aspect: ${aspect};`)
      }
      const gridStyle = parts.length > 0 ? parts.join(' ') : undefined

      groupElements.push(
        h('div.base-card-group', [
          h('h3.base-card-group-header', label),
          h('div.base-card-grid', gridStyle ? { style: gridStyle } : {}, cards),
        ]),
      )
    }

    if (isKanban) {
      return groupElements.length > 0
        ? h(
            'div.base-kanban',
            { style: varStyle, tabindex: 0, role: 'region', 'aria-label': `${view.name} board` },
            groupElements,
          )
        : h('p.base-kanban-empty', 'No results in the visible groups.')
    }

    return h('div.base-card-container', varStyle ? { style: varStyle } : {}, groupElements)
  }

  const cards = files.map(file => renderCard(file))
  return h('div.base-card-grid', varStyle ? { style: varStyle } : {}, cards)
}
