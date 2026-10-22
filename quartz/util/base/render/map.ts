import { h } from 'hastscript'
import type { MarkerData } from '../../../components/scripts/base-map-data'
import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug } from '../../path'
import { BaseView, PropertyConfig } from '../types'
import {
  resolveValueWithFormulas,
  type EvalContextFactory,
  type PropertyExprGetter,
} from './property-evaluation'
import {
  fallbackNameFromSlug,
  getFileDisplayName,
  hasRenderableValue,
  isIconToken,
  plainTextDescription,
  renderIconHtml,
  type RenderElement,
} from './property-presentation'

export function buildMap(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const resolveMapProperty = (file: QuartzPluginData, prop: string | undefined): unknown => {
    if (!prop) return undefined
    const key = prop.trim()
    if (!key) return undefined
    if (getPropertyExpr(key)) {
      return resolveValueWithFormulas(file, key, getContext, getPropertyExpr)
    }
    if (key.startsWith('note.')) {
      const stripped = key.replace(/^note\./, '')
      if (getPropertyExpr(stripped)) {
        return resolveValueWithFormulas(file, stripped, getContext, getPropertyExpr)
      }
    } else {
      const prefixed = `note.${key}`
      if (getPropertyExpr(prefixed)) {
        return resolveValueWithFormulas(file, prefixed, getContext, getPropertyExpr)
      }
    }
    return undefined
  }

  const coordinatesProp = view.coordinates || 'coordinates'

  const markers: MarkerData[] = []

  for (const file of files) {
    const coordsValue = resolveMapProperty(file, coordinatesProp)
    if (!coordsValue || !Array.isArray(coordsValue) || coordsValue.length < 2) {
      continue
    }

    const lat = parseFloat(String(coordsValue[0]))
    const lon = parseFloat(String(coordsValue[1]))

    if (isNaN(lat) || isNaN(lon)) {
      continue
    }

    const title = getFileDisplayName(file) ?? fallbackNameFromSlug((file.slug || '') as FullSlug)
    const slug = (file.slug || '') as FullSlug

    const popupFields: Record<string, unknown> = {}
    const order = Array.isArray(view.order) ? view.order : []
    for (const field of order) {
      if (typeof field !== 'string') continue
      if (field === 'title' || field === 'file.title' || field === 'note.title') continue
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (hasRenderableValue(value)) {
        popupFields[field] = value
      }
    }

    const icon = resolveMapProperty(file, view.markerIcon)
    const color = resolveMapProperty(file, view.markerColor)
    const iconHtml = isIconToken(icon)
      ? renderIconHtml(icon.value)
      : icon
        ? String(icon)
        : undefined

    markers.push({
      lat,
      lon,
      title,
      slug,
      icon: iconHtml,
      color: color ? String(color) : undefined,
      description:
        typeof file.frontmatter?.description === 'string'
          ? plainTextDescription(file.frontmatter.description, slug)
          : typeof file.description === 'string'
            ? file.description
            : undefined,
      address: typeof file.frontmatter?.address === 'string' ? file.frontmatter.address : undefined,
      categories: Array.isArray(file.frontmatter?.type)
        ? file.frontmatter.type.filter((value): value is string => typeof value === 'string')
        : typeof file.frontmatter?.type === 'string'
          ? [file.frontmatter.type]
          : [],
      rating: typeof file.frontmatter?.rating === 'number' ? file.frontmatter.rating : undefined,
      url: typeof file.frontmatter?.url === 'string' ? file.frontmatter.url : undefined,
      popupFields,
    })
  }

  const config = {
    defaultZoom: view.defaultZoom ?? 12,
    defaultCenter: view.defaultCenter,
    clustering: view.clustering !== false,
  }

  const attrs: Record<string, string> = {
    'data-markers': JSON.stringify(markers),
    'data-config': JSON.stringify(config),
    'data-current-slug': currentSlug,
  }

  if (properties) {
    attrs['data-properties'] = JSON.stringify(properties)
  }

  return h('div.base-map', attrs)
}
