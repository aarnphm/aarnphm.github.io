import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug, resolveRelative } from '../../path'
import { BaseView, PropertyConfig, VAULT_PROPERTY_DEFAULTS } from '../types'
import { buildVault } from '../vault/render'
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

export function buildVaultView(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  type Key = (typeof VAULT_PROPERTY_DEFAULTS)[number][0]
  const get = (file: QuartzPluginData, key: Key) => {
    const field = view[key] ?? VAULT_PROPERTY_DEFAULTS.find(([name]) => name === key)![1]
    return resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
  }
  const text = (file: QuartzPluginData, key: Key) => listValueToPlainText(get(file, key))
  const num = (file: QuartzPluginData, key: Key) => {
    const value = text(file, key)
    return value !== undefined && Number.isFinite(Number(value)) ? Number(value) : undefined
  }
  const grouped: [string | null, QuartzPluginData[]][] = view.groupBy
    ? [...groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)].map(
        ([value, members]) => [groupLabel(value), members],
      )
    : [[null, files]]
  const entries = grouped.flatMap(([group, members]) =>
    members.map(file => {
      const slug = (file.slug || '') as FullSlug
      const cover = resolveCardImage(get(file, 'image'), currentSlug)
      const description = file.frontmatter?.description
      return {
        slug,
        path: String(file.relativePath ?? file.filePath ?? slug),
        href: resolveRelative(currentSlug, slug),
        title: text(file, 'labelTitle') ?? getFileDisplayName(file) ?? fallbackNameFromSlug(slug),
        directors: [get(file, 'labelDirector')]
          .flat()
          .map(listValueToPlainText)
          .filter((name): name is string => Boolean(name)),
        year: num(file, 'labelYear'),
        rating: num(file, 'labelRating'),
        status: text(file, 'labelStatus'),
        studio: text(file, 'labelStudio'),
        watched: text(file, 'labelWatched'),
        runtime: num(file, 'labelRuntime'),
        description:
          typeof description === 'string' ? plainTextDescription(description, slug) : undefined,
        cover: cover?.url,
        coverSlug: cover?.slug,
        group,
        rows: (view.order ?? []).flatMap(key => {
          const value = resolveValueWithFormulas(file, key, getContext, getPropertyExpr)
          if (!hasRenderableValue(value)) return []
          const label = getPropertyDisplayName(key, properties)
          const rendered = renderPropertyValueNodes(value, currentSlug, allFiles)
          return [{ key: key.replace(/^note\./, ''), label, value: rendered }]
        }),
      }
    }),
  )
  return buildVault(entries, view)
}
