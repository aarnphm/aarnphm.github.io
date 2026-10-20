import { isRecord } from '../type-guards'
import {
  SummaryDefinition,
  ViewSummaryConfig,
  PropertyConfig,
  BuiltinSummaryType,
  BUILTIN_SUMMARY_TYPES,
} from './compiler/schema'

export type { SummaryDefinition, ViewSummaryConfig, PropertyConfig, BuiltinSummaryType }
export { BUILTIN_SUMMARY_TYPES }

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const BASE_VIEW_TYPES = [
  'table',
  'list',
  'gallery',
  'board',
  'calendar',
  'card',
  'cards',
  'map',
  'kanban',
  'bookcase',
  'vault',
] as const

type BaseViewType = (typeof BASE_VIEW_TYPES)[number]

const isBaseViewType = (value: unknown): value is BaseViewType =>
  BASE_VIEW_TYPES.some(type => type === value)

export type BaseFileFilter =
  | string
  | { and: BaseFileFilter[] }
  | { or: BaseFileFilter[] }
  | { not: BaseFileFilter[] }

export interface BaseFile {
  filters?: BaseFileFilter
  views: BaseView[]
  properties?: Record<string, PropertyConfig>
  summaries?: Record<string, string>
  formulas?: Record<string, string>
}

export interface BaseView {
  type: BaseViewType
  name: string
  order?: string[]
  sort?: BaseSortConfig[]
  columnSize?: Record<string, number>
  groupBy?: string | BaseGroupBy
  groupOrder?: BaseGroupValue[]
  limit?: number
  filters?: BaseFileFilter
  summaries?: Record<string, string> | ViewSummaryConfig
  image?: string
  cardSize?: number
  cardAspect?: number
  columnWidth?: number
  imageFit?: 'cover' | 'contain'
  imageAspectRatio?: number
  nestedProperties?: boolean
  indentProperties?: boolean
  separator?: string
  date?: string
  dateField?: string
  dateProperty?: string
  coordinates?: string
  markerIcon?: string
  markerColor?: string
  defaultZoom?: number
  defaultCenter?: [number, number]
  clustering?: boolean
  groupSizes?: Record<string, number>
  groupAspects?: Record<string, number>
  spineTitle?: string
  spineAuthor?: string
  spineColor?: string
  bookHeight?: number
  shelfWidth?: number
  /** Shelf levels in one bay (default 5); more shelves fill bays to the right. */
  shelfRows?: number
  scrollStep?: number
  pageContent?: boolean
  /** Bookcase style: straight (default), corner, spine, wall, zigzag or ladder. */
  shelfStyle?: string
  /** Main material; the style keeps its per-part exceptions. Unset keeps the style's own. */
  shelfMaterial?: string
  /**
   * Per-topic cabinet and tab name: `{ style, material, displayNames, showTags }`. A key is a groupBy value or
   * a tag; a tag key adds a tab for the books that carry it.
   */
  topicShelves?: Record<
    string,
    { style?: string; material?: string; displayNames?: string; showTags?: boolean }
  >
  /** False hides the label plates on every shelf; a topicShelves entry may override it. */
  showTags?: boolean
  /** Property that groups the shelves inside one topic (default `subcategory`). */
  topicGroupBy?: string
  labelTitle?: string
  labelDirector?: string
  labelYear?: string
  labelRating?: string
  labelStatus?: string
  labelStudio?: string
  labelWatched?: string
  labelRuntime?: string
  /** Vault rack: `rack-steel` (default) or `black-lacquer`. */
  rackMaterial?: string
  /** Vault shelf length in world units (600 to 2400); shelves per bay is `shelfRows`. */
  bayWidth?: number
  /** Cans between dividers; 0 for none. Section changes always get one. */
  dividers?: number
  /** Year cuts between tin, hammertone and polypropylene cans. */
  eraCuts?: number[]
  /** False leaves the end panel and handwheel off the vault rack. */
  crank?: boolean
}

/** Bookcase view keys that name a property, with the property used when the key is absent. */
export const BOOKCASE_PROPERTY_DEFAULTS = [
  ['image', 'note.posters'],
  ['spineTitle', 'note.title'],
  ['spineAuthor', 'note.author'],
  ['spineColor', undefined],
] as const satisfies readonly (readonly [keyof BaseView, string | undefined])[]

/** Vault view keys that name a property, with the property used when the key is absent. */
export const VAULT_PROPERTY_DEFAULTS = [
  ['image', 'note.posters'],
  ['labelTitle', 'note.title'],
  ['labelDirector', 'note.director'],
  ['labelYear', 'note.year'],
  ['labelRating', 'note.rating'],
  ['labelStatus', 'note.status'],
  ['labelStudio', 'note.studio'],
  ['labelWatched', 'note.watched'],
  ['labelRuntime', 'note.runtime'],
] as const satisfies readonly (readonly [keyof BaseView, string])[]

export interface BaseSortConfig {
  property: string
  direction: 'ASC' | 'DESC'
}

export interface BaseGroupBy {
  property: string
  direction: 'ASC' | 'DESC'
}

export type BaseGroupValue = string | number | boolean | null

const isBaseGroupValue = (value: unknown): value is BaseGroupValue =>
  value === null ||
  typeof value === 'string' ||
  typeof value === 'boolean' ||
  (typeof value === 'number' && Number.isFinite(value))

export function parseViews(raw: unknown[]): BaseView[] {
  return raw.map((entry, index) => {
    if (!isRecord(entry)) throw new Error(`View at index ${index} must be an object`)
    const { type, name } = entry
    if (!isNonEmptyString(type) || !isNonEmptyString(name)) {
      throw new Error(`View at index ${index} must have non-empty 'type' and 'name' fields`)
    }
    if (!isBaseViewType(type)) {
      throw new Error(`View at index ${index} has unsupported type '${type}'`)
    }
    const groupOrder = entry.groupOrder
    if (
      groupOrder !== undefined &&
      (!Array.isArray(groupOrder) || !groupOrder.every(isBaseGroupValue))
    ) {
      throw new Error(`View at index ${index} must have scalar values in 'groupOrder'`)
    }
    for (const option of ['columnWidth', 'imageAspectRatio']) {
      const value = entry[option]
      if (
        value !== undefined &&
        (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)
      ) {
        throw new Error(`View at index ${index} must have a positive number for '${option}'`)
      }
    }
    if (
      entry.imageFit !== undefined &&
      entry.imageFit !== 'cover' &&
      entry.imageFit !== 'contain'
    ) {
      throw new Error(`View at index ${index} must have 'cover' or 'contain' for 'imageFit'`)
    }
    return { ...entry, type, name, groupOrder }
  })
}

export function parseViewSummaries(
  viewSummaries: Record<string, string> | ViewSummaryConfig | undefined,
  topLevelSummaries?: Record<string, string>,
): ViewSummaryConfig | undefined {
  if (!viewSummaries || typeof viewSummaries !== 'object') return undefined

  if ('columns' in viewSummaries && typeof viewSummaries.columns === 'object') {
    return viewSummaries as ViewSummaryConfig
  }

  const columns: Record<string, SummaryDefinition> = {}

  for (const [column, summaryValue] of Object.entries(viewSummaries)) {
    if (typeof summaryValue !== 'string') continue

    const normalized = summaryValue.toLowerCase().trim()

    if (BUILTIN_SUMMARY_TYPES.includes(normalized as BuiltinSummaryType)) {
      columns[column] = { type: 'builtin', builtinType: normalized as BuiltinSummaryType }
      continue
    }

    if (topLevelSummaries && summaryValue in topLevelSummaries) {
      columns[column] = {
        type: 'formula',
        formulaRef: summaryValue,
        expression: topLevelSummaries[summaryValue],
      }
      continue
    }

    if (summaryValue.includes('(') || summaryValue.includes('.')) {
      columns[column] = { type: 'formula', expression: summaryValue }
    }
  }

  return Object.keys(columns).length > 0 ? { columns } : undefined
}
