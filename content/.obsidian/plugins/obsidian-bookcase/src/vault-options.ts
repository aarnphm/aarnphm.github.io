import type { BasesAllOptions, BasesOptions, BasesPropertyId, BasesViewConfig } from 'obsidian'
import { VAULT_DEFAULTS, VAULT_LIMITS, type VaultView } from './vault-model'

/** Keys that name a property, with the garden's default for each (`VAULT_PROPERTY_DEFAULTS`). */
export const VAULT_LABELS = {
  image: 'note.posters',
  labelTitle: 'note.title',
  labelDirector: 'note.director',
  labelYear: 'note.year',
  labelRating: 'note.rating',
  labelStatus: 'note.status',
  labelStudio: 'note.studio',
  labelWatched: 'note.watched',
  labelRuntime: 'note.runtime',
} as const

export type VaultLabel = keyof typeof VAULT_LABELS

export interface VaultConfig {
  props: Record<VaultLabel, BasesPropertyId>
  /** Options the garden's `buildVault` reads, as the `.base` file stores them. */
  view: VaultView
  pageContent: boolean
  editorPane: boolean
}

const property = (key: VaultLabel, displayName: string): BasesOptions => ({
  type: 'property',
  key,
  displayName,
  default: VAULT_LABELS[key],
})

const slider = (key: keyof typeof VAULT_LIMITS, displayName: string, step: number) => ({
  type: 'slider' as const,
  key,
  displayName,
  default: VAULT_DEFAULTS[key],
  min: VAULT_LIMITS[key][0],
  max: VAULT_LIMITS[key][1],
  step,
})

export const vaultOptions = (): BasesAllOptions[] => [
  property('image', 'Poster image'),
  {
    type: 'dropdown',
    key: 'rackMaterial',
    displayName: 'Rack',
    default: 'rack-steel',
    options: { 'rack-steel': 'Steel', 'black-lacquer': 'Black lacquer' },
  },
  {
    type: 'group',
    displayName: 'Label',
    items: [
      property('labelTitle', 'Title'),
      property('labelDirector', 'Director'),
      property('labelYear', 'Year'),
      property('labelRating', 'Rating'),
      property('labelStatus', 'Status'),
      property('labelStudio', 'Studio'),
      property('labelWatched', 'Watched'),
      property('labelRuntime', 'Runtime'),
    ],
  },
  {
    type: 'group',
    displayName: 'Rack',
    items: [
      slider('bayWidth', 'Shelf width', 40),
      slider('bayShelves', 'Shelves per bay', 1),
      slider('dividers', 'Cans between dividers', 1),
      { type: 'toggle', key: 'showTags', displayName: 'Card holders', default: true },
      { type: 'toggle', key: 'crank', displayName: 'End panel and handwheel', default: true },
      { type: 'toggle', key: 'pageContent', displayName: 'Show note body', default: true },
      {
        type: 'toggle',
        key: 'editorPane',
        displayName: 'Edit notes in a side pane',
        default: true,
      },
    ],
  },
]

export function readVaultConfig(config: BasesViewConfig, name: string): VaultConfig {
  const props = {} as Record<VaultLabel, BasesPropertyId>
  for (const [key, fallback] of Object.entries(VAULT_LABELS) as [VaultLabel, BasesPropertyId][])
    props[key] = config.getAsPropertyId(key) ?? fallback
  // `buildVault` clamps and validates every option itself.
  return {
    props,
    view: {
      name,
      bayWidth: config.get('bayWidth'),
      bayShelves: config.get('bayShelves'),
      dividers: config.get('dividers'),
      eraCuts: config.get('eraCuts'),
      rackMaterial: config.get('rackMaterial'),
      showTags: config.get('showTags'),
      crank: config.get('crank'),
    },
    pageContent: config.get('pageContent') !== false,
    editorPane: config.get('editorPane') !== false,
  }
}
