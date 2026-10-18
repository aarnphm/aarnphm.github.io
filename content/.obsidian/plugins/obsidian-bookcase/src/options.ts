import type { BasesAllOptions, BasesPropertyId, BasesViewConfig } from 'obsidian'
import {
  BOOKCASE_DEFAULTS,
  BOOKCASE_LIMITS,
  BOOKCASE_STYLES,
  type BookcaseStyle,
  SHELF_MATERIALS,
  type ShelfMaterial,
  clampOption,
} from './scene-model'

const STYLE_NAMES: Record<BookcaseStyle, string> = {
  straight: 'Straight case',
  corner: 'L-shaped',
  spine: 'Spine tower',
  wall: 'Wall-hung',
  zigzag: 'Zigzag',
  ladder: 'Ladder',
}

const MATERIAL_NAMES: Record<ShelfMaterial, string> = {
  paper: 'Paper',
  walnut: 'Walnut',
  'birch-ply': 'Birch ply',
  'black-lacquer': 'Black lacquer',
  steel: 'Steel',
}

const names = <K extends string>(keys: readonly K[], labels: Record<K, string>) =>
  Object.fromEntries(keys.map(key => [key, labels[key]])) as Record<string, string>

const oneOf = <K extends string>(keys: readonly K[], value: unknown): K | null =>
  typeof value === 'string' && (keys as readonly string[]).includes(value) ? (value as K) : null

export interface BookcaseConfig {
  image: BasesPropertyId
  spineTitle: BasesPropertyId
  spineAuthor: BasesPropertyId
  spineColor: BasesPropertyId | null
  bookHeight: number
  shelfWidth: number
  scrollStep: number
  pageContent: boolean
  shelfStyle: BookcaseStyle
  /** Null when the view stores none: the style's own materials apply. */
  shelfMaterial: ShelfMaterial | null
  /** Label plates on the shelves; a topic's `showTags` overrides it. */
  showTags: boolean
}

export const bookcaseOptions = (): BasesAllOptions[] => [
  { type: 'property', key: 'image', displayName: 'Cover image', default: 'note.posters' },
  {
    type: 'dropdown',
    key: 'shelfStyle',
    displayName: 'Shelf style',
    default: 'straight',
    options: names(BOOKCASE_STYLES, STYLE_NAMES),
  },
  {
    type: 'dropdown',
    key: 'shelfMaterial',
    displayName: 'Material',
    default: 'paper',
    options: names(SHELF_MATERIALS, MATERIAL_NAMES),
  },
  {
    type: 'group',
    displayName: 'Spine',
    items: [
      { type: 'property', key: 'spineTitle', displayName: 'Title', default: 'note.title' },
      { type: 'property', key: 'spineAuthor', displayName: 'Author', default: 'note.author' },
      { type: 'property', key: 'spineColor', displayName: 'Colour', placeholder: 'from cover' },
    ],
  },
  {
    type: 'group',
    displayName: 'Shelf',
    items: [
      {
        type: 'slider',
        key: 'bookHeight',
        displayName: 'Book height',
        default: BOOKCASE_DEFAULTS.bookHeight,
        min: BOOKCASE_LIMITS.bookHeight[0],
        max: BOOKCASE_LIMITS.bookHeight[1],
        step: 10,
      },
      {
        type: 'slider',
        key: 'shelfWidth',
        displayName: 'Shelf width',
        default: BOOKCASE_DEFAULTS.shelfWidth,
        min: BOOKCASE_LIMITS.shelfWidth[0],
        max: BOOKCASE_LIMITS.shelfWidth[1],
        step: 20,
      },
      {
        type: 'slider',
        key: 'scrollStep',
        displayName: 'Scroll per book',
        default: BOOKCASE_DEFAULTS.scrollStep,
        min: BOOKCASE_LIMITS.scrollStep[0],
        max: BOOKCASE_LIMITS.scrollStep[1],
        step: 4,
      },
      { type: 'toggle', key: 'showTags', displayName: 'Shelf labels', default: true },
      { type: 'toggle', key: 'pageContent', displayName: 'Show note body', default: true },
    ],
  },
]

export function readConfig(config: BasesViewConfig): BookcaseConfig {
  const pageContent = config.get('pageContent')
  return {
    image: config.getAsPropertyId('image') ?? 'note.posters',
    spineTitle: config.getAsPropertyId('spineTitle') ?? 'note.title',
    spineAuthor: config.getAsPropertyId('spineAuthor') ?? 'note.author',
    spineColor: config.getAsPropertyId('spineColor'),
    bookHeight: clampOption(
      config.get('bookHeight'),
      BOOKCASE_DEFAULTS.bookHeight,
      BOOKCASE_LIMITS.bookHeight,
    ),
    shelfWidth: clampOption(
      config.get('shelfWidth'),
      BOOKCASE_DEFAULTS.shelfWidth,
      BOOKCASE_LIMITS.shelfWidth,
    ),
    scrollStep: clampOption(
      config.get('scrollStep'),
      BOOKCASE_DEFAULTS.scrollStep,
      BOOKCASE_LIMITS.scrollStep,
    ),
    pageContent: typeof pageContent === 'boolean' ? pageContent : true,
    shelfStyle: oneOf(BOOKCASE_STYLES, config.get('shelfStyle')) ?? 'straight',
    shelfMaterial: oneOf(SHELF_MATERIALS, config.get('shelfMaterial')),
    showTags: config.get('showTags') !== false,
  }
}
