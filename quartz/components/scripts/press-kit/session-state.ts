import type { ActivityCardData, OverlayKey, PowerWindow } from '../../../util/press-kit-activity'
import type { Locale } from '../../../util/triathlon-presentation'
import type { Select } from '../../controls/select'
import type { PressKitPage } from '../../PressKit'
import type { CalendarCardData } from '../press-kit-calendar'
import type { BlockDisplay, DayCardData } from '../press-kit-day'
import type { Run } from '../press-kit-text'
import { PressKitEditor } from '../../PressKitEditor'
import { FORMATS, type Fonts, type FormatKey, type ThemeKey } from '../press-kit-card'
import { cardLocale } from '../press-kit-i18n'

export type PressKitTab = 'note' | 'quote' | 'links'

export const FORMAT_KEYS = Object.keys(FORMATS) as FormatKey[]

const PREFS_KEY = 'press-kit:prefs'

export interface Subject {
  kind: 'page' | 'entry' | 'activity' | 'day' | 'calendar'
  url: string
  title: Run[]
  description: Run[]
  excerpt: Run[]
  date?: string
  iso?: string
  reading?: string
  tags: string[]
  crumbs: string[]
  /** headings that a link can point into */
  headings: HTMLElement[]
  activity?: ActivityCardData
  day?: DayCardData
  calendar?: CalendarCardData
  entryTitle?: string
  entryDate?: string
}

export interface Quote {
  runs: Run[]
  attribution?: string
  /** false while the quote is the page's own excerpt, before any selection or edit */
  chosen: boolean
  /** `text=` directive for a scroll-to-text link, when the passage is plain prose */
  fragment?: string
  section?: HTMLElement
}

export interface OpenRequest {
  tab?: PressKitTab
  entry?: HTMLElement
  entryUrl?: string
  range?: Range
  activity?: ActivityCardData
  calendar?: CalendarCardData
}

export const session: {
  dialog: HTMLDialogElement | null
  page: PressKitPage | null
  activeSignal: AbortSignal | undefined
  tab: PressKitTab
  subject: Subject | null
  quote: Quote
  anchor: string
  format: FormatKey
  theme: ThemeKey
  overlay: OverlayKey
  language: Locale | null
  powerWindow: PowerWindow
  powerCurveScale: 'linear' | 'log'
  noteEdited: boolean
  anchorSelect: Select<string> | null
  unmountAnchor: (() => void) | null
  overlaySelect: Select<OverlayKey> | null
  fonts: Fonts | null
  frame: number
  blob: Promise<Blob | null> | null
  blobTimer: number
  rendering: Promise<void> | null
  drawVersion: number
  renderFailed: boolean
  editor: PressKitEditor | null
  pendingG: number
  lastRange: { range: Range; at: number } | null
  pill: HTMLButtonElement | null
} = {
  dialog: null,
  page: null,
  activeSignal: undefined,
  tab: 'note',
  subject: null,
  quote: { runs: [], chosen: false },
  anchor: 'page',
  format: 'portrait',
  theme: 'papier',
  overlay: 'card',
  language: null,
  powerWindow: 30,
  powerCurveScale: 'log',
  noteEdited: false,
  anchorSelect: null,
  unmountAnchor: null,
  overlaySelect: null,
  fonts: null,
  frame: 0,
  blob: null,
  blobTimer: 0,
  rendering: null,
  drawVersion: 0,
  renderFailed: false,
  editor: null,
  pendingG: -Infinity,
  lastRange: null,
  pill: null,
}
export const selectedMetrics = new Map<string, Set<string>>()
export const show = {
  art: false,
  description: true,
  excerpt: false,
  meta: true,
  highlight: false,
  section: true,
  title: true,
  date: true,
  metrics: true,
  map: true,
  distance: true,
  time: true,
  pace: true,
  elevation: true,
  heartRate: true,
  power: true,
  entry: false,
  transparent: false,
  cyclingPower: false,
  powerCurve: false,
  cyclingLaps: false,
  runPace: false,
  runLaps: false,
}
// The base card by daylight, and its night twin for the encre ground.
export const arts: { day?: HTMLImageElement; night?: HTMLImageElement } = {}
export const selectedBlocks = new Set<string>()
export const selectedActivities = new Set<string>()
export const blockDisplays = new Map<string, BlockDisplay>()

export const $ = <T extends Element = HTMLElement>(selector: string) =>
  session.dialog?.querySelector<T>(selector) ?? null

export const shareLocale = (): Locale => session.language ?? cardLocale()

export function toast(message: string) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message } }))
}

export function readPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) ?? 'null') as {
      format?: FormatKey
      art?: boolean
    } | null
    if (saved?.format && FORMAT_KEYS.includes(saved.format)) session.format = saved.format
    if (typeof saved?.art === 'boolean') show.art = saved.art
  } catch {}
  session.theme =
    document.documentElement.getAttribute('saved-theme') === 'dark' ? 'encre' : 'papier'
}

export function writePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ format: session.format, art: show.art }))
  } catch {}
}
