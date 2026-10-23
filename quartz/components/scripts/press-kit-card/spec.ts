import type { Locale } from '../../../util/triathlon-presentation'
import type { CalendarCardData } from '../press-kit-calendar'
import type { DayCardData } from '../press-kit-day'
import type { LayerRenderer } from '../press-kit-layers'
import type { Run } from '../press-kit-text'
import type { MathImage } from './math'
import {
  isActivityCardStatKey,
  type ActivityCardData,
  type ActivityCardStat,
  type OverlayKey,
  type PowerWindow,
} from '../../../util/press-kit-activity'

export type CardRun = Run & { math?: MathImage }

export type FormatKey = 'story' | 'portrait' | 'square' | 'landscape'
export type ThemeKey = 'papier' | 'encre' | 'rose' | 'sauge'
export type CardKind = 'note' | 'quote' | 'entry' | 'activity' | 'day' | 'calendar'

export interface FormatSpec {
  label: string
  width: number
  height: number
  /** top, right, bottom, left. Story keeps clear of the platform's header and reply bar. */
  pad: [number, number, number, number]
  meta: number
  title: [number, number]
  titleLines: number
  prose: [number, number]
  quote: [number, number]
  gap: number
}

export const FORMATS: Record<FormatKey, FormatSpec> = {
  story: {
    label: 'story',
    width: 1080,
    height: 1920,
    pad: [220, 96, 220, 96],
    meta: 28,
    title: [64, 136],
    titleLines: 7,
    prose: [34, 50],
    quote: [44, 96],
    gap: 48,
  },
  portrait: {
    label: 'portrait',
    width: 1080,
    height: 1350,
    pad: [96, 88, 88, 88],
    meta: 26,
    title: [56, 116],
    titleLines: 6,
    prose: [30, 44],
    quote: [40, 84],
    gap: 40,
  },
  square: {
    label: 'carré',
    width: 1080,
    height: 1080,
    pad: [80, 80, 72, 80],
    meta: 24,
    title: [48, 100],
    titleLines: 5,
    prose: [28, 40],
    quote: [36, 72],
    gap: 34,
  },
  landscape: {
    label: 'paysage',
    width: 1200,
    height: 630,
    pad: [52, 64, 48, 64],
    meta: 20,
    title: [40, 76],
    titleLines: 3,
    prose: [22, 30],
    quote: [28, 52],
    gap: 22,
  },
}

export function cardResolution(format: FormatKey) {
  const spec = FORMATS[format]
  const scale = 3840 / Math.max(spec.width, spec.height)
  return { width: Math.round(spec.width * scale), height: Math.round(spec.height * scale), scale }
}

export interface SiteColors {
  light: string
  lightgray: string
  gray: string
  darkgray: string
  dark: string
  secondary: string
}

export interface Palette {
  label: string
  bg: string
  ink: string
  muted: string
  rule: string
  /** quotation mark */
  mark: string
  /** highlighter behind text, and the ink drawn on it */
  hl: string
  hlInk: string
}

const ROSE = '#fdb2a2'
const SAGE = '#cdd597'

function rgb(color: string): [number, number, number] {
  const hex = color.match(/^#([0-9a-f]{6})$/i)
  if (hex) {
    const n = Number.parseInt(hex[1], 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }
  const parts = color.match(/\d+(\.\d+)?/g)?.map(Number) ?? [0, 0, 0]
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0]
}

export const alpha = (color: string, a: number) => `rgba(${rgb(color).join(', ')}, ${a})`

export function palettes(light: SiteColors, dark: SiteColors): Record<ThemeKey, Palette> {
  const onColor = (label: string, bg: string): Palette => ({
    label,
    bg,
    ink: light.dark,
    muted: alpha(light.dark, 0.66),
    rule: alpha(light.dark, 0.28),
    mark: light.dark,
    hl: light.light,
    hlInk: light.dark,
  })
  return {
    papier: {
      label: 'papier',
      bg: light.light,
      ink: light.dark,
      muted: light.darkgray,
      rule: alpha(light.dark, 0.2),
      mark: ROSE,
      hl: light.secondary,
      hlInk: light.dark,
    },
    encre: {
      label: 'encre',
      bg: dark.light,
      ink: dark.dark,
      muted: dark.darkgray,
      rule: alpha(dark.dark, 0.22),
      mark: SAGE,
      hl: ROSE,
      hlInk: dark.light,
    },
    rose: onColor('rose', ROSE),
    sauge: onColor('sauge', SAGE),
  }
}

export interface Fonts {
  header: string
  headerWeight: number
  body: string
  serif: string
  mono: string
}

export interface CardModel {
  kind: CardKind
  crumbs: string[]
  title: CardRun[]
  customTitle?: string
  description: CardRun[]
  body: CardRun[]
  propositions?: { number: string; depth: number; runs: CardRun[] }[]
  date?: string
  reading?: string
  tags: string[]
  section?: string
  /** the site's base social card, or its night twin on encre (`/static/og-image{,-dark}.webp`) */
  art?: HTMLImageElement
  /** the point crops keep in view, in the 1200×630 card's pixels */
  artFocus?: [number, number]
  activity?: ActivityCardData
  day?: DayCardData
  calendar?: CalendarCardData
  training?: HTMLImageElement
  trainingSvg?: string
  trainingLayers?: Map<string, LayerRenderer>
  overlay?: OverlayKey
  metricGap?: number
  railTextScale?: number
  powerWindow?: PowerWindow
  powerCurveScale?: 'linear' | 'log'
  metricKeys?: Record<string, string[]>
  locale?: Locale
  show: {
    tractatus?: boolean
    art: boolean
    description: boolean
    excerpt: boolean
    meta: boolean
    highlight: boolean
    section: boolean
    title: boolean
    date: boolean
    metrics?: boolean
    map: boolean
    distance: boolean
    time: boolean
    pace: boolean
    elevation: boolean
    heartRate: boolean
    power: boolean
    entry: boolean
    transparent: boolean
    cyclingPower: boolean
    powerCurve: boolean
    cyclingLaps: boolean
    runPace: boolean
    runLaps: boolean
  }
}

export function metricShown(
  model: CardModel,
  activity: ActivityCardData,
  stat: ActivityCardStat,
): boolean {
  const keys = model.metricKeys?.[activity.url]
  return keys ? keys.includes(stat.key) : isActivityCardStatKey(stat.key) && model.show[stat.key]
}
