import type { ActivityKind, StravaActivityDetail } from '../../../plugins/stores/strava'
import { KM_TO_MI, M_TO_FT, clock } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  type TriathlonPresentation,
} from '../../../util/triathlon-presentation'

export interface TriathlonFormatter {
  presentation: TriathlonPresentation
  text(key: string): string
  number(value: number, minimumDigits?: number, maximumDigits?: number): string
  shortDate(iso: string): string
  longDate(iso: string): string
  month(iso: string): string
  monthYear(iso: string): string
  weekdayNarrow(day: number): string
  distance(kilometres: number, sport: ActivityKind): string
  elevation(metres: number): string
  temperature(celsius: number): string
  weight(kilograms: number): string
  pace(secondsPerKilometre: number): string
  powerView(activity: StravaActivityDetail): StravaActivityDetail
}

const date = (iso: string): Date | null => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!match) return null
  const value = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])))
  return Number.isNaN(value.valueOf()) ? null : value
}

export const createTriathlonFormatter = (value: TriathlonPresentation): TriathlonFormatter => {
  const presentation = Object.freeze({ ...value })
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const imperial = presentation.distance === 'imperial'
  return Object.freeze({
    presentation,
    text: (key: string) => triText(presentation.locale, key),
    number: (input: number, minimumDigits = 0, maximumDigits = minimumDigits) =>
      input.toLocaleString(locale, {
        minimumFractionDigits: minimumDigits,
        maximumFractionDigits: maximumDigits,
      }),
    shortDate: (iso: string) => {
      const value = date(iso)
      return value
        ? value.toLocaleDateString(locale, { month: 'short', day: 'numeric', timeZone: 'UTC' })
        : iso
    },
    longDate: (iso: string) => {
      const value = date(iso)
      return value
        ? value.toLocaleDateString(locale, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            timeZone: 'UTC',
          })
        : iso
    },
    month: (iso: string) => {
      const value = date(iso)
      return value ? value.toLocaleDateString(locale, { month: 'short', timeZone: 'UTC' }) : iso
    },
    monthYear: (iso: string) => {
      const value = date(iso)
      return value
        ? value.toLocaleDateString(locale, { year: 'numeric', month: 'long', timeZone: 'UTC' })
        : iso
    },
    weekdayNarrow: (day: number) =>
      new Date(Date.UTC(2024, 0, 7 + Math.min(6, Math.max(0, day))))
        .toLocaleDateString(locale, { weekday: 'narrow', timeZone: 'UTC' })
        .toUpperCase(),
    distance: (kilometres: number, sport: ActivityKind) => {
      if (sport === 'swim') return `${Math.round(kilometres * 1000).toLocaleString(locale)} m`
      return imperial ? `${(kilometres * KM_TO_MI).toFixed(1)} mi` : `${kilometres.toFixed(1)} km`
    },
    elevation: (metres: number) =>
      imperial
        ? `${Math.round(metres * M_TO_FT).toLocaleString(locale)} ft`
        : `${Math.round(metres).toLocaleString(locale)} m`,
    temperature: (celsius: number) =>
      imperial ? `${((celsius * 9) / 5 + 32).toFixed(1)}°F` : `${celsius.toFixed(1)}°C`,
    weight: (kilograms: number) =>
      imperial ? `${(kilograms / 0.45359237).toFixed(1)} lb` : `${kilograms.toFixed(1)} kg`,
    pace: (secondsPerKilometre: number) =>
      imperial
        ? `${clock(secondsPerKilometre / KM_TO_MI)} /mi`
        : `${clock(secondsPerKilometre)} /km`,
    powerView: (activity: StravaActivityDetail) => {
      const filtered = activity.powerWithoutZeros
      if (presentation.powerSamples !== 'exclude-zero' || activity.sport !== 'bike' || !filtered)
        return activity
      return {
        ...activity,
        avgWatts: filtered.avgWatts ?? activity.avgWatts,
        powerZones: filtered.powerZones ?? activity.powerZones,
        powerHist: filtered.powerHist ?? activity.powerHist,
      }
    },
  })
}

export const DEFAULT_TRIATHLON_FORMATTER = createTriathlonFormatter(DEFAULT_TRIATHLON_PRESENTATION)
