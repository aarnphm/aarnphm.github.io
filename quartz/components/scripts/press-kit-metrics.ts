import {
  activityCardStatKey,
  parseActivityCardData,
  type ActivityCardData,
  type ActivityCardStat,
} from '../../util/press-kit-activity'
import { activityMetricLabel, cardLocale, localizeActivity } from './press-kit-i18n'

// The shifting caption keeps both the teeth pairing and the gear-number pairing in the DOM
// (CSS shows one), so textContent would read "54×172×7". The card wants the ratio only.
function dominantGearValue(range: Element | null): string | undefined {
  const teeth = range?.querySelector('[data-shift-label="teeth"]')?.textContent?.trim()
  const match = teeth?.match(/^(\d+)×(\d+)$/)
  if (!match) return undefined
  const ratio = (Number(match[1]) / Number(match[2])).toFixed(2)
  const clock = range?.lastElementChild?.textContent?.trim() ?? ''
  return `${teeth} · ${ratio} ${clock}`
}

function textOf(node: Element | null): string {
  return node?.getAttribute('aria-label') ?? node?.textContent?.replace(/\s+/g, ' ').trim() ?? ''
}

export function activityCardFromElement(button: HTMLElement): ActivityCardData | null {
  const parsed = parseActivityCardData(button.dataset.pressKitActivity)
  if (!parsed) return null
  const locale = cardLocale()
  const activity = localizeActivity(parsed, locale)
  const root = button.closest<HTMLElement>('.tri-act')
  if (!root) return activity
  const stats: ActivityCardStat[] = [...activity.stats]
  const add = (key: string, label: string, value: string, provenance?: string): void => {
    if (
      !label ||
      !value ||
      value === '—' ||
      stats.some(stat => stat.key.toLowerCase() === key.toLowerCase())
    )
      return
    stats.push({ key, label: activityMetricLabel(label, locale), value, provenance })
  }
  for (const row of root.querySelectorAll(
    'table.tri-act-stats tr, table.tri-environment-summary-table tr',
  )) {
    const cells = Array.from(row.querySelectorAll(':scope > td, :scope > th'))
    if (cells.length < 2 || row.closest('thead')) continue
    const label = textOf(cells[0])
    const ownerKey =
      cells[0].getAttribute('data-i18n') ??
      cells[0].querySelector<HTMLElement>('[data-i18n]')?.dataset.i18n ??
      label
    const key = activityCardStatKey(ownerKey) ?? `metric:${ownerKey.toLowerCase()}`
    const values = cells
      .slice(1)
      .map(cell => cell.textContent?.replace(/\s+/g, ' ').trim())
      .filter(Boolean)
    const provenance = cells
      .slice(1)
      .map(cell => cell.getAttribute('aria-label') ?? cell.getAttribute('title'))
      .filter(Boolean)
      .join(' · ')
    add(key, label, values.join(' · '), provenance || undefined)
  }
  for (const cap of root.querySelectorAll('.tri-elev-cap')) {
    const trace = cap.closest<HTMLElement>('[data-tri-trace]')?.dataset.triTrace
    const label = cap.querySelector('.tri-elev-d')?.textContent?.trim()
    const range = cap.querySelector('.tri-elev-range:not(.tri-trace-reference-k)')
    const value =
      trace === 'electronic-shifting'
        ? dominantGearValue(range)
        : range?.textContent?.trim()
    if (trace && label && value)
      add(
        `trace:${trace}`,
        label,
        value,
        cap.querySelector('[data-gloss-def]')?.getAttribute('data-gloss-def') ?? undefined,
      )
  }
  for (const meter of root.querySelectorAll<HTMLElement>('.tri-training-effect [role="meter"]')) {
    const label = meter.getAttribute('aria-label')
    const value = meter.getAttribute('aria-valuenow')
    if (label && value) add(`training-effect:${label}`, label, value)
  }
  return { ...activity, stats }
}
