import {
  activityCardStatKey,
  parseActivityCardData,
  type ActivityCardData,
  type ActivityCardStat,
} from '../../util/press-kit-activity'
import { WIND_TRACE_COLORS } from '../triathlon/activity/trace-colors'

export function activityMetricColor(key: string): string {
  if (/heart.?rate|heartRate/i.test(key)) return WIND_TRACE_COLORS[4]
  if (/cadence|smoothness|stride/i.test(key)) return WIND_TRACE_COLORS[2]
  if (/power|torque|effectiveness/i.test(key)) return WIND_TRACE_COLORS[0]
  if (/pace|speed/i.test(key)) return WIND_TRACE_COLORS[3]
  if (/elevation|grade/i.test(key)) return WIND_TRACE_COLORS[6]
  let hash = 0
  for (const character of key) hash = (hash * 31 + character.charCodeAt(0)) >>> 0
  return WIND_TRACE_COLORS[hash % WIND_TRACE_COLORS.length]
}

function textOf(node: Element | null): string {
  return node?.getAttribute('aria-label') ?? node?.textContent?.replace(/\s+/g, ' ').trim() ?? ''
}

export function activityCardFromElement(button: HTMLElement): ActivityCardData | null {
  const activity = parseActivityCardData(button.dataset.pressKitActivity)
  if (!activity) return null
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
    const estimate = provenance && /estim|model|Garden/i.test(provenance)
    stats.push({ key, label: `${label}${estimate ? ' · estimation' : ''}`, value, provenance })
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
    const value = cap
      .querySelector('.tri-elev-range:not(.tri-trace-reference-k)')
      ?.textContent?.trim()
    if (trace && label && value)
      add(
        `trace:${trace}`,
        `${label} · trace`,
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
