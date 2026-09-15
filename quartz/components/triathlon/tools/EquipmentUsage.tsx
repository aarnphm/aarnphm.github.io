import type { TriathlonEquipmentUsage } from '../../../util/triathlon-equipment'
import { formatEquipmentDistance } from '../../../util/triathlon-equipment'

export const EquipmentUsage = ({ usage }: { usage: TriathlonEquipmentUsage | undefined }) => {
  if (!usage) return null
  const kilometres = usage.lifetimeDistanceM == null ? null : usage.lifetimeDistanceM / 1_000
  return (
    <span class="tri-gear-usage" data-gear-id={usage.id} data-equipment-source={usage.source}>
      <span data-i18n="distance">distance</span>
      {' - '}
      {kilometres == null ? (
        '—'
      ) : (
        <span class="tri-unit-distance" data-kind="equipment" data-km={kilometres}>
          {formatEquipmentDistance(kilometres * 1_000, 'imperial')}
        </span>
      )}
      {', '}
      {usage.activityCount} <span data-i18n="activities">activities</span>
    </span>
  )
}
