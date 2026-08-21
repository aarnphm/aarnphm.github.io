import type {
  TriathlonMaintenance,
  TriathlonMaintenanceRange,
} from '../../../util/triathlon-maintenance'
import { formatTriathlonMaintenanceDistance } from '../../../util/triathlon-maintenance'

const MaintenanceDistance = ({ distanceMiles }: { distanceMiles: number | null }) =>
  distanceMiles === null ? null : (
    <span class="tri-unit-distance" data-kind="maintenance" data-mi={distanceMiles}>
      {formatTriathlonMaintenanceDistance(distanceMiles, 'imperial')}
    </span>
  )

const UsageMeta = ({
  ranges,
  distanceMiles,
  reason,
  repaired = null,
}: {
  ranges: TriathlonMaintenanceRange[]
  distanceMiles: number | null
  reason: string | null
  repaired?: boolean | null
}) => (
  <span class="tri-maintenance-meta">
    {ranges.map(range => (
      <span>
        {range.start} → {range.end ?? <span data-i18n="current">current</span>}
      </span>
    ))}
    <MaintenanceDistance distanceMiles={distanceMiles} />
    {repaired !== null && (
      <span>
        <span data-i18n="repaired">repaired</span>{' '}
        <span data-i18n={repaired ? 'yes' : 'no'}>{repaired ? 'yes' : 'no'}</span>
      </span>
    )}
    {reason && (
      <span>
        <span data-i18n="reason">reason</span>: {reason}
      </span>
    )}
  </span>
)

const ServiceRecords = ({ maintenance }: { maintenance: TriathlonMaintenance }) => {
  if (maintenance.services.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="service">
        service
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.services.map(entry => (
          <li class="tri-maintenance-entry">
            <div class="tri-maintenance-entry-head">
              <span class="tri-maintenance-entry-label" data-i18n={entry.bike}>
                {entry.bike}
              </span>
              <span class="tri-maintenance-entry-name">{entry.place}</span>
            </div>
            <span class="tri-maintenance-meta">
              <span>{entry.date}</span>
              <MaintenanceDistance distanceMiles={entry.distanceMiles} />
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

const ComponentRecords = ({ maintenance }: { maintenance: TriathlonMaintenance }) => {
  if (maintenance.components.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="components">
        components
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.components.map(entry => (
          <li class="tri-maintenance-entry">
            <div class="tri-maintenance-entry-head">
              <span class="tri-maintenance-entry-label" data-i18n={entry.component}>
                {entry.component}
              </span>
              <span class="tri-maintenance-entry-name">{entry.type}</span>
            </div>
            <UsageMeta
              ranges={entry.ranges}
              distanceMiles={entry.distanceMiles}
              reason={entry.reason}
            />
          </li>
        ))}
      </ol>
    </div>
  )
}

const ChainRecords = ({ maintenance }: { maintenance: TriathlonMaintenance }) => {
  if (maintenance.chains.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="chains">
        chains
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.chains.map(entry => (
          <li class="tri-maintenance-entry">
            <div class="tri-maintenance-entry-head">
              <span class="tri-maintenance-entry-label">
                <span data-i18n="chain">chain</span> {entry.id}
              </span>
              <span class="tri-maintenance-entry-name">{entry.lubricant}</span>
            </div>
            <span class="tri-maintenance-meta">
              <span>
                <span data-i18n="since">since</span> {entry.since}
              </span>
              <MaintenanceDistance distanceMiles={entry.distanceMiles} />
              <span>
                <span data-i18n="waxed">waxed</span>{' '}
                <span data-i18n={entry.waxed ? 'yes' : 'no'}>{entry.waxed ? 'yes' : 'no'}</span>
              </span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

const WheelRecords = ({ maintenance }: { maintenance: TriathlonMaintenance }) => {
  if (maintenance.wheels.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="tires">
        tires
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.wheels.map(entry => (
          <li class="tri-maintenance-entry">
            <div class="tri-maintenance-entry-head">
              <span class="tri-maintenance-entry-label">
                <span data-i18n={entry.position}>{entry.position}</span>{' '}
                <span data-i18n={entry.part}>{entry.part}</span>
              </span>
              <span class="tri-maintenance-entry-name">{entry.type}</span>
            </div>
            <UsageMeta
              ranges={entry.ranges}
              distanceMiles={entry.distanceMiles}
              reason={entry.reason}
              repaired={entry.repaired}
            />
          </li>
        ))}
      </ol>
    </div>
  )
}

export const Maintenance = ({ maintenance }: { maintenance: TriathlonMaintenance | null }) => {
  if (!maintenance) return null
  return (
    <section class="tri-maintenance" aria-label="maintenance" data-i18n-aria-label="maintenance">
      <span class="tri-maintenance-heading" data-i18n="maintenance">
        maintenance
      </span>
      <ServiceRecords maintenance={maintenance} />
      <ComponentRecords maintenance={maintenance} />
      <ChainRecords maintenance={maintenance} />
      <WheelRecords maintenance={maintenance} />
    </section>
  )
}
