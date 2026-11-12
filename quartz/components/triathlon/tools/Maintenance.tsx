import { useId } from 'preact/hooks'
import type {
  TriathlonComponentMaintenance,
  TriathlonMaintenance,
  TriathlonMaintenanceRange,
} from '../../../util/triathlon-maintenance'
import { formatTriathlonMaintenanceDistance } from '../../../util/triathlon-maintenance'

const componentAnchor = (scope: string, index: number) => `${scope}-component-${index}`

const ComponentLink = ({
  id,
  component,
  components,
  scope,
}: {
  id: string | undefined
  component: string
  components: TriathlonComponentMaintenance[]
  scope: string
}) => {
  if (!id) return null
  const matches = components.flatMap((entry, index) =>
    entry.component === component && entry.id === id ? [{ entry, index }] : [],
  )
  const match = matches.length === 1 ? matches[0] : undefined
  return (
    <span>
      <span data-i18n={component}>{component}</span>:{' '}
      {match ? (
        <a href={`#${componentAnchor(scope, match.index)}`} data-no-popover>
          {match.entry.type}
        </a>
      ) : (
        id
      )}
    </span>
  )
}

const MaintenanceDistance = ({ distanceMiles }: { distanceMiles: number | null }) =>
  distanceMiles === null ? (
    <span>–</span>
  ) : (
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
              {entry.type && <span>{entry.type}</span>}
              <MaintenanceDistance distanceMiles={entry.distanceMiles} />
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

const ComponentRecords = ({
  maintenance,
  scope,
}: {
  maintenance: TriathlonMaintenance
  scope: string
}) => {
  if (maintenance.components.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="components">
        components
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.components.map((entry, index) => (
          <li class="tri-maintenance-entry" id={componentAnchor(scope, index)} tabIndex={-1}>
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

const ChainRecords = ({
  maintenance,
  scope,
}: {
  maintenance: TriathlonMaintenance
  scope: string
}) => {
  if (maintenance.chains.length === 0) return null
  return (
    <div class="tri-maintenance-group">
      <span class="tri-maintenance-group-label" data-i18n="chains">
        chains
      </span>
      <ol class="tri-maintenance-list">
        {maintenance.chains.map(entry => (
          <li class="tri-maintenance-entry" data-chain-id={entry.id}>
            <div class="tri-maintenance-entry-head">
              <span class="tri-maintenance-entry-label">
                <span data-i18n="chain">chain</span> {entry.id}
              </span>
              <span class="tri-maintenance-entry-name">
                {entry.wax ?? entry.lubricant ?? '–'}
                {entry.wax && (
                  <span class="tri-maintenance-tag" data-i18n="wax">
                    wax
                  </span>
                )}
                {entry.status && (
                  <span class="tri-maintenance-tag" data-i18n={entry.status}>
                    {entry.status}
                  </span>
                )}
              </span>
            </div>
            <span class="tri-maintenance-meta">
              {entry.ranges ? (
                entry.ranges.map(range => (
                  <span>
                    {range.start} → {range.end ?? <span data-i18n="current">current</span>}
                  </span>
                ))
              ) : (
                <span>
                  <span data-i18n="since">since</span> {entry.since}
                </span>
              )}
              <MaintenanceDistance distanceMiles={entry.distanceMiles} />
              <span>
                <span data-i18n="waxed">waxed</span>{' '}
                <span data-i18n={entry.waxed ? 'yes' : 'no'}>{entry.waxed ? 'yes' : 'no'}</span>
              </span>
            </span>
            <span class="tri-maintenance-meta">
              <ComponentLink
                id={entry.chainring}
                component="chainrings"
                components={maintenance.components}
                scope={scope}
              />
              <ComponentLink
                id={entry.ospw}
                component="OSPW"
                components={maintenance.components}
                scope={scope}
              />
              {entry.waxKit && (
                <span>
                  <a href={entry.waxKit.url}>{entry.waxKit.name}</a>
                </span>
              )}
            </span>
            {entry.cycles && entry.cycles.length > 0 && (
              <ol
                class="tri-maintenance-cycles"
                aria-label="chain cycles"
                data-i18n-aria-label="chain cycles"
              >
                {entry.cycles.map((cycle, index) => (
                  <li class="tri-maintenance-meta">
                    <span>
                      <span data-i18n="cycle">cycle</span> {index + 1}
                    </span>
                    <span>
                      {cycle.start && <>{cycle.start} → </>}
                      {cycle.end ?? <span data-i18n="current">current</span>}
                    </span>
                    <MaintenanceDistance distanceMiles={cycle.distanceMiles} />
                    {cycle.reasoning && <span>{cycle.reasoning}</span>}
                  </li>
                ))}
              </ol>
            )}
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
  const scope = useId()
  if (!maintenance) return null
  return (
    <section class="tri-maintenance" aria-label="maintenance" data-i18n-aria-label="maintenance">
      <span class="tri-maintenance-heading" data-i18n="maintenance">
        maintenance
      </span>
      <ServiceRecords maintenance={maintenance} />
      <ComponentRecords maintenance={maintenance} scope={scope} />
      <ChainRecords maintenance={maintenance} scope={scope} />
      <WheelRecords maintenance={maintenance} />
    </section>
  )
}
