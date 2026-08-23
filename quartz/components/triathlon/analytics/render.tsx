import type { Analytics } from '../../../plugins/stores/analytics'
import type { AnalyticsPanelDefinition } from './catalog'
import { DEFAULT_TRIATHLON_FORMATTER, type TriathlonFormatter } from '../runtime/formatter'

export interface AnalyticsChartDomain {
  minimum: number
  maximum: number
}

export const analyticsChartPath = (
  values: readonly number[],
  domain?: AnalyticsChartDomain,
): string => {
  if (values.length === 0) return ''
  let minimum = domain?.minimum ?? Infinity
  let maximum = domain?.maximum ?? -Infinity
  if (domain == null) {
    for (const value of values) {
      if (value < minimum) minimum = value
      if (value > maximum) maximum = value
    }
  }
  const span = Math.max(maximum - minimum, Math.abs(maximum) * 0.01, 1)
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 50 : (index / (values.length - 1)) * 100
      const y = 29 - ((value - minimum) / span) * 26
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

export const AnalyticsServerPanel = ({
  definition,
  data,
  formatter = DEFAULT_TRIATHLON_FORMATTER,
}: {
  definition: AnalyticsPanelDefinition
  data: Analytics
  formatter?: TriathlonFormatter
}) => {
  const content = definition.server(data, formatter)
  const series = (content.series ?? []).filter(item => item.values.length > 0)
  const sharedDomain: AnalyticsChartDomain | undefined =
    content.seriesDomain === 'shared-zero'
      ? { minimum: 0, maximum: Math.max(1, ...series.flatMap(item => item.values)) }
      : undefined
  return (
    <section
      class="tri-ana-ssr"
      aria-label={content.title}
      data-tri-ssr="true"
      data-tri-server-panel={definition.key}
      data-tri-series-count={series.length}
      data-tri-series-domain={content.seriesDomain ?? 'independent'}
    >
      <h2 class="tri-ana-block-title">{content.title}</h2>
      <dl class="tri-ana-ssr-values">
        {content.values.map(item => (
          <div class="tri-ana-ssr-value">
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
      {series.length > 0 && (
        <figure class="tri-ana-ssr-figure">
          <svg
            class="tri-ana-ssr-chart"
            viewBox="0 0 100 32"
            preserveAspectRatio="none"
            role="img"
            aria-label={`${content.title} trends`}
          >
            <title>{`${content.title} trends`}</title>
            <line class="tri-ana-ssr-grid" x1="0" y1="3" x2="100" y2="3" />
            <line class="tri-ana-ssr-grid" x1="0" y1="16" x2="100" y2="16" />
            <line class="tri-ana-ssr-grid" x1="0" y1="29" x2="100" y2="29" />
            {series.map((item, index) => (
              <path
                class={`tri-ana-ssr-line tri-ana-ssr-line--${index % 4}`}
                d={analyticsChartPath(item.values, sharedDomain)}
                vector-effect="non-scaling-stroke"
                data-series={item.label}
              />
            ))}
          </svg>
          <figcaption class="tri-ana-ssr-legend">
            {series.map((item, index) => (
              <span class={`tri-ana-ssr-legend-item tri-ana-ssr-legend-item--${index % 4}`}>
                {item.label}
              </span>
            ))}
          </figcaption>
        </figure>
      )}
    </section>
  )
}
