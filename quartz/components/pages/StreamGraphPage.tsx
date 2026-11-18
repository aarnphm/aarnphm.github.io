import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { getStreamDatePartsFromIso } from '../../util/stream'
import {
  listStreamGraphIds,
  listStreamGraphSummaries,
  streamGraphHref,
  streamGraphsIndexHref,
  streamGraphSlug,
} from '../../util/stream-graph'
//@ts-ignore
import script from '../scripts/inquiry-map.inline'
import style from '../styles/inquiry-map.scss'
import streamStyle from '../styles/stream.scss'
import streamGraphStyle from '../styles/streamGraph.scss'

const graphIdFromSlug = (slug: string | undefined): string | null => {
  if (!slug) return null
  const parts = slug.split('/')
  if (parts.length !== 3 || parts[0] !== 'stream' || parts[1] !== 'graphs') return null
  try {
    return decodeURIComponent(parts[2])
  } catch {
    return parts[2]
  }
}

const isGraphsIndexSlug = (slug: string | undefined): boolean => slug === 'stream/graphs'

const plural = (count: number, singular: string): string => {
  if (singular === 'entry') return `${count} ${count === 1 ? 'entry' : 'entries'}`
  if (singular === 'graph') return `${count} ${count === 1 ? 'graph' : 'graphs'}`
  return `${count} ${count === 1 ? singular : `${singular}s`}`
}

const pluralLabel = (count: number, singular: string): string => {
  if (singular === 'entry') return count === 1 ? 'entry' : 'entries'
  return count === 1 ? singular : `${singular}s`
}

const renderSummaryMetric = (count: number, singular: string) => (
  <span class={`stream-legend-summary-metric stream-legend-summary-metric-${singular}`}>
    <span class="stream-legend-summary-number">{count}</span>
    <span>{pluralLabel(count, singular)}</span>
  </span>
)

const renderEmptySummaryMetric = () => (
  <span class="stream-legend-summary-metric stream-legend-summary-metric-empty" aria-hidden="true">
    <span class="stream-legend-summary-number"></span>
    <span></span>
  </span>
)

const slugFromPath = (path: string): string => path.replace(/^\//, '')

const renderCrumb = (graphId: string) => (
  <nav class="stream-graph-page-crumb" aria-label="breadcrumb">
    <a class="internal" href="/stream" data-no-popover>
      stream
    </a>
    <span aria-hidden="true"> / </span>
    <a class="internal" href={streamGraphsIndexHref()} data-no-popover>
      graphs
    </a>
    <span aria-hidden="true"> / </span>
    <span class="stream-graph-page-id">{graphId}</span>
  </nav>
)

export default (() => {
  const StreamGraphPage: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    const slug = typeof fileData.slug === 'string' ? fileData.slug : undefined
    const entries = fileData.streamData?.entries ?? []

    if (isGraphsIndexSlug(slug) || fileData.frontmatter?.streamGraphsIndex === true) {
      const summaries = listStreamGraphSummaries(entries)
      return (
        <article
          class="stream main-col popover-hint"
          data-stream-view="graphs"
          data-stream-layout="table"
        >
          {summaries.length === 0 ? (
            <p class="stream-graph-page-empty">no published graphs.</p>
          ) : (
            <section class="stream-legend" aria-label="stream graphs">
              <p class="stream-legend-count">{plural(summaries.length, 'graph')}</p>
              <div class="stream-legend-groups">
                {summaries.map(graph => {
                  const href = streamGraphHref(graph.id)
                  const latestDay = getStreamDatePartsFromIso(graph.latest)?.dayText
                  return (
                    <section key={graph.id} class="stream-legend-group stream-legend-year">
                      <div class="stream-legend-row">
                        <a
                          class="stream-legend-row-title internal"
                          href={href}
                          data-slug={slugFromPath(href)}
                          data-no-popover
                        >
                          {graph.id}
                        </a>
                        {renderEmptySummaryMetric()}
                        {latestDay ? (
                          <span class="stream-legend-summary-metric stream-legend-summary-metric-day">
                            <span class="stream-legend-summary-number">{latestDay}</span>
                            <span>latest</span>
                          </span>
                        ) : (
                          renderEmptySummaryMetric()
                        )}
                        {renderSummaryMetric(graph.entries, 'entry')}
                      </div>
                    </section>
                  )
                })}
              </div>
            </section>
          )}
        </article>
      )
    }

    const graphId =
      (typeof fileData.frontmatter?.streamGraphId === 'string'
        ? fileData.frontmatter.streamGraphId
        : null) ?? graphIdFromSlug(slug)
    if (!graphId) {
      return <article class="stream-graph-page main-col">missing graph.</article>
    }

    const known = new Set(listStreamGraphIds(entries))
    if (!known.has(graphId)) {
      return (
        <article class="stream-graph-page main-col">
          {renderCrumb(graphId)}
          <p class="stream-graph-page-empty">
            no published members for <span class="stream-graph-page-id">{graphId}</span>.
          </p>
        </article>
      )
    }

    return (
      <article class="stream-graph-page main-col" data-stream-graph-page={graphId}>
        <header class="stream-graph-page-header">{renderCrumb(graphId)}</header>
        <figure
          class="inquiry-map stream-inquiry-map stream-inquiry-map-page"
          data-inquiry-map
          data-inquiry-src={`/${streamGraphSlug(graphId)}`}
          data-inquiry-seed="chrono"
          data-graph-id={graphId}
          role="group"
          aria-label={`${graphId} graph`}
        >
          <canvas class="inquiry-map-halo" aria-hidden="true" />
          <svg class="inquiry-map-links" aria-hidden="true" />
          <div class="inquiry-map-list" />
          <aside class="stream-graph-preview" data-stream-graph-preview hidden aria-live="polite" />
        </figure>
      </article>
    )
  }

  StreamGraphPage.css = [streamStyle, style, streamGraphStyle]
  StreamGraphPage.afterDOMLoaded = script
  return StreamGraphPage
}) satisfies QuartzComponentConstructor
