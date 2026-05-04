import type { StreamEntry } from '../../plugins/transformers/stream'
import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import {
  buildStreamDayPathFromIso,
  buildStreamEntryPathFromIso,
  groupStreamEntries,
  groupStreamEntriesByYear,
  type StreamMonthGroup,
  type StreamYearGroup,
} from '../../util/stream'
// @ts-ignore
import script from '../scripts/stream.inline'
import {
  formatWordCount,
  getStreamEntryWordCount,
  isProtectedEntry,
  isRestrictedEntry,
  renderStreamEntry,
} from '../stream/Entry'
import { StreamLockIcon } from '../stream/UnlockIcon'
import StreamSearchConstructor from '../StreamSearch'
import style from '../styles/stream.scss'

type StreamView =
  | { kind: 'root' }
  | { kind: 'index' }
  | { kind: 'year'; year: string }
  | { kind: 'month'; year: string; month: string }
  | { kind: 'day'; year: string; month: string; day: string }

const streamViewForSlug = (slug: string): StreamView => {
  const parts = slug.split('/')

  if (parts.length === 1 && parts[0] === 'stream') return { kind: 'root' }
  if (parts.length === 2 && parts[0] === 'stream' && parts[1] === 'on') {
    return { kind: 'index' }
  }
  if (parts.length === 3 && parts[0] === 'stream' && parts[1] === 'on') {
    return { kind: 'year', year: parts[2] }
  }
  if (parts.length === 4 && parts[0] === 'stream' && parts[1] === 'on') {
    return { kind: 'month', year: parts[2], month: parts[3] }
  }
  if (parts.length === 5 && parts[0] === 'stream' && parts[1] === 'on') {
    return { kind: 'day', year: parts[2], month: parts[3], day: parts[4] }
  }
  return { kind: 'root' }
}

const pluralLabel = (count: number, singular: string): string => {
  if (singular === 'entry') return count === 1 ? 'entry' : 'entries'
  return count === 1 ? singular : `${singular}s`
}

const plural = (count: number, singular: string): string =>
  `${count} ${pluralLabel(count, singular)}`

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

const restrictedLabel = (count: number): string =>
  count === 1 ? '1 locked entry' : `${count} locked entries`

const restrictedCopy = 'metadata only here. daily views keep the normal unlock flow.'

const renderLockPopover = (count: number) => {
  const label = restrictedLabel(count)

  return (
    <span
      class="stream-legend-lock"
      tabindex={0}
      aria-label={`${label}. ${restrictedCopy}`}
      data-lock-popover-title={label}
      data-lock-popover-copy={restrictedCopy}
    >
      <StreamLockIcon />
    </span>
  )
}

const renderRestrictedMetric = (count: number) =>
  count > 0 ? (
    <span class="stream-legend-summary-metric stream-legend-summary-metric-restricted">
      <span class="stream-legend-summary-number">{count}</span>
      {renderLockPopover(count)}
    </span>
  ) : (
    renderEmptySummaryMetric()
  )

const tagsForEntry = (entry: StreamEntry): string[] =>
  Array.isArray(entry.metadata.tags)
    ? entry.metadata.tags.map(tag => String(tag).trim()).filter(tag => tag.length > 0)
    : []

const entryTitle = (entry: StreamEntry): string => {
  const title = entry.title?.trim()
  if (title) return title

  const description = entry.description?.trim()
  if (description) return description

  return 'entry'
}

const entryMeta = (entry: StreamEntry): string => {
  const words = getStreamEntryWordCount(entry)
  return words > 0 ? formatWordCount(words) : ''
}

const slugFromPath = (path: string): string => path.replace(/^\//, '')

const formatLegendDay = (isoDate: string | undefined | null): string => {
  if (!isoDate) return ''
  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return ''

  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
    .format(date)
    .toLowerCase()
}

const formatLegendMonth = (month: StreamMonthGroup): string =>
  new Intl.DateTimeFormat('en-US', { month: 'long', timeZone: 'UTC' })
    .format(new Date(Date.UTC(month.year, month.month - 1, 1)))
    .toLowerCase()

const restrictedCount = (entries: StreamEntry[]): number => entries.filter(isRestrictedEntry).length

export default (() => {
  const StreamSearch = StreamSearchConstructor()

  const StreamPage: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData } = props
    if (!fileData.streamData || fileData.streamData.entries.length === 0) {
      return <article class="stream-empty main-col popover-hint">stream is empty.</article>
    }

    const canonicalPathRaw = fileData.frontmatter?.streamCanonical
    const canonicalPath =
      typeof canonicalPathRaw === 'string' && canonicalPathRaw.trim().length > 0
        ? canonicalPathRaw.startsWith('/')
          ? canonicalPathRaw
          : `/${canonicalPathRaw}`
        : '/stream'

    const slug = fileData.slug ?? ''
    const streamView = streamViewForSlug(typeof slug === 'string' ? slug : '')
    const isDailyView = streamView.kind === 'day'

    const groups = groupStreamEntries(fileData.streamData.entries)
    const entriesWithContext = groups.flatMap(group =>
      group.entries.map(entry => ({ entry, group })),
    )
    const protectedPayloads = fileData.streamData.protectedPayloads
    const mode = isDailyView ? 'daily' : 'listing'
    const protectedEntryCount = isDailyView
      ? entriesWithContext.filter(
          ({ entry }) => isProtectedEntry(entry) && protectedPayloads?.[entry.id],
        ).length
      : 0
    const hasProtectedUnlockEntries = protectedEntryCount > 0
    const useCompactProtectedPrompts = protectedEntryCount > 2

    const renderUnlockPanel = () =>
      hasProtectedUnlockEntries ? (
        <section class="stream-protected-unlock" data-protected-unlock-panel hidden>
          <form class="stream-protected-unlock-form" data-protected-unlock-all="true">
            <input
              class="password-input"
              type="password"
              placeholder="enter password"
              autocomplete="off"
              required
            />
            <button class="password-submit" type="submit">
              unlock all
            </button>
          </form>
          <p class="password-error" style="display: none;">
            incorrect password
          </p>
        </section>
      ) : null

    const renderFeed = () => (
      <>
        {streamView.kind === 'root' && <StreamSearch {...props} />}
        {isDailyView && renderUnlockPanel()}
        <ol class="stream-feed">
          {entriesWithContext.map(({ entry, group }) =>
            renderStreamEntry(entry, fileData.filePath!, {
              groupId: group.id,
              timestampValue: group.timestamp,
              showDate: true,
              resolvedIsoDate: entry.date ?? group.isoDate,
              showWordCount: true,
              mode,
              encryptedPayload: protectedPayloads?.[entry.id],
              protectedPrompt: useCompactProtectedPrompts ? 'icon' : 'form',
            }),
          )}
        </ol>
      </>
    )

    const renderLegendEntries = (entries: StreamEntry[]) => (
      <div class="stream-legend-table">
        <div class="stream-legend-head" aria-hidden="true">
          <span>date</span>
          <span>entry</span>
          <span>tags</span>
          <span>meta</span>
        </div>
        <ol class="stream-legend-entries">
          {entries.map(entry => {
            const dayPath = buildStreamDayPathFromIso(entry.date)
            const entryPath = buildStreamEntryPathFromIso(entry.date, entry.id)
            const href = entryPath ?? (dayPath ? `${dayPath}#${entry.id}` : '/stream')
            const dayLabel = formatLegendDay(entry.date)
            const tags = tagsForEntry(entry)
            const restricted = isRestrictedEntry(entry)

            return (
              <li
                key={entry.id}
                class="stream-legend-entry"
                data-stream-restricted={restricted ? 'true' : undefined}
              >
                <a
                  class="stream-legend-date internal"
                  href={dayPath ?? '/stream'}
                  data-slug={slugFromPath(dayPath ?? '/stream')}
                  data-no-popover
                >
                  {dayLabel || 'undated'}
                </a>
                <a
                  class="stream-legend-title internal"
                  href={href}
                  data-slug={slugFromPath(dayPath ?? '/stream')}
                  data-no-popover
                >
                  {entryTitle(entry)}
                </a>
                <div class="stream-legend-tags">
                  {tags.map(tag => (
                    <span key={tag} class="stream-entry-tag">
                      {tag}
                    </span>
                  ))}
                </div>
                <div class="stream-legend-meta">
                  {restricted ? renderLockPopover(1) : entryMeta(entry)}
                </div>
              </li>
            )
          })}
        </ol>
      </div>
    )

    const renderMonthGroups = (months: StreamMonthGroup[]) => (
      <div class="stream-legend-groups">
        {months.map(month => {
          const locked = restrictedCount(month.entries)

          return (
            <details key={month.id} class="stream-legend-group">
              <summary class="stream-legend-summary">
                <span class="stream-legend-summary-title">{formatLegendMonth(month)}</span>
                {renderRestrictedMetric(locked)}
                {renderSummaryMetric(month.entries.length, 'entry')}
                {renderEmptySummaryMetric()}
                <a
                  class="stream-legend-open internal"
                  href={month.path}
                  data-slug={slugFromPath(month.path)}
                  data-no-popover
                >
                  open
                </a>
              </summary>
              {renderLegendEntries(month.entries)}
            </details>
          )
        })}
      </div>
    )

    const renderIndex = (years: StreamYearGroup[]) => (
      <section class="stream-legend" aria-label="stream entries by year">
        <p class="stream-legend-count">
          {plural(
            years.reduce((sum, year) => sum + year.entries.length, 0),
            'entry',
          )}
        </p>
        <div class="stream-legend-groups">
          {years.map(year => {
            const locked = restrictedCount(year.entries)

            return (
              <details key={year.id} class="stream-legend-group">
                <summary class="stream-legend-summary">
                  <span class="stream-legend-summary-title">{year.yearText}</span>
                  {renderRestrictedMetric(locked)}
                  {renderSummaryMetric(year.months.length, 'month')}
                  {renderSummaryMetric(year.entries.length, 'entry')}
                  <a
                    class="stream-legend-open internal"
                    href={year.path}
                    data-slug={slugFromPath(year.path)}
                    data-no-popover
                  >
                    open
                  </a>
                </summary>
                {renderMonthGroups(year.months)}
              </details>
            )
          })}
        </div>
      </section>
    )

    const years = groupStreamEntriesByYear(fileData.streamData.entries)
    const matchingYear =
      streamView.kind === 'year' || streamView.kind === 'month'
        ? years.find(year => year.yearText === streamView.year)
        : null
    const matchingMonth =
      streamView.kind === 'month'
        ? matchingYear?.months.find(month => month.monthText === streamView.month)
        : null

    return (
      <article
        class="stream main-col popover-hint"
        data-stream-canonical={canonicalPath}
        data-stream-view={streamView.kind}
      >
        {streamView.kind === 'root' || streamView.kind === 'day' ? (
          renderFeed()
        ) : streamView.kind === 'index' ? (
          renderIndex(years)
        ) : streamView.kind === 'year' ? (
          matchingYear ? (
            renderMonthGroups(matchingYear.months)
          ) : (
            <p class="stream-legend-count">0 entries</p>
          )
        ) : matchingMonth ? (
          renderLegendEntries(matchingMonth.entries)
        ) : (
          <p class="stream-legend-count">0 entries</p>
        )}
        {isDailyView && (
          <div class="stream-backlink">
            <a href="/stream">← back to stream</a>
          </div>
        )}
      </article>
    )
  }

  StreamPage.css = style
  StreamPage.afterDOMLoaded = script

  return StreamPage
}) satisfies QuartzComponentConstructor
