import type { TriathlonTreeYear } from '../../../util/triathlon-date-route'
import { distCombined } from '../../../util/triathlon-card'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../../util/triathlon-presentation'
import { ShortcutHint } from '../../triathlon/shell/ShortcutHint'

const treeDur = (timeS: number): string => {
  const minutes = Math.round(timeS / 60)
  const h = Math.floor(minutes / 60)
  return h > 0 ? `${h}h ${(minutes % 60).toString().padStart(2, '0')}m` : `${minutes}m`
}

const TreeSum = ({ count, km, timeS }: { count: number; km: number; timeS: number }) => (
  <span class="tri-tree-sum">
    <span class="tri-tree-c tri-tree-c--n">{count}</span>
    <span class="tri-tree-c tri-tree-c--km tri-unit-distance" data-km={km} data-kind="combined">
      {distCombined(DEFAULT_TRIATHLON_PRESENTATION, km)}
    </span>
    <span class="tri-tree-c tri-tree-c--t">{treeDur(timeS)}</span>
  </span>
)

const OnModeSwitch = () => (
  <div class="tri-on-modes" role="group" aria-label="view" data-i18n-aria-label="view">
    {(['log', 'feed'] as const).map(mode => (
      <button
        type="button"
        class="tri-on-mode"
        data-on-mode-select={mode}
        aria-pressed={mode === 'log' ? 'true' : 'false'}
      >
        <span data-i18n={mode}>{mode}</span>
      </button>
    ))}
  </div>
)

// log is the date tree; feed is the flat, searchable activity list over the same date scope.
export const OnTreePanel = ({ tree, root }: { tree: TriathlonTreeYear[]; root: string }) => (
  <section
    class="tri-feed tri-tree"
    aria-label="training log by date"
    data-on-mode="log"
    tabindex={-1}
    data-keyboard-scroll-scope
  >
    <div class="tri-ana-bar tri-feed-bar">
      <span class="tri-tree-count">{tree.reduce((total, year) => total + year.count, 0)}</span>
      <div class="tri-feed-search-wrap tri-key-anchor tri-key-search">
        <input
          class="tri-ana-search tri-feed-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk|strength|yoga|sauna, sort:distance|cadence|pace)"
          aria-label="search activities"
          aria-controls="tri-feed-results"
          aria-expanded="false"
          autocomplete="off"
        />
        <ShortcutHint>/</ShortcutHint>
        <div id="tri-feed-results" class="tri-ana-results tri-feed-results" aria-hidden="true" />
      </div>
      <span class="tri-feed-count" aria-live="polite" />
      <OnModeSwitch />
    </div>
    <div class="tri-tree-list" data-keyboard-scroll>
      {tree.map(year => (
        <section class="tri-tree-year">
          <div class="tri-tree-row tri-tree-row--year">
            <a href={`${root}/${year.slug}`}>{year.year}</a>
            <TreeSum count={year.count} km={year.km} timeS={year.timeS} />
          </div>
          {year.months.map(month => (
            <section class="tri-tree-month">
              <div class="tri-tree-row tri-tree-row--month">
                <a href={`${root}/${month.slug}`}>{`${year.year} / ${month.month}`}</a>
                <TreeSum count={month.count} km={month.km} timeS={month.timeS} />
              </div>
              <div class="tri-tree-days" role="list">
                {month.days.map(day => (
                  <a class="tri-tree-day" role="listitem" href={`${root}/${day.slug}`}>
                    <span class="tri-tree-day-d">{day.day}</span>
                    <span
                      class="tri-tree-day-sports"
                      data-i18n={day.sports.length === 0 ? 'rest' : undefined}
                    >
                      {day.sports.length === 0 ? 'rest' : day.sports.join(' · ')}
                    </span>
                    <TreeSum count={day.count} km={day.km} timeS={day.timeS} />
                  </a>
                ))}
              </div>
            </section>
          ))}
        </section>
      ))}
      {tree.length === 0 && <div class="tri-ana-empty">no activities</div>}
    </div>
    <div class="tri-feed-list" role="list" aria-busy="true" data-keyboard-scroll />
  </section>
)
