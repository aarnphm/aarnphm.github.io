import katex from 'katex'
import { customMacros, katexOptions } from '../cfg'
import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'
// @ts-ignore
import script from './scripts/pagerank.inline'
import style from './styles/pagerank.scss'

const tex = (source: string, displayMode = false) =>
  katex.renderToString(source, {
    ...katexOptions,
    displayMode,
    output: 'htmlAndMathml',
    macros: customMacros,
  })

const Tex = ({ source, display }: { source: string; display?: boolean }) => (
  <span class="pagerank-tex" dangerouslySetInnerHTML={{ __html: tex(source, display) }} />
)

const stats: [key: string, label: string][] = [
  ['notes', 'notes'],
  ['links', 'links'],
  ['median', 'median'],
  ['max', 'highest'],
  ['gini', 'gini'],
  ['top10', 'top 10% hold'],
  ['noInbound', 'no inbound link'],
  ['noOutbound', 'no outbound link'],
]

const sortHeader = (key: string, label: string, title: string) => (
  <th scope="col" class={`pagerank-col-${key}`} aria-sort="none">
    <button type="button" data-pr-sort={key} title={title}>
      {label}
    </button>
  </th>
)

export default (() => {
  const PageRankContent: QuartzComponent = () => (
    <article class="popover-hint main-col pagerank-page">
      <div id="pagerank-root" class="pagerank" data-pagerank>
        <section class="pagerank-intro" aria-label="what the score means">
          <p>
            each note is scored by a random walk over the links i wrote. at each step the walker
            follows one of the current note's links with probability <Tex source="d" />, or jumps
            to any note. a note without links sends the walker to any note.
          </p>
          <Tex
            display
            source="\mathrm{PR}(p) = \frac{1-d}{N} + \frac{d}{N}\sum_{L(q)=0}\mathrm{PR}(q) + d\sum_{q \to p}\frac{\mathrm{PR}(q)}{L(q)}"
          />
          <p>
            scores read as <Tex source="x_p = N\cdot\mathrm{PR}(p)" />, times the mean: 1.00× is an
            average note. <Tex source="L(q)" /> counts the outbound links of <Tex source="q" />.
          </p>
          <p class="pagerank-meta" data-pr-meta>
            reading the link graph…
          </p>
          <p class="pagerank-aside">
            ctrl+o orders suggestions differently: it restarts the walk from the note you are
            reading and follows links in both directions, so it ranks what is near that note. this
            page ranks every note by one walk over the whole garden.
          </p>
        </section>

        <dl class="pagerank-readout" aria-label="corpus summary">
          {stats.map(([key, label]) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd data-pr-stat={key}>–</dd>
            </div>
          ))}
        </dl>

        <figure class="pagerank-figure">
          <div class="pagerank-chart" data-pr-chart />
          <figcaption>
            every note's score against its rank, both on log scales. the dashed line is a power
            law <Tex source="x \propto r^{-\alpha}" /> fitted to notes with an inbound link; the flat
            tail on the right is notes nobody links to.{' '}
            <span class="pagerank-chart-readout" data-pr-chart-readout aria-live="polite" />
          </figcaption>
        </figure>

        <div class="pagerank-toolbar" role="search">
          <input
            type="search"
            class="pagerank-search"
            data-pr-query
            placeholder="title or path"
            aria-label="filter notes by title or path"
            autocomplete="off"
            spellcheck={false}
          />
          <div class="pagerank-folder" data-pr-folder />
          <div class="pagerank-toggles" role="group" aria-label="filters">
            <button type="button" data-pr-toggle="hideCourse" aria-pressed="false">
              hide course indexes
            </button>
            <button type="button" data-pr-toggle="linked" aria-pressed="false">
              linked only
            </button>
          </div>
        </div>

        <div class="pagerank-workspace">
          <div class="pagerank-list">
            <div class="pagerank-frame">
              <div class="pagerank-scroll" tabindex={0} role="region" aria-label="global ranking">
                <table class="pagerank-table">
                  <thead>
                    <tr>
                      {sortHeader('position', '#', 'global position')}
                      <th scope="col" class="pagerank-col-note">
                        note
                      </th>
                      <th scope="col" class="pagerank-col-times">
                        ×mean
                      </th>
                      <th scope="col" class="pagerank-col-axis">
                        log axis
                      </th>
                      {sortHeader('inbound', 'in', 'inbound links')}
                      {sortHeader('outbound', 'out', 'outbound links')}
                      {sortHeader('delta', 'Δ in', 'rank by inbound count minus rank by pagerank')}
                    </tr>
                  </thead>
                  <tbody data-pr-rows>
                    <tr class="pagerank-status">
                      <td colSpan={7}>reading the link graph…</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div class="pagerank-pager">
                <div class="pagerank-pages">
                  <button type="button" data-pr-page="-1" aria-label="previous page" disabled>
                    ‹
                  </button>
                  <span data-pr-page-label>1 / 1</span>
                  <button type="button" data-pr-page="1" aria-label="next page" disabled>
                    ›
                  </button>
                </div>
                <span class="pagerank-filtered" data-pr-filtered aria-live="polite" />
              </div>
            </div>
            <p class="pagerank-legend">
              100 rows a page. # is the global position; filters never renumber it. the log axis
              column puts every note on one scale, so its marks are the distribution. Δ in is rank by
              inbound count minus rank by pagerank: positive when well-ranked sources lift a note.
            </p>
          </div>

          <aside class="pagerank-inspector" aria-label="selected note" data-pr-inspector>
            <header class="pagerank-inspector-head">
              <p class="pagerank-kicker">selected note</p>
              <h2 data-pr-title>–</h2>
              <p class="pagerank-slug" data-pr-slug />
              <dl class="pagerank-facts" data-pr-facts />
            </header>
            <section class="pagerank-hood" aria-label="authored links">
              <p class="pagerank-kicker">links</p>
              <div class="pagerank-hood-graph" data-pr-hood />
              <p class="pagerank-hood-caption" data-pr-hood-caption />
            </section>
            <section class="pagerank-ledger" aria-label="where the score comes from">
              <p class="pagerank-kicker">
                ledger: each inbound link gives <Tex source="d\,x_q / L(q)" />
              </p>
              <div class="pagerank-ledger-scroll" tabindex={0} role="region" aria-label="ledger">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">inbound from</th>
                      <th scope="col">gives</th>
                      <th scope="col">share</th>
                    </tr>
                  </thead>
                  <tbody data-pr-ledger-in />
                  <tfoot>
                    <tr>
                      <th scope="row">
                        every note <Tex source="(1-d) + d\sum_{L(q)=0}\mathrm{PR}(q)" />
                      </th>
                      <td data-pr-ledger-base>–</td>
                      <td data-pr-ledger-base-share>–</td>
                    </tr>
                    <tr>
                      <th scope="row">total, the score</th>
                      <td data-pr-ledger-total>–</td>
                      <td>100%</td>
                    </tr>
                  </tfoot>
                </table>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">outbound to</th>
                      <th scope="col">gives</th>
                      <th scope="col">of its score</th>
                    </tr>
                  </thead>
                  <tbody data-pr-ledger-out />
                </table>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </article>
  )
  PageRankContent.css = style
  PageRankContent.afterDOMLoaded = script
  return PageRankContent
}) satisfies QuartzComponentConstructor
