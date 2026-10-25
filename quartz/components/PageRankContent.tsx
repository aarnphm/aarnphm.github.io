import katex from 'katex'
import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'
import { customMacros, katexOptions } from '../cfg'
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

// What each column measures; shown in the line above the table on hover or focus.
const explain = {
  position: 'place in the full ranking by score. filters keep this number.',
  note: 'the note title, with its path under it.',
  times: 'score divided by the average score. 1.00× is an average note.',
  axis: 'score on one log scale for all notes. together the marks show how scores spread.',
  inbound: 'how many other notes link to this note.',
  outbound: 'how many notes this note links to.',
  delta:
    'rank by inbound link count minus rank by score. above zero, links from high-ranked notes lift it.',
}

const header = (key: keyof typeof explain, label: preact.ComponentChildren) => (
  <th scope="col" class={`pagerank-col-${key}`} data-pr-explain={explain[key]} tabindex={0}>
    {label}
  </th>
)

const sortHeader = (key: keyof typeof explain, label: preact.ComponentChildren, name: string) => (
  <th scope="col" class={`pagerank-col-${key}`} aria-sort="none" data-pr-explain={explain[key]}>
    <button type="button" data-pr-sort={key} aria-label={`sort by ${name}`}>
      {label}
    </button>
  </th>
)

const defaultExplain = '100 notes a page. point at a column heading to see what it measures.'

const facts: [key: string, label: string][] = [
  ['rank', '\\#'],
  ['times', '\\times\\,\\text{mean}'],
  ['links', '\\text{in}\\cdot\\text{out}'],
  ['delta', '\\Delta\\,\\text{in}'],
  ['source', '\\text{top source}'],
]

const LinkIcon = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
    <path d="M6.5 9.5l3-3M7 4.5l1-1a2.5 2.5 0 013.5 3.5l-1 1M9 11.5l-1 1a2.5 2.5 0 01-3.5-3.5l1-1" />
  </svg>
)

export default (() => {
  const PageRankContent: QuartzComponent = ({ fileData }) => (
    <article class="popover-hint pagerank-page">
      <div id="pagerank-root" class="pagerank" data-pagerank>
        <section class="pagerank-intro" aria-label="what the score means">
          <h1 class="pagerank-title">{fileData.frontmatter?.title ?? 'pagerank'}</h1>
          {fileData.frontmatter?.description && (
            <p class="pagerank-description">{fileData.frontmatter.description}</p>
          )}
          <p>
            picture a reader who moves through the garden at random. at each step they follow one of
            the links on the current note with probability <Tex source="d" />, or they jump to any
            note. a note with no links always sends them to any note. a note's score is the part of
            a long walk that the reader spends on it.
          </p>
          <Tex
            display
            source="\mathrm{PR}(p) = \frac{1-d}{N} + \frac{d}{N}\sum_{L(q)=0}\mathrm{PR}(q) + d\sum_{q \to p}\frac{\mathrm{PR}(q)}{L(q)}"
          />
          <p>
            <Tex source="L(q)" /> is the number of links on note <Tex source="q" />, so each link
            passes on an equal part of what <Tex source="q" /> holds. the table divides each score
            by the average, <Tex source="x_p = N\cdot\mathrm{PR}(p)" />, so 1.00× is an average
            note. select a note to see the links that make up its score.
          </p>
          <p class="pagerank-meta" data-pr-meta>
            reading the link graph…
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
            <p>
              each mark is one note, placed by its rank and its score on log scales. a few notes
              hold most of the score, and the dashed line <Tex source="x \propto r^{-\alpha}" />{' '}
              follows them down. the flat run on the right is the notes that nothing links to.
            </p>
            <p class="pagerank-chart-readout" data-pr-chart-readout aria-live="polite" />
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
            <button type="button" data-pr-toggle="courses" aria-pressed="false">
              courses
            </button>
            <button
              type="button"
              class="pagerank-icon-toggle"
              data-pr-toggle="linked"
              aria-pressed="false"
              aria-label="only notes with an inbound link"
            >
              <LinkIcon />
            </button>
          </div>
        </div>

        <div class="pagerank-workspace">
          <div class="pagerank-list">
            <div class="pagerank-frame">
              <p class="pagerank-explain" data-pr-explain-line>
                {defaultExplain}
              </p>
              <div class="pagerank-scroll" tabindex={0} role="region" aria-label="global ranking">
                <table class="pagerank-table">
                  <thead>
                    <tr>
                      {sortHeader('position', <Tex source="\#" />, 'global position')}
                      {header('note', 'note')}
                      {header('times', <Tex source="\times\,\text{mean}" />)}
                      {header('axis', 'log axis')}
                      {sortHeader('inbound', 'in', 'inbound links')}
                      {sortHeader('outbound', 'out', 'outbound links')}
                      {sortHeader('delta', <Tex source="\Delta\,\text{in}" />, 'rank change')}
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
          </div>

          <aside class="pagerank-inspector" aria-label="selected note" data-pr-inspector>
            <header class="pagerank-inspector-head">
              <p class="pagerank-kicker">selected note</p>
              <h2 data-pr-title>–</h2>
              <p class="pagerank-slug" data-pr-slug />
              <dl class="pagerank-facts">
                {facts.map(([key, label]) => (
                  <div key={key}>
                    <dt>
                      <Tex source={label} />
                    </dt>
                    <dd data-pr-fact={key}>–</dd>
                  </div>
                ))}
              </dl>
            </header>
            <section class="pagerank-links" aria-label="links of the selected note">
              <p class="pagerank-kicker">
                links in, largest first. a link from <Tex source="q" /> gives{' '}
                <Tex source="d\,x_q / L(q)" />
              </p>
              <div class="pagerank-ledger-scroll" tabindex={0} role="region" aria-label="links">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">from</th>
                      <th scope="col" title="the part of the score this link gives, times the mean">
                        gives
                      </th>
                      <th scope="col" title="the part of this note's score">
                        share
                      </th>
                    </tr>
                  </thead>
                  <tbody data-pr-ledger-in />
                  <tfoot>
                    <tr>
                      <th scope="row">
                        every note gets <Tex source="(1-d) + d\sum_{L(q)=0}\mathrm{PR}(q)" />
                      </th>
                      <td data-pr-ledger-base>–</td>
                      <td data-pr-ledger-base-share>–</td>
                    </tr>
                    <tr>
                      <th scope="row">score</th>
                      <td data-pr-ledger-total>–</td>
                      <td>100%</td>
                    </tr>
                  </tfoot>
                </table>
                <p class="pagerank-kicker">links out</p>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">to</th>
                      <th scope="col" title="the part of the score this link gives, times the mean">
                        gives
                      </th>
                      <th scope="col" title="the part of the target note's score">
                        share
                      </th>
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
