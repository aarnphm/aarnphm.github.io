import katex from 'katex'
import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'
import { customMacros, katexOptions } from '../cfg'
import { filterNodes, type PageRankModel } from '../util/pagerank-view'
// @ts-ignore
import script from './scripts/pagerank.inline'
import {
  describeNode,
  emptyInbound,
  emptyOutbound,
  inspectorData,
  metaLine,
  pagerLine,
  readState,
  summaryValues,
} from './scripts/pagerank/state'
import { LinkRows, NoteLink, pageSize, RankRows, RankSizePlot } from './scripts/pagerank/views'
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
  <th
    scope="col"
    class={`pagerank-col-${key}`}
    aria-sort={key === 'position' ? 'ascending' : 'none'}
    data-pr-explain={explain[key]}
  >
    <button type="button" data-pr-sort={key} aria-label={`sort by ${name}`}>
      {label}
    </button>
  </th>
)

const defaultExplain = 'settings: 100 notes/page.'

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

interface Options {
  /** The build's ranking; the page renders its first view from it so previews show real data. */
  model?: PageRankModel
}

export default ((opts?: Options) => {
  const model = opts?.model
  // The view the script shows for an empty URL.
  const state = model && readState(model, new URLSearchParams())
  const rows = model && state ? filterNodes(model, state.query) : []
  const selected = (model && state && model.nodes[state.selected]) || null
  const panel = model && selected ? inspectorData(model, selected) : undefined
  const values = model ? summaryValues(model) : {}
  const pages = Math.max(1, Math.ceil(rows.length / pageSize))
  const loading = 'reading the link graph…'

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
            passes on an equal part of what <Tex source="q" /> holds. We normalize each score with
            the average <Tex source="x_p = N\cdot\mathrm{PR}(p)" />, therefore a{' '}
            <Tex source="1.00 \times" /> is considered an average note.{' '}
            <i>select a note to see the links that make up its score.</i>
          </p>
          <p>
            <i>
              every note gets <Tex source="(1-d) + d\sum_{L(q)=0}\mathrm{PR}(q)" />
            </i>
          </p>
          <p class="pagerank-meta" data-pr-meta>
            {model ? metaLine(model) : loading}
          </p>
        </section>

        <dl class="pagerank-readout" aria-label="corpus summary">
          {stats.map(([key, label]) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd data-pr-stat={key}>{values[key] ?? '–'}</dd>
            </div>
          ))}
        </dl>

        <figure class="pagerank-figure pagerank-field">
          <div
            class="pagerank-field-plot"
            data-pr-field
            tabindex={0}
            role="img"
            aria-label="every note as one dot on a log score axis"
          >
            <canvas aria-hidden="true" />
            <svg aria-hidden="true" />
            <div class="pagerank-field-notes" data-pr-field-notes aria-hidden="true" />
            <p class="pagerank-field-hover" data-pr-field-hover aria-hidden="true" />
          </div>
          <p
            class="pagerank-chart-readout pagerank-field-readout"
            data-pr-field-readout
            aria-live="polite"
          >
            –
          </p>
          <figcaption>
            Each dot represents a note, placed by its score <Tex source="x = N\cdot\mathrm{PR}" />{' '}
            via log-scaled axis. notes nobody links to all share the lowest score, so they wait in
            the block on the left, off the axis. <br />A solid dot denotes backlinks, a ring denotes
            forward linkage, a solid dot within a ring denotes bidirectional linkage. You can use
            arrow keys to walk between the ranking.
          </figcaption>
        </figure>

        <figure class="pagerank-figure">
          <div class="pagerank-chart" data-pr-chart>
            {model && <RankSizePlot model={model} selected={selected} pointer={null} />}
          </div>
          <figcaption>
            <p>
              The line is log-scaled, and they trace the score of every available notes within the
              garden. The dashed line <Tex source="x \propto r^{-\alpha}" /> proposed a linear
              trend.
            </p>
            <br />
            <p class="pagerank-chart-readout" data-pr-chart-readout aria-live="polite">
              {describeNode('is selected', selected)}
            </p>
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
                    {model ? (
                      <RankRows rows={rows.slice(0, pageSize)} selected={selected?.id ?? -1} />
                    ) : (
                      <tr class="pagerank-status">
                        <td colSpan={7}>{loading}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div class="pagerank-pager">
                <div class="pagerank-pages">
                  <button type="button" data-pr-page="-1" aria-label="previous page" disabled>
                    ‹
                  </button>
                  <span data-pr-page-label>1 / {pages}</span>
                  <button
                    type="button"
                    data-pr-page="1"
                    aria-label="next page"
                    disabled={pages <= 1}
                  >
                    ›
                  </button>
                </div>
                <span class="pagerank-filtered" data-pr-filtered aria-live="polite">
                  {model ? pagerLine(model, rows) : ''}
                </span>
              </div>
            </div>
          </div>

          <aside class="pagerank-inspector" aria-label="selected note" data-pr-inspector>
            <header class="pagerank-inspector-head">
              <h2 data-pr-title>{selected ? <NoteLink node={selected} /> : '–'}</h2>
              <p class="pagerank-slug" data-pr-slug>
                {selected?.slug}
              </p>
              <dl class="pagerank-facts">
                {facts.map(([key, label]) => (
                  <div key={key}>
                    <dt>
                      <Tex source={label} />
                    </dt>
                    <dd data-pr-fact={key}>{panel?.facts[key] ?? '–'}</dd>
                  </div>
                ))}
              </dl>
            </header>
            <section class="pagerank-links" aria-label="links of the selected note">
              <p class="pagerank-kicker">
                A link from <Tex source="q" /> gives <Tex source="d\,x_q / L(q)" />
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
                  <tbody data-pr-ledger-in>
                    {panel && <LinkRows links={panel.inbound} empty={emptyInbound} />}
                  </tbody>
                  <tfoot>
                    <tr>
                      <th scope="row">score</th>
                      <td data-pr-ledger-total>{panel?.total ?? '–'}</td>
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
                  <tbody data-pr-ledger-out>
                    {panel && <LinkRows links={panel.outbound} empty={emptyOutbound} />}
                  </tbody>
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
