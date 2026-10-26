import { QuartzComponentConstructor, QuartzComponentProps } from '../types/component'
import { createBaseQueryDocument } from '../util/base/browser-data'
import { classNames } from '../util/lang'
// @ts-ignore
import queryScript from './scripts/base-query.inline'
// @ts-ignore
import script from './scripts/base-search.inline'
import queryStyle from './styles/baseQuery.scss'
import style from './styles/search.scss'

function BaseSearchBar({ displayClass, fileData }: QuartzComponentProps) {
  if (!fileData.bases) {
    return null
  }
  const semanticManifest = fileData.basesSemantic?.manifest
  const queryDocument = fileData.basesQueryDocument ?? createBaseQueryDocument(fileData)
  return (
    <div
      class={classNames(displayClass, 'search', 'bases-search')}
      data-base-query={queryDocument ? JSON.stringify(queryDocument) : undefined}
      data-base-query-view={fileData.basesMetadata?.currentView}
    >
      <div class="base-query-toolbar">
        <div class="base-query-actions">
          <div
            id="base-search-bar"
            data-semantic-manifest={semanticManifest}
            data-base-geo={fileData.basesGeo ? JSON.stringify(fileData.basesGeo) : undefined}
          >
            <span class="base-search-label" aria-hidden="true">
              search
            </span>
            <input
              type="text"
              id="base-search-input"
              placeholder="Search this base"
              aria-label="Search this base"
              autocomplete="off"
            />
            {semanticManifest && (
              <>
                <div class="search-mode-toggle" role="radiogroup" aria-label="Search mode">
                  <button
                    type="button"
                    class="mode-option active"
                    data-mode="lexical"
                    aria-pressed="true"
                    aria-label="Full-text search"
                    aria-keyshortcuts="Alt+Shift+S"
                  >
                    <svg viewBox="0 0 20 20" role="img" aria-hidden="true">
                      <g
                        fill="none"
                        stroke="currentColor"
                        stroke-width="1.5"
                        stroke-linecap="round"
                      >
                        <path d="M4 6h12M4 10h8M4 14h6" />
                      </g>
                    </svg>
                    <span class="sr-only">Full-text</span>
                  </button>
                  <button
                    type="button"
                    class="mode-option"
                    data-mode="semantic"
                    aria-pressed="false"
                    aria-label="Semantic search"
                    aria-keyshortcuts="Alt+Shift+S"
                  >
                    <svg viewBox="0 0 20 20" role="img" aria-hidden="true">
                      <g
                        fill="none"
                        stroke="currentColor"
                        stroke-width="1.5"
                        stroke-linecap="round"
                      >
                        <circle cx="5.2" cy="10" r="2.4" />
                        <circle cx="14.8" cy="4.8" r="2.1" />
                        <circle cx="14.8" cy="15.2" r="2.1" />
                        <path d="M7.1 8.7l5.2-2.4M7.1 11.3l5.2 2.4M14.8 6.9v6.2" />
                      </g>
                    </svg>
                    <span class="sr-only">Semantic</span>
                  </button>
                </div>
              </>
            )}
          </div>
          {queryDocument && (
            <div
              class="base-query-tabs"
              role="tablist"
              aria-label="Edit base view"
              aria-orientation="horizontal"
            >
              {['filter', 'sort'].map(mode => (
                <button
                  id={`base-query-tab-${mode}`}
                  type="button"
                  role="tab"
                  data-base-query-toggle={mode}
                  aria-selected={mode === 'filter'}
                  aria-expanded="false"
                  aria-controls="base-query-panel"
                  tabIndex={mode === 'filter' ? 0 : -1}
                >
                  {mode}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      {queryDocument && (
        <div
          id="base-query-panel"
          class="base-query-panel"
          role="tabpanel"
          aria-labelledby="base-query-tab-filter"
          hidden
        >
          <div class="base-query-panel-heading">
            <span data-base-query-heading>Filter this view</span>
            <button type="button" data-base-query-close aria-label="Close base editor">
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path d="m4 4 8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="1.5" />
              </svg>
            </button>
          </div>
          <p class="base-query-status" data-base-query-status role="status" aria-live="polite"></p>
          <div data-base-query-surface></div>
          <p class="base-query-error" data-base-query-error role="alert" hidden></p>
          <div class="base-query-footer">
            <a
              class="base-query-syntax"
              href="https://obsidian.md/help/bases/syntax"
              target="_blank"
              rel="noopener noreferrer"
            >
              Bases syntax
            </a>
            <div class="base-query-footer-actions">
              <button
                type="button"
                data-base-query-download
                aria-label="Download .base"
                title="Download .base"
                disabled
              >
                <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                  <path
                    d="M8 2v8m-3-3 3 3 3-3M3 10v3h10v-3"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </button>
              <button type="button" data-base-query-reset disabled>
                reset
              </button>
              <button type="button" data-base-query-apply disabled>
                apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

BaseSearchBar.css = style + queryStyle
BaseSearchBar.afterDOMLoaded = [script, queryScript]

export default (() => BaseSearchBar) satisfies QuartzComponentConstructor
