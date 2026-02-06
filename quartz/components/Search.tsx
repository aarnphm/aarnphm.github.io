import { i18n } from '../i18n'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/search.inline'
import style from './styles/search.scss'

export interface SearchOptions {
  enablePreview: boolean
  includeButton: boolean
}

const defaultOptions: SearchOptions = { enablePreview: true, includeButton: true }

export default ((userOpts?: Partial<SearchOptions>) => {
  const Search: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const opts = { ...defaultOptions, ...userOpts }
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div class={classNames(displayClass, 'search')}>
        {opts.includeButton && (
          <button type="button" class="search-button" aria-label="Toggle search mode" tabindex={-1}>
            <svg
              role="img"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 -1 19.9 19.7"
              width={18}
              height={18}
            >
              <title>Search</title>
              <g class="search-path" fill="none">
                <path stroke-linecap="square" d="M18.5 18.3l-5.4-5.4" />
                <circle cx="8" cy="8" r="7" />
              </g>
            </svg>
          </button>
        )}
        <search class="search-container">
          <form class="search-space">
            <div class="input-container">
              <input
                autocomplete="off"
                class="search-bar"
                name="search"
                type="text"
                aria-label={searchPlaceholder}
                placeholder={searchPlaceholder}
              />
              <span class="semantic-status" data-state="loading" aria-live="polite">
                semantic not available yet
              </span>
              <div class="search-mode-toggle" role="radiogroup" aria-label="Search mode">
                <button
                  type="button"
                  class="mode-option"
                  data-mode="lexical"
                  aria-pressed="true"
                  aria-label="Full-text search"
                >
                  <svg viewBox="0 0 20 20" role="img" aria-hidden="true">
                    <g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
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
                >
                  <svg viewBox="0 0 20 20" role="img" aria-hidden="true">
                    <g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
                      <circle cx="5.2" cy="10" r="2.4" />
                      <circle cx="14.8" cy="4.8" r="2.1" />
                      <circle cx="14.8" cy="15.2" r="2.1" />
                      <path d="M7.1 8.7l5.2-2.4M7.1 11.3l5.2 2.4M14.8 6.9v6.2" />
                    </g>
                  </svg>
                  <span class="sr-only">Semantic</span>
                </button>
              </div>
            </div>
            <output class="search-layout" data-preview={opts.enablePreview} />
          </form>
        </search>
      </div>
    )
  }

  Search.afterDOMLoaded = script
  Search.css = style

  return Search
}) satisfies QuartzComponentConstructor
