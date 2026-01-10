import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/pageListSearch.scss"
// @ts-ignore
import script from "./scripts/pagelist-search.inline"
import { i18n } from "../i18n"

export interface PageListSearchOptions {
  placeholder?: string
}

const defaultOptions: PageListSearchOptions = {
  placeholder: "filtering...",
}

export default ((userOpts?: Partial<PageListSearchOptions>) => {
  const PageListSearch: QuartzComponent = ({ cfg, allTags }: QuartzComponentProps) => {
    const opts = { ...defaultOptions, ...userOpts }
    const placeholder = opts.placeholder ?? i18n(cfg.locale).components.search.searchBarPlaceholder

    return (
      <div class="page-list-search-container" data-all-tag={allTags ?? false}>
        <form class="page-list-search-form">
          <div class="page-list-search-input-wrapper">
            <input
              type="text"
              name="filter"
              class="page-list-search-input"
              placeholder={placeholder}
              autocomplete="off"
              aria-label="Filter list entries"
            />
            <button type="button" class="page-list-search-clear" aria-label="Clear filter">
              <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
                <path
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  d="M5 5l10 10M15 5L5 15"
                />
              </svg>
            </button>
          </div>
        </form>
        <output class="page-list-search-status" aria-live="polite" />
      </div>
    )
  }

  PageListSearch.afterDOMLoaded = script
  PageListSearch.css = style

  return PageListSearch
}) satisfies QuartzComponentConstructor
