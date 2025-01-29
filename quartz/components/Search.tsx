import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/search.scss"
// @ts-ignore
import script from "./scripts/search.inline"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

export interface SearchOptions {
  enablePreview: boolean
  includeButton: boolean
}

const defaultOptions: SearchOptions = {
  enablePreview: true,
  includeButton: true,
}

export default ((userOpts?: Partial<SearchOptions>) => {
  const Search: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const opts = { ...defaultOptions, ...userOpts }
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div class={classNames(displayClass, "search")}>
        {opts.includeButton && (
          <span
            class="search-button"
            id="search-button"
            aria-label="Toggle search mode"
            type="button"
            tabindex={-1}
          >
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
          </span>
        )}
        <search id="search-container">
          <form id="search-space">
            <div class="input-container">
              <input
                autocomplete="off"
                id="search-bar"
                name="search"
                type="text"
                aria-label={searchPlaceholder}
                placeholder={searchPlaceholder}
              />
            </div>
            <output id="search-layout" data-preview={opts.enablePreview} />
          </form>
        </search>
      </div>
    )
  }

  Search.afterDOMLoaded = script
  Search.css = style

  return Search
}) satisfies QuartzComponentConstructor
