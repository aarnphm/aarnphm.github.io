import { QuartzComponentConstructor, QuartzComponentProps } from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/base-search.inline'
import style from './styles/search.scss'

function BaseSearchBar({ displayClass, fileData }: QuartzComponentProps) {
  if (!fileData.bases) {
    return null
  }
  const semanticManifest = fileData.basesSemantic?.manifest
  return (
    <div class={classNames(displayClass, 'search', 'bases-search')}>
      <div id="base-search-bar" data-semantic-manifest={semanticManifest}>
        <input
          type="text"
          id="base-search-input"
          placeholder="Search this base"
          aria-label="Search this base"
          autocomplete="off"
        />
        {semanticManifest && (
          <>
            <span class="semantic-status sr-only" data-state="idle" aria-live="polite"></span>
            <div class="search-mode-toggle" role="radiogroup" aria-label="Search mode">
              <button
                type="button"
                class="mode-option active"
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
          </>
        )}
      </div>
    </div>
  )
}

BaseSearchBar.css = style
BaseSearchBar.afterDOMLoaded = script

export default (() => BaseSearchBar) satisfies QuartzComponentConstructor
