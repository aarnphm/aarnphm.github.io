import { i18n } from '../../i18n'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { EXTERNAL_LINK_ICON_PATH } from '../../util/external-link-icon'
import { classNames } from '../../util/lang'
//@ts-ignore
import curiusFriendScript from '../scripts/curius-friends.inline'
//@ts-ignore
import curiusScript from '../scripts/curius.inline'

// Curius components
export const CuriusContent: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder

  return (
    <>
      <div class={classNames(displayClass, 'curius', 'curius-col')} id="curius">
        <div class="curius-page-container">
          <header class="curius-header">
            <div class="curius-toolbar">
              <div class="curius-search">
                <input
                  id="curius-bar"
                  type="search"
                  aria-label={searchPlaceholder}
                  aria-keyshortcuts="Meta+K Control+K"
                  placeholder={searchPlaceholder}
                  autoComplete="off"
                />
                <svg
                  class="curius-search-icon"
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M10 6.5C10 8.433 8.433 10 6.5 10C4.567 10 3 8.433 3 6.5C3 4.567 4.567 3 6.5 3C8.433 3 10 4.567 10 6.5ZM9.30884 10.0159C8.53901 10.6318 7.56251 11 6.5 11C4.01472 11 2 8.98528 2 6.5C2 4.01472 4.01472 2 6.5 2C8.98528 2 11 4.01472 11 6.5C11 7.56251 10.6318 8.53901 10.0159 9.30884L12.8536 12.1464C13.0488 12.3417 13.0488 12.6583 12.8536 12.8536C12.6583 13.0488 12.3417 13.0488 12.1464 12.8536L9.30884 10.0159Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
                <div id="curius-search-container" />
              </div>
              <CuriusNavigation {...props} />
              <button
                type="button"
                id="curius-preview-toggle"
                class="curius-icon-button"
                aria-label="Open preview"
                aria-controls="curius-preview"
                aria-expanded="false"
                title="Toggle preview"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.5"
                  aria-hidden="true"
                >
                  <rect x="3" y="4" width="18" height="16" rx="2" />
                  <path d="M14 4v16" />
                </svg>
              </button>
            </div>
          </header>
          <ul id="curius-fragments" aria-label="Saved links" aria-busy="true" />
          <div class="highlight-modal" id="highlight-modal">
            <ul id="highlight-modal-list" />
          </div>
        </div>
      </div>
      <button
        type="button"
        id="curius-preview-backdrop"
        aria-label="Close preview"
        tabIndex={-1}
        hidden
      />
      <aside
        id="curius-preview"
        class="curius-preview"
        role="dialog"
        aria-modal="false"
        aria-labelledby="curius-preview-title"
        tabIndex={-1}
        hidden
      >
        <header class="curius-preview-header">
          <h2 id="curius-preview-title">preview</h2>
          <button
            type="button"
            id="curius-preview-close"
            class="curius-icon-button"
            aria-label="Close preview"
            title="Close preview (Esc)"
          >
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
              <path
                d="M11.7816 4.03157C12.0062 3.80702 12.0062 3.44295 11.7816 3.2184C11.5571 2.99385 11.193 2.99385 10.9685 3.2184L7.50005 6.68682L4.03164 3.2184C3.80708 2.99385 3.44301 2.99385 3.21846 3.2184C2.99391 3.44295 2.99391 3.80702 3.21846 4.03157L6.68688 7.49999L3.21846 10.9684C2.99391 11.193 2.99391 11.557 3.21846 11.7816C3.44301 12.0061 3.80708 12.0061 4.03164 11.7816L7.50005 8.31316L10.9685 11.7816C11.193 12.0061 11.5571 12.0061 11.7816 11.7816C12.0062 11.557 12.0062 11.193 11.7816 10.9684L8.31322 7.49999L11.7816 4.03157Z"
                fill="currentColor"
                fill-rule="evenodd"
                clip-rule="evenodd"
              />
            </svg>
          </button>
        </header>
        <div class="curius-preview-heading">
          <a
            id="curius-preview-source"
            class="curius-preview-source"
            target="_blank"
            rel="noopener noreferrer"
            data-no-popover
            data-skip-icons
          />
          <div class="curius-preview-actions">
            <button
              type="button"
              id="curius-preview-retry"
              class="curius-icon-button"
              aria-label="Retry preview"
              title="Retry preview"
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 15 15"
                fill="none"
                aria-hidden="true"
                focusable="false"
              >
                <path
                  d="M7.50037 0.850006C10.6644 0.850189 12.2943 3.06869 12.9994 4.31094L13.0004 4.31192V2.5004C13.0004 2.22425 13.2242 2.0004 13.5004 2.0004C13.7763 2.00059 14.0004 2.22438 14.0004 2.5004V5.5004C14.0002 5.77625 13.7762 6.0002 13.5004 6.0004H10.5004C10.2243 6.0004 10.0006 5.77637 10.0004 5.5004C10.0004 5.22425 10.2242 5.0004 10.5004 5.0004H12.2328L12.1215 4.79239C11.4802 3.66597 10.1107 1.85019 7.50037 1.85001C4.06019 1.85001 1.84998 4.665 1.84998 7.5004C1.85018 10.3357 4.06034 13.1498 7.50037 13.1498C9.16525 13.1497 10.5296 12.496 11.5013 11.5072L11.6927 11.3031C12.126 10.8159 12.4715 10.2575 12.7172 9.66055L12.765 9.57071C12.8948 9.37795 13.1462 9.2963 13.3695 9.38809C13.6248 9.49314 13.7468 9.78515 13.642 10.0404L13.5111 10.3373C13.2362 10.9247 12.877 11.4767 12.4398 11.9682L12.2142 12.2084C11.062 13.3807 9.44396 14.1497 7.50037 14.1498C3.43771 14.1498 0.850179 10.8149 0.849976 7.5004C0.849976 4.1858 3.43755 0.850006 7.50037 0.850006Z"
                  fill="currentColor"
                />
              </svg>
            </button>
            <a
              id="curius-preview-original"
              class="curius-icon-button"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open original in a new tab"
              title="Open original in a new tab"
              data-no-popover
              data-skip-icons
              hidden
            >
              <svg viewBox="0 0 15 15" fill="none" aria-hidden="true" focusable="false">
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d={EXTERNAL_LINK_ICON_PATH}
                  fill="currentColor"
                />
              </svg>
            </a>
          </div>
        </div>
        <div id="curius-preview-content" class="curius-preview-content" />
      </aside>
    </>
  )
}
CuriusContent.afterDOMLoaded = curiusScript

export const CuriusFriends: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  return (
    <div class={classNames(displayClass, 'curius-friends')} aria-busy="true">
      <h2>mes amis</h2>
      <p id="curius-friends-status" class="curius-sidebar-status" role="status">
        Chargement des amis…
      </p>
      <ul id="friends-list" />
      <button
        type="button"
        id="see-more-friends"
        aria-expanded="false"
        aria-controls="friends-list"
        hidden
      >
        <span>
          Voir <span id="more">de plus</span>
        </span>
        <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
          <path
            d="m3.5 5.5 4 4 4-4"
            stroke="currentColor"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
    </div>
  )
}
CuriusFriends.afterDOMLoaded = curiusFriendScript

export const CuriusTrail: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  return (
    <>
      <div
        class={classNames(displayClass, 'curius-trail')}
        aria-busy="true"
        data-num-trails={3}
        data-limits={4}
        data-locale={cfg.locale}
      >
        <h2>sentiers</h2>
        <p id="curius-trails-status" class="curius-sidebar-status" role="status">
          Chargement des sentiers…
        </p>
        <ul id="trail-list" />
      </div>
      <a
        class="curius-profile"
        hidden
        href="https://curius.app/aaron-pham"
        target="_blank"
        rel="noopener noreferrer"
        data-no-popover
        data-skip-icons
      >
        curius.app/aaron-pham ↗
      </a>
    </>
  )
}

export const CuriusNavigation: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  return (
    <nav
      class={classNames(displayClass, 'curius-pagination')}
      id="curius-pagination"
      aria-label="Saved links pages"
    >
      <button
        type="button"
        id="curius-prev"
        class="curius-icon-button"
        aria-label="Previous page"
        title="Previous page"
        disabled
      >
        <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
          <path
            d="M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z"
            fill="currentColor"
            fill-rule="evenodd"
            clip-rule="evenodd"
          />
        </svg>
      </button>
      <button
        type="button"
        id="curius-next"
        class="curius-icon-button"
        aria-label="Next page"
        title="Next page"
        disabled
      >
        <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
          <path
            d="M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z"
            fill="currentColor"
            fill-rule="evenodd"
            clip-rule="evenodd"
          />
        </svg>
      </button>
    </nav>
  )
}
