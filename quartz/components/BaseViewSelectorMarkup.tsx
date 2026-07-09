import type { ComponentChild } from 'preact'
import type { QuartzComponentProps } from '../types/component'
import { classNames } from '../util/lang'
import { resolveRelative } from '../util/path'

const icons = {
  table: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M3 9h18" />
      <path d="M3 15h18" />
      <path d="M9 3v18" />
      <path d="M15 3v18" />
    </svg>
  ),
  list: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <line x1="8" x2="21" y1="6" y2="6" />
      <line x1="8" x2="21" y1="12" y2="12" />
      <line x1="8" x2="21" y1="18" y2="18" />
      <line x1="3" x2="3.01" y1="6" y2="6" />
      <line x1="3" x2="3.01" y1="12" y2="12" />
      <line x1="3" x2="3.01" y1="18" y2="18" />
    </svg>
  ),
  chevronsUpDown: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="m7 15 5 5 5-5" />
      <path d="m7 9 5-5 5 5" />
    </svg>
  ),
  chevronRight: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  ),
  search: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  ),
  x: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  ),
  map: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
    </svg>
  ),
  card: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <rect width="7" height="7" x="3" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="14" rx="1" />
      <rect width="7" height="7" x="3" y="14" rx="1" />
    </svg>
  ),
}

const viewTypeIcons: Partial<Record<BaseViewMetadata['type'], ComponentChild>> = {
  table: icons.table,
  list: icons.list,
  gallery: icons.card,
  board: icons.table,
  calendar: icons.table,
  map: icons.map,
  card: icons.card,
}

const iconForView = (type: BaseViewMetadata['type']): ComponentChild =>
  viewTypeIcons[type] ?? icons.table

export const BaseViewSelectorMarkup = ({
  fileData,
  displayClass,
}: Pick<QuartzComponentProps, 'fileData' | 'displayClass'>) => {
  const baseMeta = fileData.basesMetadata
  const slug = fileData.slug

  if (!baseMeta || !baseMeta.allViews || baseMeta.allViews.length <= 1 || !slug) {
    return null
  }

  const currentViewName = baseMeta.currentView
  const allViews = baseMeta.allViews
  const currentView = allViews.find(v => v.name === currentViewName)
  const currentIcon = currentView ? iconForView(currentView.type) : icons.table

  return (
    <div class={classNames(displayClass, 'bases-toolbar')} data-base-view-selector>
      <div class="bases-toolbar-item bases-toolbar-views-menu">
        <button
          type="button"
          class="text-icon-button"
          aria-label="Select view"
          aria-expanded="false"
        >
          <span class="text-button-icon">{currentIcon}</span>
          <span class="text-button-label">{currentViewName?.toLowerCase()}</span>
          <span class="text-button-icon mod-aux">{icons.chevronsUpDown}</span>
        </button>
      </div>

      <div class="menu-scroll" data-dropdown hidden>
        <div class="bases-toolbar-menu-container">
          <div class="search-input-container">
            <input
              type="search"
              placeholder="Search..."
              aria-label="Search views"
              data-search-input
            />
            <button
              type="button"
              class="search-input-clear-button"
              aria-label="Clear view search"
              data-clear-search
              hidden
            >
              {icons.x}
            </button>
          </div>
          <div class="bases-toolbar-items">
            <div class="suggestion-group" data-group="views" data-view-list>
              {allViews.map(view => {
                const isActive = view.name === currentViewName
                const icon = iconForView(view.type)
                const href = resolveRelative(slug, view.slug)

                return (
                  <a
                    href={href}
                    data-slug={view.slug}
                    class={
                      isActive
                        ? 'suggestion-item bases-toolbar-menu-item mod-active is-selected'
                        : 'suggestion-item bases-toolbar-menu-item'
                    }
                    data-view-name={view.name}
                    data-view-type={view.type}
                    aria-current={isActive ? 'page' : undefined}
                  >
                    <div class="bases-toolbar-menu-item-info">
                      <div class="bases-toolbar-menu-item-info-icon">{icon}</div>
                      <div class="bases-toolbar-menu-item-name">{view.name.toLowerCase()}</div>
                    </div>
                    <div class="clickable-icon bases-toolbar-menu-item-icon">
                      {icons.chevronRight}
                    </div>
                  </a>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
