import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import MetaConstructor from "../Meta"
import NavigationConstructor from "../Navigation"

import style from "../styles/curiusPage.scss"

//@ts-ignore
import script from "../scripts/curius.inline"
import { i18n } from "../../i18n"

const Title = () => (
  <div class="curius-title">
    <span>
      Voir plus sur{" "}
      <a href="https://curius.app/aaron-pham" target="_blank">
        curius.app/aaron-pham
      </a>
    </span>
    <svg
      id="curius-refetch"
      aria-labelledby="refetch"
      data-tooltip="refresh"
      data-id="refetch"
      height="12"
      type="button"
      viewBox="0 -4 24 24"
      width="12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.31-3.67.37-6.69 3.35-7.1 7.02C3.52 15.91 7.27 20 12 20c3.19 0 5.93-1.87 7.21-4.56.32-.67-.16-1.44-.9-1.44-.37 0-.72.2-.88.53-1.13 2.43-3.84 3.97-6.8 3.31-2.22-.49-4.01-2.3-4.48-4.52C5.31 9.44 8.26 6 12 6c1.66 0 3.14.69 4.22 1.78l-1.51 1.51c-.63.63-.19 1.71.7 1.71H19c.55 0 1-.45 1-1V6.41c0-.89-1.08-1.34-1.71-.71z"></path>
    </svg>
    <span class="total-links"></span>
  </div>
)

const ContainerConstructor = (() => {
  function Container({ cfg }: QuartzComponentProps) {
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div id="curius-container">
        <div class="curius-search">
          <input
            id="curius-bar"
            type="text"
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
          />
          <div id="curius-search-container"></div>
        </div>
        <div id="curius-fetching-text"></div>
        <div id="curius-fragments"></div>
        <div class="highlight-modal" id="highlight-modal">
          <ul id="highlight-modal-list"></ul>
        </div>
      </div>
    )
  }

  return Container
}) satisfies QuartzComponentConstructor

function CuriusContent(componentData: QuartzComponentProps) {
  const Meta = MetaConstructor({ enableSearch: false })
  const CuriusContainer = ContainerConstructor()
  const Navigation = NavigationConstructor({ prev: "/quotes", next: "/books" })

  return (
    <div class="popover-hint">
      <Meta {...componentData} />
      <div id="curius">
        <Title />
        <CuriusContainer {...componentData} />
        <div id="curius-notes">
          <div class="curius-note-title">
            <a id="note-link"></a>
            <div class="icon-container">
              <svg
                id="note-close"
                aria-labelledby="close"
                data-tooltip="close"
                data-id="close"
                height="15"
                type="button"
                viewBox="0 0 24 24"
                width="15"
                fill="rgba(170, 170, 170, 0.5)"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path>
              </svg>
            </div>
          </div>
          <div class="curius-note-snippet"></div>
          <div class="curius-note-highlights"></div>
        </div>
      </div>
      <Navigation {...componentData} />
    </div>
  )
}

CuriusContent.css = style
CuriusContent.afterDOMLoaded = script

export default (() => CuriusContent) satisfies QuartzComponentConstructor
