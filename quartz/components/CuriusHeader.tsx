import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/curiusHeader.scss"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"

export default (() => {
  const CuriusHeader: QuartzComponent = ({ cfg, displayClass }: QuartzComponentProps) => {
    const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
    return (
      <div class={classNames(displayClass, "curius-header")}>
        <div class="curius-search">
          <input
            id="curius-bar"
            type="text"
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
          />
          <div id="curius-search-container"></div>
        </div>
        <div class="curius-title">
          <em>
            Voir de plus{" "}
            <a href="https://curius.app/aaron-pham" target="_blank">
              curius.app/aaron-pham
            </a>
          </em>
          <svg
            id="curius-refetch"
            aria-labelledby="refetch"
            data-tooltip="refresh"
            data-id="refetch"
            height="12"
            type="button"
            viewBox="0 0 24 24"
            width="12"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.31-3.67.37-6.69 3.35-7.1 7.02C3.52 15.91 7.27 20 12 20c3.19 0 5.93-1.87 7.21-4.56.32-.67-.16-1.44-.9-1.44-.37 0-.72.2-.88.53-1.13 2.43-3.84 3.97-6.8 3.31-2.22-.49-4.01-2.3-4.48-4.52C5.31 9.44 8.26 6 12 6c1.66 0 3.14.69 4.22 1.78l-1.51 1.51c-.63.63-.19 1.71.7 1.71H19c.55 0 1-.45 1-1V6.41c0-.89-1.08-1.34-1.71-.71z"></path>
          </svg>
        </div>
      </div>
    )
  }

  CuriusHeader.css = style

  return CuriusHeader
}) satisfies QuartzComponentConstructor