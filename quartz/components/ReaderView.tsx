import { QuartzComponent, QuartzComponentConstructor } from "./types"
import style from "./styles/readerview.scss"
// @ts-ignore
import script from "./scripts/readerview.inline"

export default (() => {
  const ReaderView: QuartzComponent = () => {
    return (
      <span
        id="reader-view-toggle"
        title="Toggle reader view"
        aria-label="Toggle reader view"
        type="button"
        role="switch"
        aria-checked="false"
      >
        <div class="view-toggle-slide"></div>
        <div class="view-toggle-switch">
          <svg
            class="single-view-icon"
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <circle cx="12" cy="12" r="8" />
          </svg>
          <svg
            class="stacked-view-icon"
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <circle cx="14" cy="12" r="6" opacity="1" />
            <circle cx="10" cy="12" r="6" opacity="0.5" />
          </svg>
        </div>
      </span>
    )
  }

  ReaderView.css = style
  ReaderView.afterDOMLoaded = script

  return ReaderView
}) satisfies QuartzComponentConstructor
