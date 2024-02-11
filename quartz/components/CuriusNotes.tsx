import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/curiusNotes.scss"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"

export default (() => {
  function CuriusNotes({ displayClass }: QuartzComponentProps) {
    return (
      <div class={classNames(displayClass, "curius-notes")}>
        <div class="curius-note-title">
          <a id="note-link"></a>
          <div class="icon-container">
            <svg
              id="curius-close"
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
        <ul class="curius-note-highlights"></ul>
        <div class="curius-note-snippet"></div>
      </div>
    )
  }

  CuriusNotes.css = style

  return CuriusNotes
}) satisfies QuartzComponentConstructor
