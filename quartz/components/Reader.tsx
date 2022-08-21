import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import style from "./styles/reader.scss"
// @ts-ignore
import readerScript from "./scripts/reader.inline"

export default (() => {
  const Reader: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "reader")} id="reader-view">
        <div class="reader-backdrop" />
        <div class="reader-container">
          <div class="reader-header">
            <button class="reader-close" aria-label="Close reader">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div class="reader-content" />
        </div>
      </div>
    )
  }
  Reader.css = style
  Reader.afterDOMLoaded = readerScript

  return Reader
}) satisfies QuartzComponentConstructor
