import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/image.inline"
import style from "./styles/image.scss"

export default (() => {
  const Image: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <dialog class={classNames(displayClass, "image-popup-modal")} id="image-popup-modal">
        <div class="image-popup-content">
          <button class="image-popup-close" aria-label="Close popup">
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
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
          <img class="image-popup-img" src="" alt="" />
        </div>
      </dialog>
    )
  }

  Image.css = style
  Image.afterDOMLoaded = script

  return Image
}) satisfies QuartzComponentConstructor
