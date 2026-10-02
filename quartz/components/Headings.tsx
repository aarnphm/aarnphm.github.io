import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/headings-modal.inline'
import style from './styles/headings.scss'

export default (() => {
  const Headings: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <dialog
        class={classNames(displayClass, 'headings-modal')}
        aria-labelledby="headings-modal-title"
        tabindex={-1}
      >
        <header class="headings-modal-header">
          <span id="headings-modal-title">jump to heading</span>
          <button type="button" class="headings-modal-close" aria-label="close">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              aria-hidden="true"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </header>
        <p class="headings-modal-help" role="status" />
        <div class="headings-modal-body">
          <ol class="headings-list" />
        </div>
      </dialog>
    )
  }

  Headings.afterDOMLoaded = script
  Headings.css = style

  return Headings
}) satisfies QuartzComponentConstructor
