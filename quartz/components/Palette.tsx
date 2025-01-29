import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/palette.inline"
import style from "./styles/palette.scss"
import { classNames } from "../util/lang"

export default (() => {
  const placeholder = "sélectionnez une option..."
  const Palette: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div class={classNames(displayClass, "palette")}>
      <search id="palette-container">
        <div id="space">
          <div class="input-container">
            <input
              autocomplete="off"
              id="bar"
              name="palette"
              type="text"
              aria-label={placeholder}
              placeholder={placeholder}
            />
          </div>
          <output id="result" />
          <ul id="helper">
            <li>
              <kbd>↑↓</kbd> to navigate
            </li>
            <li>
              <kbd>↵</kbd> to open
            </li>
            <li data-quick-open>
              <kbd>⌘ ↵</kbd> to open in a new tab
            </li>
            <li data-quick-open>
              <kbd>⌘ ⌥ ↵</kbd> to open in a side panel
            </li>
            <li>
              <kbd>esc</kbd> to dismiss
            </li>
          </ul>
        </div>
      </search>
    </div>
  )

  Palette.css = style
  Palette.afterDOMLoaded = script

  return Palette
}) satisfies QuartzComponentConstructor
