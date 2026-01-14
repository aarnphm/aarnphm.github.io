import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
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
              <kbd>↑↓</kbd> pour naviguer
            </li>
            <li>
              <kbd>↵</kbd> pour ouvrir
            </li>
            <li data-quick-open>
              <kbd>⟶</kbd> pour sélectionner
            </li>
            <li data-quick-open>
              <kbd>⌘ ⌥ ↵</kbd> pour ouvrir dans un panneau
            </li>
            <li>
              <kbd>esc</kbd> pour rejeter
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
