import { render } from 'preact-render-to-string'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/palette.inline'
import style from './styles/palette.scss'

const placeholder = 'sélectionnez une option...'
const paletteInnerHtml = render(
  <search id="palette-container" popover="manual">
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
        <li data-command style={{ display: 'none' }}>
          <kbd>←→</kbd> pour naviguer
        </li>
        <li>
          <kbd>esc</kbd> pour rejeter
        </li>
      </ul>
    </div>
  </search>,
)

export default (() => {
  const Palette: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div
      class={classNames(displayClass, 'palette')}
      dangerouslySetInnerHTML={{ __html: paletteInnerHtml }}
    />
  )

  Palette.css = style
  Palette.afterDOMLoaded = script

  return Palette
}) satisfies QuartzComponentConstructor
