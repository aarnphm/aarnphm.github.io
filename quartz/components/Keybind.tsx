import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

import style from "./styles/keybind.scss"
// @ts-ignore
import script from "./scripts/keybind.inline"
import { classNames } from "../util/lang"

interface Options {
  default?: string[]
  enableTooltip?: boolean
}

export const KeybindAlias = {
  "cmd+k": "recherche",
  "cmd+g": "graphique",
  // "cmd+o": "mode sombre",
  "cmd+\\": "page d'accueil",
  "cmd+.": "curseur de chat",
  "cmd+l": "projets",
  "cmd+j": "curius",
}

const defaultOptions: Options = {
  default: ["⌘ /", "⌃ /"],
  enableTooltip: true,
}

const convert = (key: string) =>
  key
    .replace("cmd", "⌘")
    .replace("ctrl", "⌃")
    .replace("alt", "⌥")
    .replace("shift", "⇧")
    .replace("+", " ")

const revert = (key: string) =>
  key
    .replace("⌘", "cmd")
    .replace("⌃", "ctrl")
    .replace("⌥", "alt")
    .replace("⇧", "shift")
    .replace(" ", "--")
    .replace("+", "--")

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }
  const defaultKey = opts.default![0]

  const Keybind: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "keybind")} lang={"fr"}>
        <kbd id="shortcut-key" data-mapping={JSON.stringify(opts.default?.map(revert))}>
          {defaultKey}
        </kbd>
        {opts.enableTooltip && (
          <div class="highlight-modal bind" id="highlight-modal">
            <span>afficher les raccourcis clavier</span>
          </div>
        )}
        <div id="shortcut-container">
          <div id="shortcut-space">
            <div id="title">raccourcis clavier</div>
            <ul>
              {Object.entries(KeybindAlias).map(([key, value]) => (
                <li>
                  <div id="shortcuts">
                    <div>
                      <kbd id="clickable-kbd" data-keybind={revert(key)}>
                        {convert(key)}
                      </kbd>
                    </div>
                    <span>{value}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    )
  }

  Keybind.css = style
  Keybind.afterDOMLoaded = script

  return Keybind
}) satisfies QuartzComponentConstructor
