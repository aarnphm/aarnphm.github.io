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
  "cmd+/": "recherche",
  "cmd+\\": "page d'accueil",
  "cmd+j": "curius",
  "cmd+b": "lecteur",
  "cmd+g": "graphique",
  // "cmd+o": "mode sombre",
  // "cmd+.": "curseur de chat",
  // "cmd+l": "projets",
}

const defaultOptions: Options = {
  default: ["⌘ '", "⌃ '"],
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
            <ul id="shortcut-list">
              {Object.entries(KeybindAlias).map(([key, value]) => (
                <li>
                  <div
                    id="shortcuts"
                    data-key={convert(key).replace(" ", "--")}
                    data-value={value}
                  ></div>
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
