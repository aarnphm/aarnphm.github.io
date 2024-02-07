import { i18n } from "../i18n"
import Darkmode from "./Darkmode"
import { QuartzComponentConstructor, QuartzComponentProps } from "./types"

import style from "./styles/keybind.scss"
// @ts-ignore
import script from "./scripts/keybind.inline"
import { classNames } from "../util/lang"

interface Options {
  default?: string[]
}

export const KeybindAlias = {
  "cmd+k": "recherche",
  "cmd+g": "graphique",
  "cmd+o": "mode sombre",
  "cmd+\\": "page d'accueil",
  "/": "curseur de chat",
}

const defaultOptions: Options = {
  default: ["⌘ ."],
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

  function Keybind({ displayClass }: QuartzComponentProps) {
    return (
      <div class={classNames(displayClass, "keybind")}>
        <kbd id="shortcut-key" data-keybind={revert(defaultKey)}>
          {defaultKey}
        </kbd>
        <div class="highlight-modal bind" id="highlight-modal">
          <span>afficher les raccourcis clavier</span>
        </div>
        <div id="shortcut-container">
          <div id="shortcut-space">
            <span id="title">raccourcis clavier</span>
            <ul>
              {Object.entries(KeybindAlias).map(([key, value]) => (
                <li>
                  <div id="shortcuts">
                    <kbd id="clickable-kbd" data-keybind={revert(key)}>
                      {convert(key)}
                    </kbd>
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
