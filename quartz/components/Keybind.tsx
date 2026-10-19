import { render } from 'preact-render-to-string'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/keybind.inline'
import style from './styles/keybind.scss'

interface Options {
  default?: string[]
}

const defaultOptions: Options = { default: ["⌘ '", "⌃ '"] }

const convert = (key: string) =>
  key
    .replace('cmd', '⌘')
    .replace('ctrl', '⌃')
    .replace('alt', '⌥')
    .replace('shift', '⇧')
    .replaceAll('+', ' ')

const revert = (key: string) =>
  key
    .replace('⌘', 'cmd')
    .replace('⌃', 'ctrl')
    .replace('⌥', 'alt')
    .replace('⇧', 'shift')
    .replace(' ', '--')
    .replace('+', '--')

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }
  const defaultKey = opts.default![0]
  const mappingAttr = JSON.stringify(opts.default?.map(revert))
  const innerHtmlCache = new Map<string, string>()

  const Keybind: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    const isIndex = fileData.slug === 'index'
    const cacheKey = `${isIndex ? '1' : '0'}:${JSON.stringify(cfg.keybinds ?? {})}`
    let innerHtml = innerHtmlCache.get(cacheKey)
    if (innerHtml === undefined) {
      const shortcuts = Object.entries(cfg.keybinds ?? {}).map(([key, value]) =>
        isIndex && key === 'gh' ? (['gt', 'sentier'] as const) : ([key, value] as const),
      )
      innerHtml = render(
        <>
          <kbd id="shortcut-key" data-mapping={mappingAttr}>
            {defaultKey}
          </kbd>
          <div id="shortcut-container">
            <div id="shortcut-space">
              <div id="title">raccourcis clavier</div>
              <ul id="shortcut-list">
                {shortcuts.map(([key, value]) => (
                  <li>
                    <div
                      id="shortcuts"
                      data-key={convert(key).replace(' ', '--')}
                      data-value={value}
                    ></div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>,
      )
      innerHtmlCache.set(cacheKey, innerHtml)
    }
    return (
      <div
        class={classNames(displayClass, 'keybind')}
        lang={'fr'}
        dangerouslySetInnerHTML={{ __html: innerHtml }}
      />
    )
  }

  Keybind.css = style
  Keybind.afterDOMLoaded = script

  return Keybind
}) satisfies QuartzComponentConstructor
