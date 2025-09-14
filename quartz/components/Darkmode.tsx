// @ts-ignore: this is safe, we don't want to actually make darkmode.inline.ts a module as
// modules are automatically deferred and we don't want that to happen for critical beforeDOMLoads
// see: https://v8.dev/features/modules#defer
import darkmodeScript from "./scripts/darkmode.inline"
import styles from "./styles/darkmode.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default (() => {
  const Darkmode: QuartzComponent = (_: QuartzComponentProps) => {
    return null
  }

  Darkmode.beforeDOMLoaded = darkmodeScript
  Darkmode.css = styles

  return Darkmode
}) satisfies QuartzComponentConstructor
