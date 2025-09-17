// @ts-ignore: this is safe, we don't want to actually make darkmode.inline.ts a module as
// modules are automatically deferred and we don't want that to happen for critical beforeDOMLoads
// see: https://v8.dev/features/modules#defer
import darkmodeScript from "./scripts/darkmode.inline"
import styles from "./styles/darkmode.scss"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default (() => {
  const Darkmode: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <span
      id="light-toggle"
      class={classNames(displayClass, "darkmode")}
      type="button"
      role="button"
      tabIndex={0}
      title="Change theme"
      aria-label="Toggle theme"
    >
      <i class="ti ti-sun-moon" id="light-toggle-system" aria-hidden="true" />
      <i class="ti ti-moon-filled" id="light-toggle-dark" aria-hidden="true" />
      <i class="ti ti-sun-filled" id="light-toggle-light" aria-hidden="true" />
    </span>
  )

  Darkmode.beforeDOMLoaded = darkmodeScript
  Darkmode.css = styles

  return Darkmode
}) satisfies QuartzComponentConstructor
