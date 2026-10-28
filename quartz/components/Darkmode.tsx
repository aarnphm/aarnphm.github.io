import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore: this is safe, we don't want to actually make darkmode.inline.ts a module as
// modules are automatically deferred and we don't want that to happen for critical beforeDOMLoads
// see: https://v8.dev/features/modules#defer
import darkmodeScript from './scripts/darkmode.inline'
import styles from './styles/darkmode.scss'

export default (() => {
  const Darkmode: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <span
      id="light-toggle"
      class={classNames(displayClass, 'darkmode')}
      // @ts-ignore
      type="button"
      role="button"
      tabIndex={0}
      title="Change theme"
      aria-label="Toggle theme"
    >
      <svg id="light-toggle-system" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 7a5 5 0 0 0 0 10" />
        <path d="M12 7a5 5 0 0 1 0 10" />
        <path d="M14.2 8.6a4.2 4.2 0 0 0 0 6.8" />
        <path d="M12 3v1.5M5.6 6.4l1.1 1.1M3 12h1.5M5.6 17.6l1.1-1.1M12 21v-1.5" />
      </svg>
      <svg id="light-toggle-dark" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20.5 13.4A8 8 0 1 1 10.6 3.5a6.5 6.5 0 0 0 9.9 9.9z" />
      </svg>
      <svg id="light-toggle-light" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8" />
        <path d="M12 3v1.5M12 19.5V21M3 12h1.5M19.5 12H21M5.6 5.6l1.1 1.1M17.3 17.3l1.1 1.1M5.6 18.4l1.1-1.1M17.3 6.7l1.1-1.1" />
      </svg>
    </span>
  )

  Darkmode.beforeDOMLoaded = darkmodeScript
  Darkmode.css = styles

  return Darkmode
}) satisfies QuartzComponentConstructor
