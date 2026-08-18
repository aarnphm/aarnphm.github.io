import style from '../styles/cursor.scss'
import { QuartzComponent, QuartzComponentConstructor } from '../types/component'
// @ts-ignore
import script from './scripts/cursor.inline'

export default (() => {
  const Cursor: QuartzComponent = () => (
    <span class="site-cursor" data-mode="diamond" data-visible="false" aria-hidden="true">
      <span class="site-cursor-diamond" />
      <span class="site-cursor-question">?</span>
      <span class="site-cursor-crosshair" />
      <span class="site-cursor-line" />
    </span>
  )

  Cursor.css = style
  Cursor.afterDOMLoaded = script

  return Cursor
}) satisfies QuartzComponentConstructor
