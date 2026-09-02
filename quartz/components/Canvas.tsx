import { QuartzComponent, QuartzComponentConstructor } from '../types/component'
// @ts-ignore
import script from './scripts/canvas.inline'
import style from './styles/canvas.scss'

export default (() => {
  const Canvas: QuartzComponent = () => <></>

  Canvas.css = style
  Canvas.afterDOMLoaded = script

  return Canvas
}) satisfies QuartzComponentConstructor
