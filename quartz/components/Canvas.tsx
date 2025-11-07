import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/canvas.inline"
import style from "./styles/canvas.scss"

export default (() => {
  const Canvas: QuartzComponent = (_: QuartzComponentProps) => <></>

  Canvas.css = style
  Canvas.afterDOMLoaded = script

  return Canvas
}) satisfies QuartzComponentConstructor
