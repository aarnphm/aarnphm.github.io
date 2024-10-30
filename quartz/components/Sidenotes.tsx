import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/sidenotes.inline"
import style from "./styles/sidenotes.scss"
import { classNames } from "../util/lang"

export default (() => {
  const Sidenotes: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div class={classNames(displayClass, "sidenotes")}></div>
  )

  Sidenotes.css = style
  Sidenotes.afterDOMLoaded = script

  return Sidenotes
}) satisfies QuartzComponentConstructor
