// @ts-ignore
import clipboardScript from "./scripts/clipboard.inline"
import clipboardStyle from "./styles/clipboard.scss"
// @ts-ignore
import equationScript from "./scripts/equation.inline"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import Pseudocode from "./Pseudocode"

const Body: QuartzComponent = (props: QuartzComponentProps) => {
  const { children } = props
  const Pseudo = Pseudocode()
  return (
    <div id="quartz-body">
      <>{children}</>
      <Pseudo {...props} />
    </div>
  )
}

Body.beforeDOMLoaded = equationScript
Body.afterDOMLoaded = clipboardScript
Body.css = clipboardStyle

export default (() => Body) satisfies QuartzComponentConstructor
