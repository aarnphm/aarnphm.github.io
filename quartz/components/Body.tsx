// @ts-ignore
import clipboardScript from "./scripts/clipboard.inline"
import clipboardStyle from "./styles/clipboard.scss"
// @ts-ignore
import equationScript from "./scripts/equation.inline"
// @ts-ignore
import pseudoScript from "./scripts/clipboard-pseudo.inline"
import pseudoStyle from "./styles/pseudocode.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Body: QuartzComponent = ({ children }: QuartzComponentProps) => (
  <section id="quartz-body">{children}</section>
)

Body.beforeDOMLoaded = equationScript
Body.afterDOMLoaded = clipboardScript + pseudoScript
Body.css = clipboardStyle + pseudoStyle

export default (() => Body) satisfies QuartzComponentConstructor
