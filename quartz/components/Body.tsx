// @ts-ignore
import clipboardScript from "./scripts/clipboard.inline"
// @ts-ignore
import progressScript from "./scripts/progress.inline"
import clipboardStyle from "./styles/clipboard.scss"
import { QuartzComponentConstructor, QuartzComponentProps } from "./types"

function Body({ children }: QuartzComponentProps) {
  return <div id="quartz-body">{children}</div>
}

Body.afterDOMLoaded = progressScript + clipboardScript
Body.css = clipboardStyle

export default (() => Body) satisfies QuartzComponentConstructor
