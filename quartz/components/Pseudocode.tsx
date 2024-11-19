import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/clipboard-pseudo.inline"
import style from "./styles/pseudocode.scss"

export default (() => {
  const Pseudocode: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    if (!fileData.pseudocode) {
      return <></>
    }
    return <div class="has-pseudocode" />
  }
  Pseudocode.afterDOMLoaded = script
  Pseudocode.css = style
  return Pseudocode
}) satisfies QuartzComponentConstructor
