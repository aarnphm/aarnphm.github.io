import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/inline-footnotes.inline"
import style from "./styles/inlineFootnotes.scss"
import { classNames } from "../util/lang"

export default (() => {
  const InlineFootnotes: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div class={classNames(displayClass, "inline-footnotes")}></div>
  )

  InlineFootnotes.css = style
  // InlineFootnotes.afterDOMLoaded = script

  return InlineFootnotes
}) satisfies QuartzComponentConstructor
