import { htmlToJsx } from "../../util/jsx"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
// @ts-ignore
import script from "../scripts/mermaid.inline"
import style from "../styles/mermaid.scss"

export default (() => {
  const Content: QuartzComponent = ({ fileData, tree }: QuartzComponentProps) => {
    const content = htmlToJsx(fileData.filePath!, tree)
    const classes: string[] = fileData.frontmatter?.cssclasses ?? []
    const classString = ["popover-hint", "main-col", ...classes].join(" ")
    return <article class={classString}>{content}</article>
  }

  Content.afterDOMLoaded = script
  Content.css = style

  return Content
}) satisfies QuartzComponentConstructor
