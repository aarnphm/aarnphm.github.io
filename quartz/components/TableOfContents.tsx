import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import modernStyle from "./styles/toc.scss"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/toc.inline"
import { i18n } from "../i18n"
import { fromHtml } from "hast-util-from-html"
import { htmlToJsx } from "../util/jsx"

const TableOfContents: QuartzComponent = ({
  fileData,
  displayClass,
  cfg,
}: QuartzComponentProps) => {
  if (!fileData.toc) {
    return null
  }

  const convertFromText = (text: string) => {
    const tocAst = fromHtml(text, { fragment: true })
    return htmlToJsx(fileData.filePath!, tocAst)
  }

  return (
    <div class={classNames(displayClass, "toc")}>
      <button type="button" id="toc" aria-controls="toc-content">
        <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
      </button>
      <div id="toc-content">
        <ul class="overflow">
          {fileData.toc.map((tocEntry) => (
            <li key={tocEntry.slug} class={`depth-${tocEntry.depth}`}>
              <a href={`#${tocEntry.slug}`} data-for={tocEntry.slug}>
                {convertFromText(tocEntry.text)}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
TableOfContents.css = modernStyle
TableOfContents.afterDOMLoaded = script

export default (() => TableOfContents) satisfies QuartzComponentConstructor
