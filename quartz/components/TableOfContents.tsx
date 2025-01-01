import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import modernStyle from "./styles/toc.scss"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/toc.inline"
import { i18n } from "../i18n"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
import { htmlToJsx } from "../util/jsx"
import Slugger from "github-slugger"

const ghSlugger = new Slugger()

interface Options {
  layout: "minimal" | "default"
}

const defaultOptions: Options = {
  layout: "minimal",
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const TableOfContents: QuartzComponent = ({
    fileData,
    displayClass,
    cfg,
  }: QuartzComponentProps) => {
    if (!fileData.toc) return null

    ghSlugger.reset()

    const convertFromText = (text: string) => {
      const tocAst = fromHtmlIsomorphic(text, { fragment: true })
      return htmlToJsx(fileData.filePath!, tocAst)
    }

    const MinimalToc = () => (
      <nav id="toc-vertical">
        {fileData.toc!.map((entry, idx) => (
          <button
            key={entry.slug}
            class={`depth-${entry.depth} toc-item`}
            data-depth={entry.depth}
            data-href={`#${entry.slug}`}
            data-for={entry.slug}
            tabindex={-1}
            type="button"
            style={{ "--animation-order": idx + 1 }}
            aria-label={`${entry.text}`}
            title={`${entry.text}`}
          >
            <div class="fill" />
            <div class="indicator">{convertFromText(entry.text)}</div>
          </button>
        ))}
      </nav>
    )

    const DefaultToc = () => (
      <nav>
        <button type="button" id="toc" aria-controls="toc-content">
          <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
        </button>
        <div id="toc-content">
          <ul class="overflow">
            {fileData.toc!.map((entry) => (
              <li key={entry.slug} class={`depth-${entry.depth}`}>
                <a href={`#${entry.slug}`} data-for={entry.slug}>
                  {convertFromText(entry.text)}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    )

    return (
      <div class={classNames(displayClass, "toc")} id="toc" data-layout={opts.layout}>
        {opts.layout === "minimal" ? <MinimalToc /> : <DefaultToc />}
      </div>
    )
  }

  TableOfContents.css = modernStyle
  TableOfContents.afterDOMLoaded = script
  return TableOfContents
}) satisfies QuartzComponentConstructor
