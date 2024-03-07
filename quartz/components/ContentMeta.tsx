import { formatDate, getDate } from "./Date"
import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import readingTime from "reading-time"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

interface ContentMetaOptions {
  /**
   * Whether to display reading time
   */
  showReadingTime: boolean
  showReturnHome: boolean
  addHomeLink: boolean
}

const defaultOptions: ContentMetaOptions = {
  showReadingTime: true,
  showReturnHome: false,
  addHomeLink: false,
}

export default ((opts?: Partial<ContentMetaOptions>) => {
  // Merge options with defaults
  const options: ContentMetaOptions = { ...defaultOptions, ...opts }

  const ContentMetadata = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    const text = fileData.text

    if (!text) return null

    let created: string | undefined
    let modified: string | undefined
    let reading: string | undefined

    if (options.showReadingTime) {
      if (fileData.dates) {
        created = formatDate(getDate(cfg, fileData)!, cfg.locale)
        modified = formatDate(fileData.dates.modified, cfg.locale)
      }

      // Display reading time if enabled
      const { minutes, text: timeTaken, words: _words } = readingTime(text)
      reading = i18n(cfg.locale).components.contentMeta.readingTime({
        minutes: Math.ceil(minutes),
      })
    }

    const home = () => (
      <li class="return-home">
        <a
          href={"/"}
          class="internal alias"
          style={["color: inherit", "font-weight: inherit"].join(";")}
        >
          home
        </a>
      </li>
    )

    return (
      <ul class={classNames(displayClass, "content-meta")}>
        {options.showReadingTime ? (
          <>
            {created !== undefined ? (
              <li>
                <span class="page-creation" title="Date de création du contenu de la page">
                  <em>{created}</em>
                </span>
              </li>
            ) : (
              <></>
            )}
            {modified !== undefined ? (
              <li>
                <a class="ref-source internal">
                  <span
                    class="page-modification"
                    title="Date de modification du contenu de la page"
                  >
                    <em>{modified}</em>
                  </span>
                  <div class="popover">
                    <div class="popover-inner" data-content-type="text/html">
                      <pre data-language="markdown">{fileData.markdown}</pre>
                    </div>
                  </div>
                </a>
              </li>
            ) : (
              <></>
            )}
            {reading !== undefined ? (
              <li>
                <span class="reading-time" title="Temps de lecture estimé">
                  {reading}
                </span>
              </li>
            ) : (
              <></>
            )}
            {options.addHomeLink ? home() : <></>}
          </>
        ) : (
          <></>
        )}
        {options.showReturnHome ? home() : <></>}
      </ul>
    )
  }

  ContentMetadata.css = contentMetaStyle
  return ContentMetadata
}) satisfies QuartzComponentConstructor
