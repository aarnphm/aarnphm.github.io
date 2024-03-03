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
}

const defaultOptions: ContentMetaOptions = {
  showReadingTime: true,
  showReturnHome: false,
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

    return (
      <p class={classNames(displayClass, "content-meta")}>
        {options.showReadingTime ? (
          <>
            <span class="date-range">
              {created !== undefined ? (
                <span class="page-creation" title="Date page contents were created">
                  <em>{created}</em>
                </span>
              ) : (
                <></>
              )}
              {"–"}
              {/* TODO: Add support for latest markdown revision popover */}
              {modified !== undefined ? (
                <span class="page-source">
                  <a class="ref-source">
                    <span class="indicator-hook"></span>
                    <span class="page-modification" title="Date page contents were modified">
                      <em>{modified}</em>
                    </span>
                  </a>
                </span>
              ) : (
                <></>
              )}
            </span>
            {reading !== undefined ? (
              <span class="reading-time" title="Estimated reading time">
                {reading}
              </span>
            ) : (
              <></>
            )}
          </>
        ) : (
          <></>
        )}
        {options.showReturnHome ? (
          <span class="return-home">
            <em>
              <a href={"/"} style={["color: inherit", "font-weight: inherit"].join(";")}>
                home
              </a>
            </em>
          </span>
        ) : (
          <></>
        )}
      </p>
    )
  }

  ContentMetadata.css = contentMetaStyle
  return ContentMetadata
}) satisfies QuartzComponentConstructor
