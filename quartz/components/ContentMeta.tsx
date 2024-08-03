import { formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import readingTime from "reading-time"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"
//@ts-ignore
import script from "./scripts/content-meta.inline"

interface MetadataOptions {
  showReadingTime?: boolean
  showModifiedTime?: boolean
  showReturnLink?: boolean
}

type LinkOptions = object

type Mode = "metadata" | "link"

type ModeOptions = {
  metadata?: MetadataOptions
  link?: LinkOptions
}

interface ContentMetaOptions {
  /**
   * Whether to display reading time
   */
  showMode: Mode
  // TODO: make sure to parse link similar how we transclude options
  link?: string
  modeOptions?: ModeOptions
}

const defaultOptions: ContentMetaOptions = {
  showMode: "metadata",
  link: "/",
  modeOptions: {
    metadata: {
      showReadingTime: true,
      showModifiedTime: true,
      showReturnLink: false,
    },
    link: {},
  },
}

export default ((opts?: Partial<ContentMetaOptions>) => {
  // Merge options with defaults
  const options: ContentMetaOptions = { ...defaultOptions, ...opts }
  const modeOptions = options.modeOptions![options.showMode!]

  const ContentMetadata: QuartzComponent = ({
    cfg,
    fileData,
    displayClass,
  }: QuartzComponentProps) => {
    const text = fileData.text

    if (!text || !modeOptions) return null

    let created: string | undefined
    let modified: string | undefined
    let reading: string | undefined

    if (options.showMode === "metadata") {
      if (fileData.dates) {
        created = formatDate(getDate(cfg, fileData)!, cfg.locale)
        modified = formatDate(fileData.dates.modified, cfg.locale)
      }

      // Display reading time if enabled
      const { minutes, text: _timeTaken, words: _words } = readingTime(text)
      reading = i18n(cfg.locale).components.contentMeta.readingTime({
        minutes: Math.ceil(minutes),
      })
    }

    const home = (link: string) => (
      <li class="return-home">
        <a
          href={link}
          class="internal alias"
          style={["color: inherit", "font-weight: inherit"].join(";")}
        >
          home
        </a>
      </li>
    )

    if (options.showMode === "link") {
      return <ul class={classNames(displayClass, "content-meta")}>{home(options.link!)}</ul>
    } else {
      const metadata = modeOptions as MetadataOptions
      return (
        <ul class={classNames(displayClass, "content-meta")}>
          {metadata.showReadingTime && (
            <>
              {created !== undefined && (
                <li>
                  <span class="page-creation" title="Date de création du contenu de la page">
                    <em>{created}</em>
                  </span>
                </li>
              )}
              {metadata.showModifiedTime && modified !== undefined && (
                <li>
                  <a class="ref-source">
                    <span
                      class="page-modification"
                      title="Date de modification du contenu de la page"
                    >
                      <em>{modified}</em>
                    </span>
                  </a>
                </li>
              )}
              {reading !== undefined && (
                <li>
                  <span className="reading-time" title="Temps de lecture estimé">
                    {reading}
                  </span>
                </li>
              )}
              {metadata.showReturnLink && home(options.link!)}
            </>
          )}
        </ul>
      )
    }
  }

  ContentMetadata.css = contentMetaStyle
  ContentMetadata.afterDOMLoaded = script

  return ContentMetadata
}) satisfies QuartzComponentConstructor
