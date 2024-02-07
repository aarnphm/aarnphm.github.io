import { formatDate, getDate } from "./Date"
import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import readingTime from "reading-time"
import { classNames } from "../util/lang"

interface ContentMetaOptions {
  /**
   * Whether to display reading time
   */
  showReadingTime: boolean
}

const defaultOptions: ContentMetaOptions = {
  showReadingTime: true,
}

export default ((opts?: Partial<ContentMetaOptions>) => {
  // Merge options with defaults
  const options: ContentMetaOptions = { ...defaultOptions, ...opts }

  function ContentMetadata({ cfg, fileData, displayClass }: QuartzComponentProps) {
    const text = fileData.text

    if (!text) return null

    let created: string | undefined
    let modified: string | undefined
    let reading: number | undefined

    if (fileData.dates) {
      created = formatDate(getDate(cfg, fileData)!, cfg.locale)
      modified = formatDate(fileData.dates.modified, cfg.locale)
    }

    // Display reading time if enabled
    if (options.showReadingTime) {
      const results = readingTime(text)
      reading = Math.ceil(results.minutes)
    }

    return (
      <p class={classNames(displayClass, "content-meta")}>
        {created !== undefined && `c: ${created}`}
        {" · "}
        <em>{modified !== undefined && `m: ${modified}`}</em>
        {" · "}
        {reading !== undefined && `r: ${reading} ${reading === 1 ? "min" : "mins"}`}
      </p>
    )
  }

  ContentMetadata.css = `
  .content-meta {
    margin-top: 0;
    color: var(--gray);
  }
  `
  return ContentMetadata
}) satisfies QuartzComponentConstructor
