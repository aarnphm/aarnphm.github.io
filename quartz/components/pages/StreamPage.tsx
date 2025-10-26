import { ComponentChild } from "preact"
import { htmlToJsx } from "../../util/jsx"
import type { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import type { Root as HastRoot, ElementContent } from "hast"
import type { FilePath } from "../../util/path"
import { StreamEntry } from "../../plugins/transformers/stream"
import style from "../styles/stream.scss"

const toJsx = (filePath: FilePath, nodes: ElementContent[]): ComponentChild => {
  if (!nodes || nodes.length === 0) return null

  return nodes.map((node, idx) => {
    const root: HastRoot = {
      type: "root",
      children: [node as any],
    }
    return <span key={idx}>{htmlToJsx(filePath, root)}</span>
  })
}

const formatDate = (isoDate: string | undefined): string | null => {
  if (!isoDate) return null

  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return null

  // format in PST timezone with time
  const formatter = new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: false,
    timeZone: "America/Los_Angeles",
    timeZoneName: "shortOffset",
  })

  return formatter.format(date)
}

const StreamEntryComponent = (entry: StreamEntry, filePath: FilePath): ComponentChild => {
  const formattedDate = formatDate(entry.date)
  const tags = Array.isArray(entry.metadata.tags) ? entry.metadata.tags : []

  return (
    <li class="stream-entry" data-entry-id={entry.id}>
      <div class="stream-entry-meta">
        {formattedDate && (
          <time class="stream-entry-date" dateTime={entry.date}>
            {formattedDate}
          </time>
        )}
        {tags.length > 0 && (
          <div class="stream-entry-tags">
            {tags.map((tag, idx) => (
              <span key={idx} class="stream-entry-tag">
                {String(tag)}
              </span>
            ))}
          </div>
        )}
      </div>
      <div class="stream-entry-body">
        {entry.title && <h2 class="stream-entry-title">{entry.title}</h2>}
        <div class="stream-entry-content">{toJsx(filePath, entry.content)}</div>
      </div>
    </li>
  )
}

export default (() => {
  const StreamPage: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    if (!fileData.streamData || fileData.streamData.entries.length === 0) {
      return <article class="stream-empty main-col popover-hint">stream is empty.</article>
    }

    return (
      <article class="stream main-col popover-hint">
        <ol class="stream-feed">
          {fileData.streamData.entries.map((entry) =>
            StreamEntryComponent(entry, fileData.filePath!),
          )}
        </ol>
      </article>
    )
  }

  StreamPage.css = style

  return StreamPage
}) satisfies QuartzComponentConstructor
