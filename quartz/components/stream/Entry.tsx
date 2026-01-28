import type { ElementContent, Root as HastRoot } from "hast"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
import { toString as hastToString } from "hast-util-to-string"
import { ComponentChild } from "preact"
import type { StreamEntry } from "../../plugins/transformers/stream"
import type { FilePath } from "../../util/path"
import { htmlToJsx } from "../../util/jsx"

export interface StreamEntryRenderOptions {
  groupId: string
  timestampValue?: number
  showDate?: boolean
  resolvedIsoDate?: string
  showWordCount?: boolean
}

const nodesToJsx = (filePath: FilePath, nodes: ElementContent[]): ComponentChild => {
  if (!nodes || nodes.length === 0) return null

  return nodes.map((node, idx) => {
    const root: HastRoot = {
      type: "root",
      children: [node as any],
    }
    return <span key={idx}>{htmlToJsx(filePath, root)}</span>
  })
}

const countWords = (value: string): number => {
  const trimmed = value.trim()
  if (!trimmed) return 0
  return trimmed.split(/\s+/).filter((token) => token.length > 0).length
}

const streamEntryText = (entry: StreamEntry): string => {
  const root: HastRoot = {
    type: "root",
    children: entry.content,
  }
  const contentText = hastToString(root)
  const titleText = entry.title ? String(entry.title) : ""
  const descriptionText = entry.description ? String(entry.description) : ""
  return [titleText, descriptionText, contentText]
    .filter((part) => part.length > 0)
    .join(" ")
    .trim()
}

const descriptionToJsx = (filePath: FilePath, descriptionHtml: string): ComponentChild => {
  const root = fromHtmlIsomorphic(descriptionHtml, { fragment: true })
  return htmlToJsx(filePath, root)
}

export const getStreamEntryWordCount = (entry: StreamEntry): number =>
  countWords(streamEntryText(entry))

export const formatWordCount = (count: number): string =>
  count === 1 ? "1 word" : `${count} words`

export const formatStreamDate = (isoDate: string | undefined): string | null => {
  if (!isoDate) return null

  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return null

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

export const buildOnPath = (isoDate: string | undefined): string | null => {
  if (!isoDate) return null

  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return null

  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, "0")
  const day = String(date.getUTCDate()).padStart(2, "0")

  return `/stream/on/${year}/${month}/${day}`
}

export const renderStreamEntry = (
  entry: StreamEntry,
  filePath: FilePath,
  options: StreamEntryRenderOptions,
): ComponentChild => {
  const tags = Array.isArray(entry.metadata.tags) ? entry.metadata.tags : []
  const socials =
    entry.metadata.socials && typeof entry.metadata.socials === "object"
      ? (entry.metadata.socials as Record<string, unknown>)
      : null

  const timestampAttr =
    typeof options.timestampValue === "number" ? String(options.timestampValue) : undefined

  const showDate = options.showDate !== undefined ? options.showDate : true
  const showWordCount = options.showWordCount !== undefined ? options.showWordCount : false

  const resolvedIsoDate = options.resolvedIsoDate ?? entry.date
  const formattedDate = showDate ? formatStreamDate(resolvedIsoDate) : null
  const ariaLabel = formattedDate ? formattedDate : undefined

  const onPath = timestampAttr ? buildOnPath(resolvedIsoDate) : null
  const wordCount = showWordCount ? getStreamEntryWordCount(entry) : 0
  const wordCountLabel = showWordCount && wordCount > 0 ? formatWordCount(wordCount) : null
  const descriptionContent = entry.descriptionHtml
    ? descriptionToJsx(filePath, entry.descriptionHtml)
    : entry.description

  return (
    <li
      key={entry.id}
      class="stream-entry"
      data-entry-id={entry.id}
      data-stream-group-id={options.groupId}
      data-stream-timestamp={timestampAttr}
    >
      <div class="stream-entry-meta">
        {formattedDate && timestampAttr && onPath ? (
          <a
            class="stream-entry-date"
            href={onPath}
            data-stream-group-id={options.groupId}
            data-stream-timestamp={timestampAttr}
            data-stream-href={onPath}
            data-stream-link
            aria-label={ariaLabel ?? undefined}
          >
            <time dateTime={resolvedIsoDate ?? undefined}>{formattedDate}</time>
          </a>
        ) : (
          formattedDate && (
            <time
              class="stream-entry-date"
              dateTime={resolvedIsoDate ?? undefined}
              data-stream-group-id={options.groupId}
              data-stream-timestamp={timestampAttr}
              aria-label={ariaLabel ?? undefined}
            >
              {formattedDate}
            </time>
          )
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
        {socials && Object.keys(socials).length > 0 && (
          <div class="stream-entry-socials">
            {Object.entries(socials).map(([name, link]) => {
              const href = typeof link === "string" ? link : (link?.toString?.() ?? "")
              const isInternal = href.startsWith("/")
              return (
                <address key={name}>
                  <a
                    href={href}
                    target={!isInternal ? "_blank" : ""}
                    rel={!isInternal ? "noopener noreferrer" : ""}
                    class={isInternal ? "internal" : "external"}
                    data-no-popover
                  >
                    {name}
                  </a>
                </address>
              )
            })}
          </div>
        )}
        {entry.importance !== undefined && (
          <div class="stream-entry-importance">
            <span class="stream-entry-importance-label">importance:</span>{" "}
            <span class="stream-entry-importance-value">{entry.importance}</span>
          </div>
        )}
      </div>
      <div class="stream-entry-body">
        {entry.title && <h2 class="stream-entry-title">{entry.title}</h2>}
        {descriptionContent && <p class="stream-entry-description">{descriptionContent}</p>}
        <div class="stream-entry-content">{nodesToJsx(filePath, entry.content)}</div>
        {wordCountLabel && (
          <div class="stream-entry-wordcount">
            <em>{wordCountLabel}</em>
          </div>
        )}
      </div>
    </li>
  )
}
