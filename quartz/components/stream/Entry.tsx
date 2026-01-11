import { ComponentChild } from "preact"
import type { ElementContent, Root as HastRoot } from "hast"
import { htmlToJsx } from "../../util/jsx"
import type { FilePath } from "../../util/path"
import type { StreamEntry } from "../../plugins/transformers/stream"

export interface StreamEntryRenderOptions {
  groupId: string
  timestampValue?: number
  showDate?: boolean
  resolvedIsoDate?: string
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

  const resolvedIsoDate = options.resolvedIsoDate ?? entry.date
  const formattedDate = showDate ? formatStreamDate(resolvedIsoDate) : null
  const ariaLabel = formattedDate ? formattedDate : undefined

  const onPath = timestampAttr ? buildOnPath(resolvedIsoDate) : null

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
        <div class="stream-entry-content">{nodesToJsx(filePath, entry.content)}</div>
      </div>
    </li>
  )
}
