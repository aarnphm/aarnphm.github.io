import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../../types/component"
import { groupStreamEntries } from "../../util/stream"
// @ts-ignore
import script from "../scripts/stream.inline"
import { renderStreamEntry } from "../stream/Entry"
import StreamSearchConstructor from "../StreamSearch"
import style from "../styles/stream.scss"

export default (() => {
  const StreamSearch = StreamSearchConstructor()

  const StreamPage: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData } = props
    if (!fileData.streamData || fileData.streamData.entries.length === 0) {
      return <article class="stream-empty main-col popover-hint">stream is empty.</article>
    }

    const canonicalPathRaw = fileData.frontmatter?.streamCanonical
    const canonicalPath =
      typeof canonicalPathRaw === "string" && canonicalPathRaw.trim().length > 0
        ? canonicalPathRaw.startsWith("/")
          ? canonicalPathRaw
          : `/${canonicalPathRaw}`
        : "/stream"

    const slug = fileData.slug ?? ""
    const isDailyView = typeof slug === "string" && slug.startsWith("stream/on/")

    const groups = groupStreamEntries(fileData.streamData.entries)
    const entriesWithContext = groups.flatMap((group) =>
      group.entries.map((entry) => ({ entry, group })),
    )

    return (
      <article class="stream main-col popover-hint" data-stream-canonical={canonicalPath}>
        {!isDailyView && <StreamSearch {...props} />}
        <ol class="stream-feed">
          {entriesWithContext.map(({ entry, group }) =>
            renderStreamEntry(entry, fileData.filePath!, {
              groupId: group.id,
              timestampValue: group.timestamp,
              showDate: true,
              resolvedIsoDate: entry.date ?? group.isoDate,
              showWordCount: true,
            }),
          )}
        </ol>
        {isDailyView && (
          <div class="stream-backlink">
            <a href="/stream">← back to stream</a>
          </div>
        )}
      </article>
    )
  }

  StreamPage.css = style
  StreamPage.afterDOMLoaded = script

  return StreamPage
}) satisfies QuartzComponentConstructor
