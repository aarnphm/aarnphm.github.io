import type { ElementContent } from 'hast'
import type { ArenaChannel, ArenaBlock } from '../../plugins/transformers/arena'
import type { QuartzComponentProps } from '../../types/component'
import { toArenaJsx, arenaBlockTimestamp } from '../../util/arena'
import { isArenaPdfUrl } from '../../util/arena-embed'
import { classNames } from '../../util/lang'
import { FullSlug, slugTag, resolveRelative, joinSegments } from '../../util/path'
import ArenaModalData from './ArenaModalData'
import { arenaReferenceText } from './reference-text'

const utcMediumDateFormatter = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
  timeZone: 'UTC',
})

const normalizeDate = (value: string): { display: string; dateTime?: string } => {
  const trimmed = value.trim()
  const match = trimmed.match(/^([0-9]{1,2})\/([0-9]{1,2})\/([0-9]{4})$/)
  if (!match) {
    return { display: trimmed.length > 0 ? trimmed : value }
  }

  const [, monthStr, dayStr, yearStr] = match
  const month = Number(monthStr)
  const day = Number(dayStr)
  const year = Number(yearStr)

  if (
    Number.isNaN(month) ||
    Number.isNaN(day) ||
    Number.isNaN(year) ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > 31
  ) {
    return { display: trimmed }
  }

  const date = new Date(Date.UTC(year, month - 1, day))
  const display = utcMediumDateFormatter.format(date)
  const iso = `${yearStr.padStart(4, '0')}-${monthStr.padStart(2, '0')}-${dayStr.padStart(2, '0')}`

  return { display, dateTime: iso }
}

export interface ArenaChannelSectionsResult {
  pinned: ArenaBlock[]
  later: ArenaBlock[]
  blocks: ArenaBlock[]
  ordered: ArenaBlock[]
  indexById: ReadonlyMap<string, number>
}

const arenaChannelSectionsCache = new WeakMap<
  ArenaChannel,
  { blocksRef: ArenaBlock[]; length: number; result: ArenaChannelSectionsResult }
>()

export function arenaChannelSections(channel: ArenaChannel): ArenaChannelSectionsResult {
  const cached = arenaChannelSectionsCache.get(channel)
  if (cached && cached.blocksRef === channel.blocks && cached.length === channel.blocks.length) {
    return cached.result
  }
  const sorted = (blocks: ArenaBlock[]) =>
    blocks.sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))
  const pinned = sorted(channel.blocks.filter(block => block.pinned))
  const later = sorted(channel.blocks.filter(block => block.later && !block.pinned))
  const blocks = sorted(channel.blocks.filter(block => !block.pinned && !block.later))
  const ordered = [...pinned, ...later, ...blocks]
  const indexById = new Map<string, number>()
  for (let i = 0; i < ordered.length; i++) {
    indexById.set(ordered[i].id, i)
  }
  const result: ArenaChannelSectionsResult = { pinned, later, blocks, ordered, indexById }
  arenaChannelSectionsCache.set(channel, {
    blocksRef: channel.blocks,
    length: channel.blocks.length,
    result,
  })
  return result
}

export function createArenaChannelRenderer(
  componentData: QuartzComponentProps,
  channel: ArenaChannel,
) {
  const { fileData, displayClass } = componentData
  const jsxFromNode = (node?: ElementContent) =>
    node
      ? toArenaJsx(fileData.filePath!, node, fileData.slug! as FullSlug, componentData)
      : undefined

  const renderInlineText = (text: string) => arenaReferenceText(text, componentData)

  return (block: ArenaBlock, blockIndex: number, view: 'card' | 'modal' | 'page' = 'card') => {
    const hasSubItems = block.subItems && block.subItems.length > 0
    const isPdfCandidate = block.url ? isArenaPdfUrl(block.url) : false
    const isPinned = block.pinned ?? false
    const accessedRaw =
      (typeof block.metadata?.accessed === 'string' ? block.metadata.accessed : undefined) ??
      (typeof block.metadata?.accessed_date === 'string'
        ? block.metadata.accessed_date
        : undefined) ??
      (typeof block.metadata?.date === 'string' ? block.metadata.date : undefined)
    const accessed = accessedRaw ? normalizeDate(accessedRaw) : undefined

    const hasMetaPreview =
      Boolean(accessed) || (block.tags && Array.isArray(block.tags) && block.tags.length > 0)

    if (view === 'modal' || view === 'page') {
      return (
        <ArenaModalData
          block={block}
          blockIndex={blockIndex}
          channel={channel}
          componentData={componentData}
          accessed={accessed}
          isPdfCandidate={isPdfCandidate}
          view={view}
          jsxFromNode={jsxFromNode}
          renderInlineText={renderInlineText}
        />
      )
    }
    if (!block.entryId) throw new Error(`Arena entry ID missing in ${channel.slug}: ${block.id}`)
    const entryHref = resolveRelative(
      fileData.slug! as FullSlug,
      joinSegments('arena', channel.slug, block.entryId) as FullSlug,
    )
    return (
      <div
        key={block.id}
        class={classNames(
          displayClass,
          'arena-block',
          block.highlighted ? 'highlighted' : '',
          isPinned ? 'pinned' : '',
        )}
        id={`arena-block-${block.id}`}
        data-block-id={block.id}
        data-entry-id={block.entryId}
        data-block-index={blockIndex}
        data-channel-slug={channel.slug}
      >
        {hasSubItems && (
          <div class="arena-block-connections-indicator">
            <svg
              width="12"
              height="12"
              viewBox="0 0 15 15"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M8 2.75C8 2.47386 7.77614 2.25 7.5 2.25C7.22386 2.25 7 2.47386 7 2.75V7H2.75C2.47386 7 2.25 7.22386 2.25 7.5C2.25 7.77614 2.47386 8 2.75 8H7V12.25C7 12.5261 7.22386 12.75 7.5 12.75C7.77614 12.75 8 12.5261 8 12.25V8H12.25C12.5261 8 12.75 7.77614 12.75 7.5C12.75 7.22386 12.5261 7 12.25 7H8V2.75Z"
                fill="currentColor"
                fill-rule="evenodd"
                clip-rule="evenodd"
              />
            </svg>
          </div>
        )}
        {isPdfCandidate && (
          <div class="arena-block-pdf-indicator" title="PDF document">
            <svg
              width="12"
              height="12"
              viewBox="0 0 15 15"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle cx="7.5" cy="7.5" r="3.5" fill="currentColor" />
            </svg>
          </div>
        )}
        {block.highlighted && !isPinned && (
          <span class="arena-block-favorite-indicator" role="img" aria-label="Favorite">
            <svg viewBox="0 0 15 15" fill="currentColor" aria-hidden="true" focusable="false">
              <path d="M3.5 1.5h8v12l-4-3-4 3z" />
            </svg>
          </span>
        )}
        <a class="arena-block-clickable" href={entryHref} data-no-popover>
          <div class="arena-block-content">{block.title || block.content}</div>
          {hasMetaPreview && (
            <div class="arena-block-meta">
              {accessed ? (
                <span class="arena-block-meta-item">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 15 15"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    class="arena-block-meta-icon"
                    aria-hidden="true"
                  >
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M7.5 1.6a5.9 5.9 0 1 0 0 11.8 5.9 5.9 0 0 0 0-11.8ZM2.4 7.5a5.1 5.1 0 1 1 10.2 0 5.1 5.1 0 0 1-10.2 0Zm5.6-3.4a.4.4 0 1 0-.8 0v3.2c0 .11.04.21.12.29l2.1 2.1a.4.4 0 0 0 .56-.56L8 7.07V4.1Z"
                      fill="currentColor"
                    />
                  </svg>
                  {accessed.dateTime ? (
                    <time dateTime={accessed.dateTime}>{accessed.display}</time>
                  ) : (
                    accessed.display
                  )}
                </span>
              ) : null}
              {block.tags && block.tags.length > 0 ? (
                <span class="arena-block-meta-item arena-block-meta-tags">
                  {block.tags.map(tag => (
                    <span class="tag-link" key={`${block.id}-tag-preview-${slugTag(tag)}`}>
                      {tag}
                    </span>
                  ))}
                </span>
              ) : null}
            </div>
          )}
        </a>
      </div>
    )
  }
}
