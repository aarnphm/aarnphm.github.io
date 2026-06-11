import type { ElementContent, Root as HastRoot } from 'hast'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toString as hastToString } from 'hast-util-to-string'
import { ComponentChild } from 'preact'
import { render } from 'preact-render-to-string'
import type { StreamEntry } from '../../plugins/transformers/stream'
import type { FilePath } from '../../util/path'
import type { EncryptedPayload } from '../../util/protected'
import { htmlToJsx } from '../../util/jsx'
import {
  buildStreamDayPathFromIso,
  formatStreamDate,
  isDraftEntry,
  isPrivateEntry,
  isProtectedEntry,
  isRestrictedEntry,
  truthyStreamFlag,
} from '../../util/stream'
import { StreamUnlockIcon } from './UnlockIcon'

export interface StreamEntryRenderOptions {
  groupId: string
  timestampValue?: number
  showDate?: boolean
  resolvedIsoDate?: string
  showWordCount?: boolean
  mode?: 'listing' | 'daily'
  encryptedPayload?: EncryptedPayload
  protectedPrompt?: 'form' | 'icon'
}

const nodesToJsx = (filePath: FilePath, nodes: ElementContent[]): ComponentChild => {
  if (!nodes || nodes.length === 0) return null

  return nodes.map((node, idx) => {
    const root: HastRoot = { type: 'root', children: [node] }
    return <span key={idx}>{htmlToJsx(filePath, root)}</span>
  })
}

const countWords = (value: string): number => {
  const trimmed = value.trim()
  if (!trimmed) return 0
  return trimmed.split(/\s+/).filter(token => token.length > 0).length
}

const streamEntryText = (entry: StreamEntry): string => {
  const root: HastRoot = { type: 'root', children: entry.content }
  const contentText = hastToString(root)
  const titleText = entry.title ? String(entry.title) : ''
  const descriptionText = entry.description ? String(entry.description) : ''
  return [titleText, descriptionText, contentText]
    .filter(part => part.length > 0)
    .join(' ')
    .trim()
}

const descriptionToJsx = (filePath: FilePath, descriptionHtml: string): ComponentChild => {
  const root = fromHtmlIsomorphic(descriptionHtml, { fragment: true })
  return htmlToJsx(filePath, root)
}

export { isDraftEntry, isPrivateEntry, isProtectedEntry, isRestrictedEntry, truthyStreamFlag }

export const getStreamEntryWordCount = (entry: StreamEntry): number =>
  countWords(streamEntryText(entry))

export const formatWordCount = (count: number): string =>
  count === 1 ? '1 word' : `${count} words`

export { formatStreamDate }

export const buildOnPath = (isoDate: string | undefined): string | null => {
  return buildStreamDayPathFromIso(isoDate)
}

const renderEntryBody = (
  entry: StreamEntry,
  filePath: FilePath,
  showWordCount: boolean,
): ComponentChild => {
  const wordCount = showWordCount ? getStreamEntryWordCount(entry) : 0
  const wordCountLabel = showWordCount && wordCount > 0 ? formatWordCount(wordCount) : null
  const descriptionContent = entry.descriptionHtml
    ? descriptionToJsx(filePath, entry.descriptionHtml)
    : entry.description

  return (
    <>
      {descriptionContent && <p class="stream-entry-description">{descriptionContent}</p>}
      <div class="stream-entry-content">{nodesToJsx(filePath, entry.content)}</div>
      {wordCountLabel && (
        <div class="stream-entry-wordcount">
          <em>{wordCountLabel}</em>
        </div>
      )}
    </>
  )
}

export const renderProtectedEntryBody = (entry: StreamEntry, filePath: FilePath): string =>
  render(<>{renderEntryBody(entry, filePath, true)}</>)

export const renderStreamEntry = (
  entry: StreamEntry,
  filePath: FilePath,
  options: StreamEntryRenderOptions,
): ComponentChild => {
  const tags = Array.isArray(entry.metadata.tags) ? entry.metadata.tags : []
  const socials =
    entry.metadata.socials && typeof entry.metadata.socials === 'object'
      ? (entry.metadata.socials as Record<string, unknown>)
      : null

  const timestampAttr =
    typeof options.timestampValue === 'number' ? String(options.timestampValue) : undefined

  const showDate = options.showDate !== undefined ? options.showDate : true
  const showWordCount = options.showWordCount !== undefined ? options.showWordCount : false

  const resolvedIsoDate = options.resolvedIsoDate ?? entry.date
  const formattedDate = showDate ? formatStreamDate(resolvedIsoDate) : null
  const ariaLabel = formattedDate ? formattedDate : undefined

  const onPath = timestampAttr ? buildOnPath(resolvedIsoDate) : null

  const protectedEntry = isProtectedEntry(entry)
  const privateEntry = isPrivateEntry(entry)
  const mode = options.mode ?? 'listing'
  const protectedPrompt = options.protectedPrompt ?? 'form'
  const restrictedLabel = privateEntry ? 'private' : 'locked'

  let body: ComponentChild
  if (!protectedEntry && !privateEntry) {
    body = (
      <>
        {entry.title && <h2 class="stream-entry-title">{entry.title}</h2>}
        {renderEntryBody(entry, filePath, showWordCount)}
      </>
    )
  } else if (!privateEntry && mode === 'daily' && options.encryptedPayload) {
    const encryptedContent = encodeURIComponent(JSON.stringify(options.encryptedPayload))

    body = (
      <>
        {entry.title && <h2 class="stream-entry-title">{entry.title}</h2>}
        {protectedPrompt === 'icon' ? (
          <div
            class="protected-content-wrapper inline compact"
            data-protected="true"
            data-slug={entry.id}
            data-encrypted-content={encryptedContent}
          >
            <button
              type="button"
              class="stream-protected-lock"
              data-protected-unlock-trigger
              aria-label="unlock protected entry"
            >
              <StreamUnlockIcon />
            </button>
          </div>
        ) : (
          <div
            class="protected-content-wrapper inline"
            data-protected="true"
            data-slug={entry.id}
            data-encrypted-content={encryptedContent}
          >
            <div class="password-prompt-overlay" style="display: flex;">
              <div class="password-prompt-container">
                <p>this content is protected</p>
                <form class="password-form">
                  <input
                    class="password-input"
                    type="password"
                    placeholder="enter password"
                    autocomplete="off"
                    required
                  />
                  <button class="password-submit" type="submit">
                    unlock
                  </button>
                </form>
                <p class="password-error" style="display: none;">
                  incorrect password
                </p>
              </div>
            </div>
          </div>
        )}
      </>
    )
  } else {
    body = (
      <>
        {entry.title && <h2 class="stream-entry-title">{entry.title}</h2>}
        <div class="stream-entry-private">
          <p>{restrictedLabel}</p>
        </div>
      </>
    )
  }

  return (
    <li
      key={entry.id}
      id={entry.id}
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
              const href = typeof link === 'string' ? link : (link?.toString?.() ?? '')
              const isInternal = href.startsWith('/')
              return (
                <address key={name}>
                  <a
                    href={href}
                    target={!isInternal ? '_blank' : ''}
                    rel={!isInternal ? 'noopener noreferrer' : ''}
                    class={isInternal ? 'internal' : 'external'}
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
            <span class="stream-entry-importance-label">importance:</span>{' '}
            <span class="stream-entry-importance-value">{entry.importance}</span>
          </div>
        )}
      </div>
      <div class="stream-entry-body">{body}</div>
    </li>
  )
}
