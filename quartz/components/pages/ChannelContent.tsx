import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../../types/component"
import { ArenaChannel, ArenaBlock } from "../../plugins/transformers/arena"
import { classNames } from "../../util/lang"
import type { ElementContent, Root } from "hast"
import type { ComponentChild } from "preact"
import style from "../styles/arena.scss"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
import { FullSlug, slugTag, resolveRelative } from "../../util/path"
import { toArenaJsx, fromHtmlStringToArenaJsx, arenaBlockTimestamp } from "../../util/arena"
import { createWikilinkRegex, parseWikilink, resolveWikilinkTarget } from "../../util/wikilinks"
import { buildYouTubeEmbed, type YouTubeEmbedSpec } from "../../util/youtube"

const substackPostRegex = /^https?:\/\/[^/]+\/p\/[^/]+/i
const pdfUrlRegex = /\.pdf(?:[?#].*)?$/i

const extractFilenameFromUrl = (url: string): string => {
  const urlObj = new URL(url)
  const pathname = urlObj.pathname
  const parts = pathname.split("/")
  const lastPart = parts[parts.length - 1]
  return lastPart || "document.pdf"
}

const isPdfUrl = (url: string): boolean => pdfUrlRegex.test(url)

type ArenaModalMapProps = {
  coordinates?: ArenaBlock["coordinates"]
  title?: string
}

const ArenaModalMap = ({ coordinates, title }: ArenaModalMapProps) => {
  if (!coordinates) {
    return null
  }

  return (
    <div class="arena-modal-map-wrapper">
      <div
        class="arena-modal-map"
        data-map-lon={coordinates.lon.toString()}
        data-map-lat={coordinates.lat.toString()}
        data-map-title={title && title.length > 0 ? title : undefined}
      />
    </div>
  )
}

type ArenaModalMainContentProps = {
  block: ArenaBlock
  embedHtml?: string
  youtubeEmbed?: YouTubeEmbedSpec
  isSubstackCandidate: boolean
  isPdfCandidate: boolean
  targetUrl?: string
  frameTitle: string
  convertFromText: (html: string) => ComponentChild
}

const ArenaModalMainContent = ({
  block,
  embedHtml,
  youtubeEmbed,
  isSubstackCandidate,
  isPdfCandidate,
  targetUrl,
  frameTitle,
  convertFromText,
}: ArenaModalMainContentProps) => {
  let content: ComponentChild

  if (embedHtml) {
    content = convertFromText(embedHtml)
  } else if (youtubeEmbed) {
    content = (
      <iframe
        class={classNames(undefined, "arena-modal-iframe", "arena-modal-iframe-youtube")}
        title={`YouTube embed: ${frameTitle}`}
        loading="lazy"
        data-block-id={block.id}
        src={youtubeEmbed.src}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowFullScreen
        referrerPolicy="strict-origin-when-cross-origin"
      />
    )
  } else if (isSubstackCandidate && block.url) {
    content = (
      <div class="arena-modal-embed arena-modal-embed-substack" data-substack-url={block.url}>
        <span class="arena-loading-spinner" role="status" aria-label="Loading Substack preview" />
      </div>
    )
  } else if (isPdfCandidate && block.url) {
    content = (
      <div
        class="arena-modal-embed arena-modal-embed-pdf"
        data-pdf-url={block.url}
        data-pdf-filename={extractFilenameFromUrl(block.url)}
      >
        <div class="arena-pdf-controls" style="display: none;">
          <div class="arena-pdf-controls-group">
            <button type="button" class="arena-pdf-btn arena-pdf-prev" disabled>
              <svg
                width="12"
                height="12"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z"
                  fill="currentColor"
                />
              </svg>
            </button>
            <span class="arena-pdf-page-info">
              <input
                type="number"
                class="arena-pdf-page-input"
                value="1"
                min="1"
                id="arena-pdf-input"
                name="arena block pdf input"
              />{" "}
              / <span class="arena-pdf-total">0</span>
            </span>
            <button type="button" class="arena-pdf-btn arena-pdf-next" disabled>
              <svg
                width="12"
                height="12"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
          <div class="arena-pdf-controls-group">
            <button type="button" class="arena-pdf-btn arena-pdf-zoom-out" title="Zoom out">
              <svg
                width="12"
                height="12"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M3.5 7C3.22386 7 3 7.22386 3 7.5C3 7.77614 3.22386 8 3.5 8H11.5C11.7761 8 12 7.77614 12 7.5C12 7.22386 11.7761 7 11.5 7H3.5Z"
                  fill="currentColor"
                />
              </svg>
            </button>
            <span class="arena-pdf-zoom-level">300%</span>
            <button type="button" class="arena-pdf-btn arena-pdf-zoom-in" title="Zoom in">
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
                />
              </svg>
            </button>
            <button type="button" class="arena-pdf-btn arena-pdf-download" title="Download PDF">
              <svg
                width="12"
                height="12"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M7.50005 1.04999C7.74858 1.04999 7.95005 1.25146 7.95005 1.49999V8.41359L10.1819 6.18179C10.3576 6.00605 10.6425 6.00605 10.8182 6.18179C10.994 6.35753 10.994 6.64245 10.8182 6.81819L7.81825 9.81819C7.64251 9.99392 7.35759 9.99392 7.18185 9.81819L4.18185 6.81819C4.00611 6.64245 4.00611 6.35753 4.18185 6.18179C4.35759 6.00605 4.64251 6.00605 4.81825 6.18179L7.05005 8.41359V1.49999C7.05005 1.25146 7.25152 1.04999 7.50005 1.04999ZM2.5 10C2.77614 10 3 10.2239 3 10.5V12C3 12.5539 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2239 12.2239 10 12.5 10C12.7761 10 13 10.2239 13 10.5V12C13 13.1041 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2239 2.22386 10 2.5 10Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
        </div>
        <div class="arena-pdf-canvas-wrapper">
          <canvas class="arena-pdf-canvas" />
        </div>
        <span class="arena-loading-spinner" role="status" aria-label="Loading PDF viewer" />
      </div>
    )
  } else if (block.embedDisabled && targetUrl) {
    content = (
      <div class="arena-iframe-error">
        <div class="arena-iframe-error-content">
          <p>embedded content unavailable</p>
          <a
            href={targetUrl}
            target="_blank"
            rel="noopener noreferrer"
            class="arena-iframe-error-link"
          >
            open in new tab →
          </a>
        </div>
      </div>
    )
  } else if (targetUrl) {
    content = (
      <iframe
        class="arena-modal-iframe"
        title={`Embedded block: ${frameTitle}`}
        loading="lazy"
        data-block-id={block.id}
        sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms"
        src={targetUrl}
      />
    )
  } else {
    content = (
      <div
        class="arena-modal-internal-host"
        data-block-id={block.id}
        data-internal-slug={block.internalSlug}
        data-internal-href={block.internalHref}
        data-internal-hash={block.internalHash}
      >
        <div class="arena-modal-internal-preview grid" />
      </div>
    )
  }

  return <div class="arena-modal-main-content">{content}</div>
}

const rewriteArxivUrl = (rawUrl: string): string => {
  const parsed = new URL(rawUrl)
  if (!parsed.hostname.toLowerCase().endsWith("arxiv.org")) {
    return rawUrl
  }

  const pathSegments = parsed.pathname.split("/").filter(Boolean)
  if (pathSegments.length === 0) {
    return rawUrl
  }

  const [head, ...rest] = pathSegments
  if (rest.length === 0) {
    return rawUrl
  }

  const normalizedHead = head.toLowerCase()
  const remainder = rest.join("/")
  const suffix = `${parsed.search}${parsed.hash}`

  if (normalizedHead === "pdf" || normalizedHead === "html" || normalizedHead === "abs") {
    const sanitized = remainder.replace(/\.pdf$/i, "")
    return `https://alphaxiv.org/abs/${sanitized}${suffix}`
  }

  return `https://alphaxiv.org/abs/${[head, ...rest].join("/")}${suffix}`
}

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
  const formatter = new Intl.DateTimeFormat("en-US", { dateStyle: "medium" })
  const display = formatter.format(date)
  const iso = `${yearStr.padStart(4, "0")}-${monthStr.padStart(2, "0")}-${dayStr.padStart(2, "0")}`

  return { display, dateTime: iso }
}

export default (() => {
  const ChannelContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData, displayClass, cfg } = componentData
    const channel = fileData.arenaChannel as ArenaChannel | undefined

    if (!channel) {
      return <article class="arena-content">Channel not found</article>
    }

    const pinnedBlocks = channel.blocks
      .filter((b) => b.pinned === true)
      .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))

    const laterBlocks = channel.blocks
      .filter((b) => b.later === true && b.pinned !== true)
      .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))

    const regularBlocks = channel.blocks
      .filter((b) => b.pinned !== true && b.later !== true)
      .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))

    const channelViewPreferenceRaw = channel.metadata?.["view"] ?? channel.metadata?.["layout"]
    const channelViewPreference =
      typeof channelViewPreferenceRaw === "string"
        ? channelViewPreferenceRaw.trim().toLowerCase()
        : undefined
    const defaultViewMode: "grid" | "list" =
      channelViewPreference === "list" || channelViewPreference === "lists" ? "list" : "grid"
    const isListDefault = defaultViewMode === "list"

    const jsxFromNode = (node?: ElementContent) =>
      node
        ? toArenaJsx(fileData.filePath!, node, fileData.slug! as FullSlug, componentData)
        : undefined

    const renderInlineText = (text: string) => {
      if (!text) return ""
      const parts: ComponentChild[] = []
      const regex = createWikilinkRegex()
      let lastIndex = 0
      let match: RegExpExecArray | null

      while ((match = regex.exec(text)) !== null) {
        const start = match.index
        if (start > lastIndex) {
          parts.push(text.slice(lastIndex, start))
        }

        const parsed = parseWikilink(match[0])
        const resolved = parsed ? resolveWikilinkTarget(parsed, "" as FullSlug) : null

        if (parsed && resolved) {
          const hrefBase = resolveRelative(fileData.slug! as FullSlug, resolved.slug)
          const href = parsed.anchor ? `${hrefBase}${parsed.anchor}` : hrefBase
          parts.push(
            <a
              href={href}
              class="internal"
              data-no-popover
              data-slug={resolved.slug}
              key={`wikilink-${parts.length}`}
            >
              {parsed.alias ?? parsed.target ?? match[0]}
            </a>,
          )
        } else {
          parts.push(parsed?.alias ?? parsed?.target ?? match[0])
        }

        lastIndex = regex.lastIndex
      }

      if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex))
      }

      if (parts.length === 1) {
        return parts[0]
      }

      return <>{parts}</>
    }

    const renderBlock = (block: ArenaBlock, blockIndex: number) => {
      const hasSubItems = block.subItems && block.subItems.length > 0
      const frameTitle = block.title ?? block.content ?? `Block ${blockIndex + 1}`
      const resolvedUrl = block.url ? rewriteArxivUrl(block.url) : undefined
      const targetUrl = block.url ? (resolvedUrl ?? block.url) : undefined
      // No srcDoc fallback for internal links; we'll hydrate with popover-hint content
      const embedHtml = block.embedHtml
      const isSubstackCandidate = block.url ? substackPostRegex.test(block.url) : false
      const isPdfCandidate = block.url ? isPdfUrl(block.url) : false
      const isPinned = block.pinned ?? false
      const youtubeEmbed = block.url ? buildYouTubeEmbed(block.url) : undefined
      const accessedRaw =
        (typeof block.metadata?.accessed === "string" ? block.metadata.accessed : undefined) ??
        (typeof block.metadata?.accessed_date === "string"
          ? block.metadata.accessed_date
          : undefined) ??
        (typeof block.metadata?.date === "string" ? block.metadata.date : undefined)
      const accessed = accessedRaw ? normalizeDate(accessedRaw) : undefined
      const displayUrl =
        block.url ??
        (block.internalSlug ? `https://${cfg.baseUrl}/${block.internalSlug}` : undefined)
      const mapTitle = block.title ?? block.content ?? block.url ?? undefined

      const metadataEntries: Array<{ label: string; value: ComponentChild }> = []

      if (accessed) {
        metadataEntries.push({
          label: "accessed",
          value: accessed.dateTime ? (
            <time dateTime={accessed.dateTime}>{accessed.display}</time>
          ) : (
            accessed.display
          ),
        })
      }

      if (block.importance !== undefined) {
        metadataEntries.push({
          label: "importance",
          value: <span class="arena-meta-importance">{block.importance}</span>,
        })
      }

      if (block.metadata) {
        const consumedKeys = new Set([
          "accessed",
          "accessed_date",
          "date",
          "tags",
          "tag",
          "coord",
          "socials",
        ])
        const additionalEntries = Object.entries(block.metadata)
          .filter((entry): entry is [string, string] => {
            const [key, value] = entry
            if (typeof value !== "string" || value.trim().length === 0) return false
            if (consumedKeys.has(key.toLowerCase())) return false
            return true
          })
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([rawKey, rawValue]) => ({
            label: rawKey.replace(/_/g, " "),
            value: renderInlineText(rawValue),
          }))

        metadataEntries.push(...additionalEntries)
      }

      if (block.tags && block.tags.length > 0) {
        metadataEntries.push({
          label: block.tags.length === 1 ? "tag" : "tags",
          value: (
            <span class="arena-meta-taglist">
              {block.tags.map((tag) => (
                <span class="tag-link" key={`${block.id}-tag-${slugTag(tag)}`}>
                  {tag}
                </span>
              ))}
            </span>
          ),
        })
      }

      const socials =
        block.metadata?.socials && typeof block.metadata.socials === "object"
          ? (block.metadata.socials as Record<string, unknown>)
          : null

      if (socials && Object.keys(socials).length > 0) {
        metadataEntries.push({
          label: "média",
          value: (
            <span class="arena-meta-socials">
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
            </span>
          ),
        })
      }

      const convertFromText = (text: string) => {
        const root = fromHtmlIsomorphic(text, { fragment: true }) as Root
        return fromHtmlStringToArenaJsx(
          fileData.filePath!,
          root,
          fileData.slug! as FullSlug,
          componentData,
        )
      }

      const hasMetaPreview =
        Boolean(accessed) || (block.tags && Array.isArray(block.tags) && block.tags.length > 0)

      return (
        <div
          key={block.id}
          class={classNames(
            displayClass,
            "arena-block",
            block.highlighted ? "highlighted" : "",
            isPinned ? "pinned" : "",
          )}
          id={`arena-block-${block.id}`}
          data-block-id={block.id}
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
          <div
            class="arena-block-clickable"
            role="button"
            tabIndex={0}
            aria-label="View block details"
          >
            <div class="arena-block-content">
              {block.titleHtmlNode
                ? jsxFromNode(block.titleHtmlNode)
                : renderInlineText(block.title || block.content || "")}
            </div>
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
                    {block.tags.map((tag) => (
                      <span class="tag-link" key={`${block.id}-tag-preview-${slugTag(tag)}`}>
                        {tag}
                      </span>
                    ))}
                  </span>
                ) : null}
              </div>
            )}
          </div>
          <div
            class="arena-block-modal-data"
            id={`arena-modal-data-${block.id}`}
            data-block-id={block.id}
            data-channel-slug={channel.slug}
            style="display: none;"
          >
            <div class="arena-modal-layout">
              <div class="arena-modal-main">
                {displayUrl && (
                  <div class="arena-modal-url-bar">
                    <button
                      type="button"
                      class="arena-url-copy-button"
                      data-url={displayUrl}
                      role="button"
                      tabIndex={0}
                      aria-label="Copy URL to clipboard"
                    >
                      <svg
                        width="15"
                        height="15"
                        viewBox="0 0 15 15"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        class="copy-icon"
                      >
                        <path
                          d="M7.49996 1.80002C4.35194 1.80002 1.79996 4.352 1.79996 7.50002C1.79996 10.648 4.35194 13.2 7.49996 13.2C10.648 13.2 13.2 10.648 13.2 7.50002C13.2 4.352 10.648 1.80002 7.49996 1.80002ZM0.899963 7.50002C0.899963 3.85494 3.85488 0.900024 7.49996 0.900024C11.145 0.900024 14.1 3.85494 14.1 7.50002C14.1 11.1451 11.145 14.1 7.49996 14.1C3.85488 14.1 0.899963 11.1451 0.899963 7.50002Z"
                          fill="currentColor"
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                        />
                        <path
                          d="M13.4999 7.89998H1.49994V7.09998H13.4999V7.89998Z"
                          fill="currentColor"
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                        />
                        <path
                          d="M7.09991 13.5V1.5H7.89991V13.5H7.09991zM10.375 7.49998C10.375 5.32724 9.59364 3.17778 8.06183 1.75656L8.53793 1.24341C10.2396 2.82218 11.075 5.17273 11.075 7.49998 11.075 9.82724 10.2396 12.1778 8.53793 13.7566L8.06183 13.2434C9.59364 11.8222 10.375 9.67273 10.375 7.49998zM3.99969 7.5C3.99969 5.17611 4.80786 2.82678 6.45768 1.24719L6.94177 1.75281C5.4582 3.17323 4.69969 5.32389 4.69969 7.5 4.6997 9.67611 5.45822 11.8268 6.94179 13.2472L6.45769 13.7528C4.80788 12.1732 3.9997 9.8239 3.99969 7.5z"
                          fill="currentColor"
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                        />
                        <path
                          d="M7.49996 3.95801C9.66928 3.95801 11.8753 4.35915 13.3706 5.19448 13.5394 5.28875 13.5998 5.50197 13.5055 5.67073 13.4113 5.83948 13.198 5.89987 13.0293 5.8056 11.6794 5.05155 9.60799 4.65801 7.49996 4.65801 5.39192 4.65801 3.32052 5.05155 1.97064 5.8056 1.80188 5.89987 1.58866 5.83948 1.49439 5.67073 1.40013 5.50197 1.46051 5.28875 1.62927 5.19448 3.12466 4.35915 5.33063 3.95801 7.49996 3.95801zM7.49996 10.85C9.66928 10.85 11.8753 10.4488 13.3706 9.6135 13.5394 9.51924 13.5998 9.30601 13.5055 9.13726 13.4113 8.9685 13.198 8.90812 13.0293 9.00238 11.6794 9.75643 9.60799 10.15 7.49996 10.15 5.39192 10.15 3.32052 9.75643 1.97064 9.00239 1.80188 8.90812 1.58866 8.9685 1.49439 9.13726 1.40013 9.30601 1.46051 9.51924 1.62927 9.6135 3.12466 10.4488 5.33063 10.85 7.49996 10.85z"
                          fill="currentColor"
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                        />
                      </svg>
                      <svg
                        width="15"
                        height="15"
                        viewBox="-2 -2 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        class="check-icon"
                      >
                        <use href="#github-check" />
                      </svg>
                    </button>
                    {block.url ? (
                      <a
                        href={block.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        class="arena-modal-link"
                      >
                        <div class="arena-modal-link-text">{displayUrl}</div>
                        <svg
                          width="15"
                          height="15"
                          viewBox="0 0 15 15"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            fill-rule="evenodd"
                            clip-rule="evenodd"
                            d="M12 13C12.5523 13 13 12.5523 13 12V3C13 2.44771 12.5523 2 12 2H3C2.44771 2 2 2.44771 2 3V6.5C2 6.77614 2.22386 7 2.5 7C2.77614 7 3 6.77614 3 6.5V3H12V12H8.5C8.22386 12 8 12.2239 8 12.5C8 12.7761 8.22386 13 8.5 13H12ZM9 6.5C9 6.5001 9 6.50021 9 6.50031V6.50035V9.5C9 9.77614 8.77614 10 8.5 10C8.22386 10 8 9.77614 8 9.5V7.70711L2.85355 12.8536C2.65829 13.0488 2.34171 13.0488 2.14645 12.8536C1.95118 12.6583 1.95118 12.3417 2.14645 12.1464L7.29289 7H5.5C5.22386 7 5 6.77614 5 6.5C5 6.22386 5.22386 6 5.5 6H8.5C8.56779 6 8.63244 6.01349 8.69139 6.03794C8.74949 6.06198 8.80398 6.09744 8.85143 6.14433C8.94251 6.23434 8.9992 6.35909 8.99999 6.49708L8.99999 6.49738"
                            fill="currentColor"
                          />
                        </svg>
                      </a>
                    ) : (
                      <span class="arena-modal-link">
                        <div class="arena-modal-link-text">{displayUrl}</div>
                      </span>
                    )}
                  </div>
                )}
                {block.coordinates ? (
                  <ArenaModalMap coordinates={block.coordinates} title={mapTitle} />
                ) : (
                  <ArenaModalMainContent
                    block={block}
                    embedHtml={embedHtml}
                    youtubeEmbed={youtubeEmbed}
                    isSubstackCandidate={isSubstackCandidate}
                    isPdfCandidate={isPdfCandidate}
                    targetUrl={targetUrl}
                    frameTitle={frameTitle}
                    convertFromText={convertFromText}
                  />
                )}
              </div>
              <div class="arena-modal-sidebar">
                <div class="arena-modal-info">
                  <h3 class="arena-modal-title">
                    {block.titleHtmlNode
                      ? jsxFromNode(block.titleHtmlNode)
                      : renderInlineText(block.title ?? "")}
                  </h3>
                  {metadataEntries.length > 0 && (
                    <div class="arena-modal-meta">
                      {metadataEntries.map(({ label, value }, index) => (
                        <div class="arena-meta-item" key={`${label}-${index}`}>
                          <span class="arena-meta-label">{label}</span>
                          <em class="arena-meta-value">{value}</em>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {block.internalSlug && block.internalHref && (
                  <div class="arena-modal-wikilink-trail">
                    <div class="arena-modal-wikilink-trail-header">
                      <svg
                        width="15"
                        height="15"
                        viewBox="0 0 15 15"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        class="arena-modal-wikilink-icon"
                      >
                        <path
                          d="M8.51194 3.00541C9.18829 2.54594 10.0435 2.53694 10.6788 2.95419C10.8231 3.04893 10.9771 3.1993 11.389 3.61119C11.8009 4.02307 11.9513 4.17714 12.046 4.32141C12.4633 4.95675 12.4543 5.81192 11.9948 6.48827C11.8899 6.64264 11.7276 6.80811 11.3006 7.23511L10.6819 7.85383C10.4867 8.04909 10.4867 8.36567 10.6819 8.56093C10.8772 8.7562 11.1938 8.7562 11.389 8.56093L12.0077 7.94221L12.0507 7.89929C12.4203 7.52976 12.6568 7.2933 12.8071 7.06391C13.5039 6.00296 13.5365 4.65788 12.8982 3.57295C12.7449 3.32991 12.5229 3.10593 12.1334 2.71636L12.1334 2.71636L12.1034 2.68638C11.7138 2.29681 11.4898 2.07484 11.2468 1.92147C10.1618 1.28319 8.81676 1.31576 7.75592 2.01257C7.52656 2.16287 7.2901 2.39932 6.92058 2.76884L6.37669 3.31272C6.18143 3.50798 6.18143 3.82456 6.37669 4.01982C6.57195 4.21508 6.88853 4.21508 7.08379 4.01982L7.62768 3.47594C8.0547 3.04892 8.22015 2.88638 8.37452 2.78141L8.51194 3.00541ZM4.01982 7.08379C4.21508 6.88853 4.21508 6.57195 4.01982 6.37669C3.82456 6.18143 3.50798 6.18143 3.31272 6.37669L2.68638 7.00303C2.31686 7.37255 2.07044 7.62282 1.92147 7.87584C1.31555 8.86962 1.31555 10.1304 1.92147 11.1242C2.07444 11.3772 2.31686 11.6273 2.68638 11.997C3.05589 12.3665 3.30617 12.6089 3.55918 12.7619C4.55296 13.3678 5.81372 13.3678 6.8075 12.7619C7.06052 12.6129 7.31079 12.3665 7.68031 11.997L8.30665 11.3706C8.50191 11.1754 8.50191 10.8588 8.30665 10.6635C8.11139 10.4683 7.79481 10.4683 7.59955 10.6635L6.97321 11.2899C6.54619 11.7169 6.36847 11.8931 6.19973 12.0017C5.50006 12.523 4.55992 12.523 3.86025 12.0017C3.69151 11.8931 3.51379 11.7169 3.08677 11.2899C2.65975 10.8629 2.48203 10.6851 2.37343 10.5164C1.85211 9.81676 1.85211 8.87662 2.37343 8.17695C2.48203 8.00821 2.65975 7.83049 3.08677 7.40347L3.71311 6.77713C3.90837 6.58187 3.90837 6.26529 3.71311 6.07003C3.51785 5.87477 3.20127 5.87477 3.00601 6.07003L2.37967 6.69637L2.36862 6.70743C2.00332 7.07273 1.75807 7.31799 1.60557 7.56168C0.965424 8.60224 0.998169 9.91893 1.67901 10.9708C1.82921 11.2119 2.05208 11.4348 2.44169 11.8244L2.47167 11.8544C2.86124 12.244 3.08418 12.4669 3.32524 12.6171C4.37732 13.2979 5.69401 13.3307 6.73456 12.6905C6.97825 12.538 7.22351 12.2928 7.58881 11.9275L8.21515 11.3011C8.41041 11.1059 8.41041 10.7893 8.21515 10.594C8.01989 10.3988 7.70331 10.3988 7.50805 10.594L6.8817 11.2204C6.45468 11.6474 6.28912 11.8099 6.13476 11.9149C5.45841 12.3743 4.60324 12.3832 3.92689 11.966C3.78261 11.8713 3.62854 11.7209 3.21566 11.3081C2.80278 10.8952 2.65271 10.7411 2.55795 10.5968C2.14072 9.92042 2.14972 9.06526 2.60919 8.38891C2.71416 8.23454 2.87671 8.06898 3.30373 7.64197L4.01982 7.08379ZM12.5587 2.43424C12.7539 2.2391 12.7539 1.92252 12.5587 1.72726C12.3634 1.532 12.0468 1.532 11.8516 1.72726L7.00001 6.57881L6.99988 6.57894L5.13456 8.44426C4.93929 8.63952 4.93929 8.9561 5.13456 9.15136C5.32982 9.34662 5.6464 9.34662 5.84166 9.15136L7.70699 7.28604L7.70712 7.28591L12.5587 2.43424Z"
                          fill="currentColor"
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                        />
                      </svg>
                      <span class="arena-modal-wikilink-trail-title">linked note</span>
                    </div>
                    <div class="arena-modal-wikilink-trail-link">
                      <a
                        href={block.internalHref}
                        class="internal arena-wikilink-trail-anchor"
                        data-slug={block.internalSlug}
                        data-no-popover
                      >
                        {block.internalTitle ?? block.internalSlug}
                      </a>
                    </div>
                  </div>
                )}
                {hasSubItems && (
                  <div class="arena-modal-connections">
                    <div class="arena-modal-connections-header">
                      <span class="arena-modal-connections-title">notes</span>
                      <span class="arena-modal-connections-count">{block.subItems!.length}</span>
                    </div>
                    <ul class="arena-modal-connections-list">
                      {[...block.subItems!]
                        .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))
                        .map((subItem) => (
                          <li key={subItem.id}>
                            {subItem.htmlNode
                              ? jsxFromNode(subItem.htmlNode)
                              : subItem.titleHtmlNode
                                ? jsxFromNode(subItem.titleHtmlNode)
                                : subItem.title || subItem.content}
                          </li>
                        ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )
    }

    return (
      <article
        class="arena-channel-page main-col"
        data-view-mode={defaultViewMode}
        data-view-default={defaultViewMode}
      >
        <div class="arena-channel-controls">
          <div class="arena-search">
            <input
              type="text"
              id="arena-search-bar"
              class="arena-search-input"
              placeholder="rechercher ce canal..."
              data-search-scope="channel"
              data-channel-slug={channel.slug}
              aria-label="Rechercher ce canal"
              aria-keyshortcuts="Meta+K Control+K"
            />
            <svg
              class="arena-search-icon"
              width="18"
              height="18"
              viewBox="0 0 15 15"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M10 6.5C10 8.433 8.433 10 6.5 10C4.567 10 3 8.433 3 6.5C3 4.567 4.567 3 6.5 3C8.433 3 10 4.567 10 6.5ZM9.30884 10.0159C8.53901 10.6318 7.56251 11 6.5 11C4.01472 11 2 8.98528 2 6.5C2 4.01472 4.01472 2 6.5 2C8.98528 2 11 4.01472 11 6.5C11 7.56251 10.6318 8.53901 10.0159 9.30884L12.8536 12.1464C13.0488 12.3417 13.0488 12.6583 12.8536 12.8536C12.6583 13.0488 12.3417 13.0488 12.1464 12.8536L9.30884 10.0159Z"
                fill="currentColor"
                fill-rule="evenodd"
                clip-rule="evenodd"
              />
            </svg>
            <div id="arena-search-container" class="arena-search-results" />
          </div>
          <div class="arena-view-toggle" role="group" aria-label="Toggle channel layout">
            <button
              type="button"
              class={classNames(
                displayClass,
                "arena-view-toggle-button",
                !isListDefault ? "active" : "",
              )}
              data-view-mode="grid"
              aria-pressed={!isListDefault}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M1.5 3A1.5 1.5 0 0 1 3 1.5h2A1.5 1.5 0 0 1 6.5 3v2A1.5 1.5 0 0 1 5 6.5H3A1.5 1.5 0 0 1 1.5 5V3Zm1 0A.5.5 0 0 1 3 2.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5V3ZM8.5 3A1.5 1.5 0 0 1 10 1.5h2A1.5 1.5 0 0 1 13.5 3v2A1.5 1.5 0 0 1 12 6.5h-2A1.5 1.5 0 0 1 8.5 5V3Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5V3ZM1.5 10a1.5 1.5 0 0 1 1.5-1.5h2A1.5 1.5 0 0 1 6.5 10v2A1.5 1.5 0 0 1 5 13.5H3A1.5 1.5 0 0 1 1.5 12v-2Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5v-2ZM8.5 10A1.5 1.5 0 0 1 10 8.5h2a1.5 1.5 0 0 1 1.5 1.5v2a1.5 1.5 0 0 1-1.5 1.5h-2A1.5 1.5 0 0 1 8.5 12v-2Zm1 0a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5v-2Z"
                  fill="currentColor"
                />
              </svg>
            </button>
            <button
              type="button"
              class={classNames(
                displayClass,
                "arena-view-toggle-button",
                isListDefault ? "active" : "",
              )}
              data-view-mode="list"
              aria-pressed={isListDefault}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 15 15"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M2 4.25A.75.75 0 0 1 2.75 3.5h9.5a.75.75 0 0 1 0 1.5h-9.5A.75.75 0 0 1 2 4.25Zm0 3.5A.75.75 0 0 1 2.75 7h9.5a.75.75 0 0 1 0 1.5h-9.5A.75.75 0 0 1 2 7.75Zm.75 3.5a.75.75 0 0 0 0 1.5h9.5a.75.75 0 0 0 0-1.5h-9.5Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
        </div>
        {pinnedBlocks.length > 0 && (
          <>
            <div class="arena-section-header">
              <h3>pinned</h3>
            </div>
            <div
              class="arena-channel-grid arena-pinned-section"
              data-view-mode={defaultViewMode}
              data-view-default={defaultViewMode}
            >
              {pinnedBlocks.map((block, idx) => renderBlock(block, idx))}
            </div>
          </>
        )}
        {laterBlocks.length > 0 && (
          <>
            <div class="arena-section-header">
              <h3>later</h3>
            </div>
            <div
              class="arena-channel-grid arena-later-section"
              data-view-mode="list"
              data-view-default="list"
            >
              {laterBlocks.map((block, idx) =>
                renderBlock(block, idx + pinnedBlocks.length + regularBlocks.length),
              )}
            </div>
          </>
        )}
        {regularBlocks.length > 0 && (
          <>
            {pinnedBlocks.length > 0 && (
              <div class="arena-section-header">
                <h3>blocks</h3>
              </div>
            )}
            <div
              class="arena-channel-grid arena-blocks-section"
              id="arena-block-collection"
              data-view-mode={defaultViewMode}
              data-view-default={defaultViewMode}
            >
              {regularBlocks.map((block, idx) => renderBlock(block, idx + pinnedBlocks.length))}
            </div>
          </>
        )}

        <div class="arena-block-modal" id="arena-modal">
          <div class="arena-modal-content">
            <div class="arena-modal-nav">
              <button type="button" class="arena-modal-nav-btn arena-modal-collapse">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <line x1="4" x2="20" y1="12" y2="12" />
                  <line x1="4" x2="20" y1="6" y2="6" />
                  <line x1="4" x2="20" y1="18" y2="18" />
                </svg>
              </button>
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-prev"
                aria-label="Previous block"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
              <button
                type="button"
                class="arena-modal-nav-btn arena-modal-next"
                aria-label="Next block"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
              <button type="button" class="arena-modal-close" aria-label="Close">
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M11.7816 4.03157C12.0062 3.80702 12.0062 3.44295 11.7816 3.2184C11.5571 2.99385 11.193 2.99385 10.9685 3.2184L7.50005 6.68682L4.03164 3.2184C3.80708 2.99385 3.44301 2.99385 3.21846 3.2184C2.99391 3.44295 2.99391 3.80702 3.21846 4.03157L6.68688 7.49999L3.21846 10.9684C2.99391 11.193 2.99391 11.557 3.21846 11.7816C3.44301 12.0061 3.80708 12.0061 4.03164 11.7816L7.50005 8.31316L10.9685 11.7816C11.193 12.0061 11.5571 12.0061 11.7816 11.7816C12.0062 11.557 12.0062 11.193 11.7816 10.9684L8.31322 7.49999L11.7816 4.03157Z"
                    fill="currentColor"
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                  />
                </svg>
              </button>
            </div>
            <div class="arena-modal-body" />
          </div>
        </div>
      </article>
    )
  }

  ChannelContent.css = style

  return ChannelContent
}) satisfies QuartzComponentConstructor
