import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { ArenaChannel, ArenaBlock } from "../../plugins/transformers/arena"
import { classNames } from "../../util/lang"
import { toHtml } from "hast-util-to-html"
import type { ElementContent, Root } from "hast"
import style from "../styles/arena.scss"
// @ts-ignore
import modalScript from "../scripts/arena.inline"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
import { FullSlug } from "../../util/path"
import {
  toArenaJsx,
  fromHtmlStringToArenaJsx,
  toArenaRoot,
  arenaBlockTimestamp,
} from "../../util/arena"

const substackPostRegex = /^https?:\/\/[^/]+\/p\/[^/]+/i

type YouTubeEmbedSpec = {
  src: string
}

const isValidYouTubeVideoId = (value: string | null | undefined): value is string =>
  !!value && /^[0-9A-Za-z_-]{11}$/.test(value)

const parseYouTubeTimestamp = (value: string | null): string | undefined => {
  if (!value) return undefined

  const trimmed = value.trim()
  if (trimmed.length === 0) return undefined

  if (/^\d+$/.test(trimmed)) {
    return trimmed.replace(/^0+/, "") || "0"
  }

  let matched = false
  let totalSeconds = 0
  const durationRegex = /(\d+)([hms])/gi
  trimmed.replace(durationRegex, (_match, amount, unit) => {
    matched = true
    const numeric = Number(amount)
    if (Number.isNaN(numeric)) return ""
    switch (unit.toLowerCase()) {
      case "h":
        totalSeconds += numeric * 3600
        break
      case "m":
        totalSeconds += numeric * 60
        break
      case "s":
        totalSeconds += numeric
        break
    }
    return ""
  })

  if (matched) {
    return totalSeconds > 0 ? String(totalSeconds) : "0"
  }

  if (/^\d{1,2}(:\d{2}){1,2}$/.test(trimmed)) {
    const parts = trimmed.split(":").map((part) => Number(part))
    if (parts.some((part) => Number.isNaN(part))) {
      return undefined
    }
    let seconds = 0
    for (const part of parts) {
      seconds = seconds * 60 + part
    }
    return seconds > 0 ? String(seconds) : "0"
  }

  return undefined
}

const extractYouTubeStart = (url: URL): string | undefined => {
  const searchStart =
    parseYouTubeTimestamp(url.searchParams.get("start")) ??
    parseYouTubeTimestamp(url.searchParams.get("t"))
  if (searchStart) return searchStart

  const hash = url.hash.startsWith("#") ? url.hash.slice(1) : url.hash
  if (!hash) return undefined

  const hashParams = hash.includes("=")
    ? new URLSearchParams(hash)
    : new URLSearchParams(`t=${hash}`)
  return (
    parseYouTubeTimestamp(hashParams.get("start")) ?? parseYouTubeTimestamp(hashParams.get("t"))
  )
}

const buildYouTubeEmbed = (rawUrl: string): YouTubeEmbedSpec | undefined => {
  try {
    const url = new URL(rawUrl)
    const normalizedHost = url.hostname.toLowerCase()
    const host = normalizedHost.startsWith("www.") ? normalizedHost.slice(4) : normalizedHost
    const isKnownHost =
      host === "youtu.be" || host.endsWith("youtube.com") || host.endsWith("youtube-nocookie.com")
    if (!isKnownHost) return undefined

    const segments = url.pathname.split("/").filter(Boolean)
    let videoId: string | undefined
    let playlistId: string | undefined

    if (host === "youtu.be") {
      videoId = segments[0]
    } else if (segments[0] === "watch") {
      videoId = url.searchParams.get("v") ?? undefined
    } else if (segments[0] === "embed" && segments[1]) {
      videoId = segments[1]
    } else if (segments[0] === "shorts" && segments[1]) {
      videoId = segments[1]
    } else if (segments[0] === "live" && segments[1]) {
      videoId = segments[1]
    }

    playlistId = url.searchParams.get("list") ?? undefined
    if (!videoId && segments[0] === "playlist") {
      playlistId = playlistId ?? segments[1]
    }

    if (!videoId) {
      const candidate = url.searchParams.get("v")
      if (candidate) {
        videoId = candidate
      }
    }

    if (videoId && !isValidYouTubeVideoId(videoId)) {
      videoId = undefined
    }

    const start = extractYouTubeStart(url)
    const indexParam = url.searchParams.get("index") ?? url.searchParams.get("i") ?? undefined
    const endParam = url.searchParams.get("end") ?? undefined

    const params = new URLSearchParams()
    if (playlistId) {
      params.set("list", playlistId)
      if (indexParam && /^\d+$/.test(indexParam)) {
        params.set("index", indexParam)
      }
    }
    if (start && /^\d+$/.test(start)) {
      params.set("start", start)
    }
    if (endParam && /^\d+$/.test(endParam)) {
      params.set("end", endParam)
    }

    if (!videoId && !playlistId) {
      return undefined
    }

    const path = videoId ? `/embed/${videoId}` : "/embed/videoseries"
    const query = params.toString()
    const src = `https://www.youtube-nocookie.com${path}${query ? `?${query}` : ""}`
    return { src }
  } catch {
    return undefined
  }
}

const rewriteArxivUrl = (rawUrl: string): string => {
  try {
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

    if (normalizedHead === "pdf") {
      return `https://ar5iv.org/pdf/${remainder}${suffix}`
    }

    if (normalizedHead === "html") {
      return `https://ar5iv.org/html/${remainder}${suffix}`
    }

    if (normalizedHead === "abs") {
      const sanitized = remainder.replace(/\.pdf$/i, "")
      return `https://ar5iv.org/html/${sanitized}${suffix}`
    }

    return `https://ar5iv.org/${[head, ...rest].join("/")}${suffix}`
  } catch {
    return rawUrl
  }
}

const escapeHtml = (value: string): string =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")

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
    const { fileData, displayClass } = componentData
    const channel = fileData.arenaChannel as ArenaChannel | undefined

    if (!channel) {
      return <article class="arena-content">Channel not found</article>
    }

    const jsxFromNode = (node?: ElementContent) =>
      node
        ? toArenaJsx(fileData.filePath!, node, fileData.slug! as FullSlug, componentData)
        : undefined

    const htmlFromNode = (node?: ElementContent) => {
      if (!node) return undefined
      const root = toArenaRoot(node, fileData.slug! as FullSlug, componentData)
      return toHtml(root, { allowDangerousHtml: true })
    }

    const renderBlock = (block: ArenaBlock, blockIndex: number) => {
      const hasSubItems = block.subItems && block.subItems.length > 0
      const frameTitle = block.title ?? block.content ?? `Block ${blockIndex + 1}`
      const resolvedUrl = block.url ? rewriteArxivUrl(block.url) : undefined
      const targetUrl = block.url ? (resolvedUrl ?? block.url) : undefined
      // No srcDoc fallback for internal links; we'll hydrate with popover-hint content
      const embedHtml = block.embedHtml
      const isSubstackCandidate = block.url ? substackPostRegex.test(block.url) : false
      const youtubeEmbed = block.url ? buildYouTubeEmbed(block.url) : undefined
      const accessedRaw =
        block.metadata?.accessed ?? block.metadata?.accessed_date ?? block.metadata?.date
      const accessed = accessedRaw ? normalizeDate(accessedRaw) : undefined

      const convertFromText = (text: string) => {
        const root = fromHtmlIsomorphic(text, { fragment: true }) as Root
        return fromHtmlStringToArenaJsx(
          fileData.filePath!,
          root,
          fileData.slug! as FullSlug,
          componentData,
        )
      }

      return (
        <div
          key={block.id}
          class={classNames(displayClass, "arena-block", block.highlighted ? "highlighted" : "")}
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
          <div
            class="arena-block-clickable"
            role="button"
            tabIndex={0}
            aria-label="View block details"
          >
            <div class="arena-block-content">
              {block.titleHtmlNode
                ? jsxFromNode(block.titleHtmlNode)
                : block.title || block.content}
            </div>
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
                {block.url && (
                  <div class="arena-modal-url-bar">
                    <span
                      // @ts-ignore
                      type="button"
                      class="arena-url-copy-button"
                      data-url={block.url!}
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
                    </span>
                    <a
                      href={block.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      class="arena-modal-link"
                    >
                      <div class="arena-modal-link-text">{block.url}</div>
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
                  </div>
                )}
                <div class="arena-modal-main-content">
                  {embedHtml ? (
                    convertFromText(embedHtml as string)
                  ) : youtubeEmbed ? (
                    <iframe
                      class={classNames(
                        undefined,
                        "arena-modal-iframe",
                        "arena-modal-iframe-youtube",
                      )}
                      title={`YouTube embed: ${frameTitle}`}
                      loading="lazy"
                      data-block-id={block.id}
                      src={youtubeEmbed.src}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                      allowFullScreen
                      referrerPolicy="strict-origin-when-cross-origin"
                    />
                  ) : isSubstackCandidate && block.url ? (
                    <div
                      class="arena-modal-embed arena-modal-embed-substack"
                      data-substack-url={block.url}
                    >
                      <span
                        class="arena-loading-spinner"
                        role="status"
                        aria-label="Loading Substack preview"
                      />
                    </div>
                  ) : targetUrl ? (
                    <iframe
                      class="arena-modal-iframe"
                      title={`Embedded block: ${frameTitle}`}
                      loading="lazy"
                      data-block-id={block.id}
                      sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox"
                      src={targetUrl}
                    />
                  ) : (
                    <div class="arena-modal-internal-host" data-block-id={block.id}>
                      {block.htmlNode
                        ? jsxFromNode(block.htmlNode)
                        : block.titleHtmlNode
                          ? jsxFromNode(block.titleHtmlNode)
                          : block.title || block.content}
                      <div class="arena-modal-fallback" aria-hidden="true">
                        <svg
                          width="64"
                          height="40"
                          viewBox="0 0 64 40"
                          xmlns="http://www.w3.org/2000/svg"
                          style="display:block;opacity:0.5;margin:12px auto;"
                        >
                          <rect x="1" y="1" width="62" height="38" rx="6" fill="none" stroke="currentColor"/>
                          <rect x="8" y="10" width="48" height="4" rx="2" fill="currentColor"/>
                          <rect x="8" y="18" width="36" height="4" rx="2" fill="currentColor"/>
                          <rect x="8" y="26" width="28" height="4" rx="2" fill="currentColor"/>
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <div class="arena-modal-sidebar">
                <div class="arena-modal-info">
                  <h3 class="arena-modal-title">{block.title ?? ""}</h3>
                  {accessed && (
                    <div class="arena-modal-meta">
                      <div class="arena-meta-item">
                        <span class="arena-meta-label">accessed</span>
                        {accessed.dateTime ? (
                          <time class="arena-meta-value" dateTime={accessed.dateTime}>
                            <em>{accessed.display}</em>
                          </time>
                        ) : (
                          <span class="arena-meta-value">
                            <em>{accessed.display}</em>
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
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
      <article class="arena-channel-page main-col">
        <div class="arena-channel-grid">
          {[...channel.blocks]
            .sort((a, b) => arenaBlockTimestamp(b) - arenaBlockTimestamp(a))
            .map((block, idx) => renderBlock(block, idx))}
        </div>

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
  ChannelContent.afterDOMLoaded = modalScript

  return ChannelContent
}) satisfies QuartzComponentConstructor
