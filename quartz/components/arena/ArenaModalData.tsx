import type { ElementContent, Root } from 'hast'
import type { ComponentChild } from 'preact'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import type { ArenaChannel, ArenaBlock } from '../../plugins/transformers/arena'
import type { QuartzComponentProps } from '../../types/component'
import { fromHtmlStringToArenaJsx, arenaBlockTimestamp } from '../../util/arena'
import {
  arenaPdfFilenameFromUrl,
  arenaPdfViewerSource,
  type ArenaExternalEmbedMode,
} from '../../util/arena-embed'
import { FullSlug, slugTag, resolveRelative, joinSegments } from '../../util/path'
import { hostnameMatches } from '../../util/url'
import ArenaModalUrlBar from './ArenaModalUrlBar'

const substackPostRegex = /^https?:\/\/[^/]+\/p\/[^/]+/i

const CONSUMED_METADATA_KEYS = new Set([
  'accessed',
  'accessed_date',
  'date',
  'tags',
  'tag',
  'coord',
  'socials',
])

type ArenaModalMapProps = { coordinates?: ArenaBlock['coordinates']; title?: string }

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
  isSubstackCandidate: boolean
  isPdfCandidate: boolean
  targetUrl?: string
  frameTitle: string
  externalEmbedMode: ArenaExternalEmbedMode
  convertFromText: (html: string) => ComponentChild
  textContent?: ComponentChild
}

const ArenaModalMainContent = ({
  block,
  embedHtml,
  isSubstackCandidate,
  isPdfCandidate,
  targetUrl,
  frameTitle,
  externalEmbedMode,
  convertFromText,
  textContent,
}: ArenaModalMainContentProps) => {
  let content: ComponentChild

  if (embedHtml) {
    content = convertFromText(embedHtml)
  } else if (isSubstackCandidate && block.url) {
    content = (
      <div class="arena-modal-embed arena-modal-embed-substack" data-substack-url={block.url}>
        <span class="arena-loading-spinner" role="status" aria-label="Loading Substack preview" />
      </div>
    )
  } else if (isPdfCandidate && block.url) {
    content = (
      <div
        class="arena-modal-embed arena-modal-embed-pdf pdf-embed"
        data-arena-pdf-src={arenaPdfViewerSource(block.url)}
        data-pdf-title={arenaPdfFilenameFromUrl(block.url)}
        data-pdf-fit="page"
        tabIndex={0}
      >
        <span class="pdf-embed-loading">Loading PDF</span>
      </div>
    )
  } else if (targetUrl) {
    content = (
      <div
        class="arena-modal-external-host"
        data-block-id={block.id}
        data-arena-url={targetUrl}
        data-arena-title={frameTitle}
        data-arena-embed-mode={externalEmbedMode}
      >
        <span class="arena-embed-loading" role="status">
          loading preview
        </span>
      </div>
    )
  } else if (block.internalSlug) {
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
  } else {
    content = <div class="arena-modal-text-preview">{textContent ?? block.content}</div>
  }

  return <div class="arena-modal-main-content">{content}</div>
}

const rewriteArxivUrl = (rawUrl: string): string => {
  const parsed = new URL(rawUrl)
  if (!hostnameMatches(parsed, 'arxiv.org')) {
    return rawUrl
  }

  const pathSegments = parsed.pathname.split('/').filter(Boolean)
  if (pathSegments.length === 0) {
    return rawUrl
  }

  const [head, ...rest] = pathSegments
  if (rest.length === 0) {
    return rawUrl
  }

  const normalizedHead = head.toLowerCase()
  const remainder = rest.join('/')
  const suffix = `${parsed.search}${parsed.hash}`

  if (normalizedHead === 'pdf' || normalizedHead === 'html' || normalizedHead === 'abs') {
    const sanitized = remainder.replace(/\.pdf$/i, '')
    return `https://alphaxiv.org/abs/${sanitized}${suffix}`
  }

  return `https://alphaxiv.org/abs/${[head, ...rest].join('/')}${suffix}`
}

export interface ArenaModalDataProps {
  block: ArenaBlock
  blockIndex: number
  channel: ArenaChannel
  componentData: QuartzComponentProps
  accessed?: { display: string; dateTime?: string }
  isPdfCandidate: boolean
  view: 'modal' | 'page'
  jsxFromNode: (node?: ElementContent) => ComponentChild | undefined
  renderInlineText: (text: string) => ComponentChild
}

export default function ArenaModalData({
  block,
  blockIndex,
  channel,
  componentData,
  accessed,
  isPdfCandidate,
  view,
  jsxFromNode,
  renderInlineText,
}: ArenaModalDataProps) {
  const { fileData, cfg } = componentData
  const hasSubItems = block.subItems && block.subItems.length > 0
  const frameTitle = block.title ?? block.content ?? `Block ${blockIndex + 1}`
  const targetUrl = block.url ? rewriteArxivUrl(block.url) : undefined
  const embedHtml = block.embedHtml
  const isSubstackCandidate = block.url ? substackPostRegex.test(block.url) : false
  const externalEmbedMode = block.embedMode ?? 'auto'
  const displayUrl =
    block.url ?? (block.internalSlug ? `https://${cfg.baseUrl}/${block.internalSlug}` : undefined)
  const mapTitle = block.title ?? block.content ?? block.url ?? undefined

  const metadataEntries: Array<{ label: string; value: ComponentChild }> = []

  if (accessed) {
    metadataEntries.push({
      label: 'accessed',
      value: accessed.dateTime ? (
        <time dateTime={accessed.dateTime}>{accessed.display}</time>
      ) : (
        accessed.display
      ),
    })
  }

  if (block.importance !== undefined) {
    metadataEntries.push({
      label: 'importance',
      value: <span class="arena-meta-importance">{block.importance}</span>,
    })
  }

  if (block.metadata) {
    const additionalEntries = Object.entries(block.metadata)
      .filter((entry): entry is [string, string] => {
        const [key, value] = entry
        if (typeof value !== 'string' || value.trim().length === 0) return false
        if (CONSUMED_METADATA_KEYS.has(key.toLowerCase())) return false
        return true
      })
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([rawKey, rawValue]) => ({
        label: rawKey.replace(/_/g, ' '),
        value: renderInlineText(rawValue),
      }))

    metadataEntries.push(...additionalEntries)
  }

  if (block.tags && block.tags.length > 0) {
    metadataEntries.push({
      label: block.tags.length === 1 ? 'tag' : 'tags',
      value: (
        <span class="arena-meta-taglist">
          {block.tags.map(tag => (
            <span class="tag-link" key={`${block.id}-tag-${slugTag(tag)}`}>
              {tag}
            </span>
          ))}
        </span>
      ),
    })
  }

  const socials =
    block.metadata?.socials && typeof block.metadata.socials === 'object'
      ? (block.metadata.socials as Record<string, unknown>)
      : null

  if (socials && Object.keys(socials).length > 0) {
    metadataEntries.push({
      label: 'média',
      value: (
        <span class="arena-meta-socials">
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
  return (
    <div
      class="arena-block-modal-data"
      id={`arena-modal-data-${block.id}`}
      data-block-id={block.id}
      data-entry-id={block.entryId}
      data-channel-slug={channel.slug}
      data-sidebar-collapsed={
        typeof block.metadata?.sidebar === 'string' &&
        ['false', '0'].includes(block.metadata.sidebar.trim().toLowerCase())
          ? 'true'
          : undefined
      }
      style={view === 'modal' ? 'display: none;' : undefined}
    >
      <div class="arena-modal-layout">
        <div class="arena-modal-main">
          {displayUrl && <ArenaModalUrlBar block={block} displayUrl={displayUrl} />}
          {block.coordinates ? (
            <ArenaModalMap coordinates={block.coordinates} title={mapTitle} />
          ) : (
            <ArenaModalMainContent
              block={block}
              embedHtml={embedHtml}
              isSubstackCandidate={view === 'modal' && isSubstackCandidate}
              isPdfCandidate={isPdfCandidate}
              targetUrl={targetUrl}
              frameTitle={frameTitle}
              externalEmbedMode={externalEmbedMode}
              convertFromText={convertFromText}
              textContent={block.htmlNode ? jsxFromNode(block.htmlNode) : undefined}
            />
          )}
        </div>
        <div
          class="arena-modal-sidebar"
          id={view === 'page' && block.entryId ? `arena-entry-details-${block.entryId}` : undefined}
        >
          <div class="arena-modal-info">
            <h3 class="arena-modal-title">
              {block.titleHtmlNode
                ? jsxFromNode(block.titleHtmlNode)
                : renderInlineText(block.title ?? '')}
            </h3>
            {view === 'modal' && block.entryId && (
              <a
                class="arena-entry-permalink internal"
                href={resolveRelative(
                  fileData.slug! as FullSlug,
                  joinSegments('arena', channel.slug, block.entryId) as FullSlug,
                )}
                data-no-popover
              >
                open entry
              </a>
            )}
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
                  .map(subItem => (
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
  )
}
