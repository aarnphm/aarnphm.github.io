import type { ArenaBlockSearchable } from './session-state'
import {
  arenaPdfFilenameFromUrl,
  arenaPdfViewerSource,
  isArenaPdfUrl,
  type ArenaExternalEmbedMode,
} from '../../../util/arena-embed'
import { EXTERNAL_LINK_ICON_PATH } from '../../../util/external-link-icon'

const SUBSTACK_POST_REGEX = /^https?:\/\/[^/]+\/p\/[^/]+/i

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, char => {
    switch (char) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      case "'":
        return '&#39;'
      default:
        return char
    }
  })
}

function renderExternalModalHtml(
  block: ArenaBlockSearchable,
  targetUrl: string,
  mode: ArenaExternalEmbedMode,
): string {
  const escapedUrl = escapeHtml(targetUrl)
  return `
    <div
      class="arena-modal-external-host"
      data-block-id="${escapeHtml(block.id)}"
      data-arena-url="${escapedUrl}"
      data-arena-title="${escapeHtml(block.title ?? block.content ?? 'Block')}"
      data-arena-embed-mode="${mode}"
    ><span class="arena-embed-loading" role="status">loading preview</span></div>
  `
}

function renderPdfModalHtml(pdfUrl: string): string {
  return `
    <div
      class="arena-modal-embed arena-modal-embed-pdf pdf-embed"
      data-arena-pdf-src="${escapeHtml(arenaPdfViewerSource(pdfUrl))}"
      data-pdf-title="${escapeHtml(arenaPdfFilenameFromUrl(pdfUrl))}"
      data-pdf-fit="page"
      tabindex="0"
    >
      <span class="pdf-embed-loading">Loading PDF</span>
    </div>
  `
}

function renderArenaSearchSubItemHtml(block: ArenaBlockSearchable): string {
  if (block.blockHtml) return block.blockHtml
  if (block.titleHtml) return block.titleHtml
  return escapeHtml(block.title ?? block.content)
}

export function createModalDataFromJson(
  block: ArenaBlockSearchable,
  channelSlug: string,
): HTMLElement {
  const container = document.createElement('div')
  container.className = 'arena-block-modal-data'
  container.id = `arena-modal-data-${block.id}`
  container.dataset.blockId = block.id
  container.dataset.channelSlug = channelSlug
  container.style.display = 'none'

  const displayUrl =
    block.url ??
    (block.internalSlug ? `${window.location.origin}/${block.internalSlug}` : undefined)
  const escapedDisplayUrl = displayUrl ? escapeHtml(displayUrl) : ''

  const metadataHtml: string[] = []

  if (block.metadata) {
    const consumedKeys = new Set(['accessed', 'accessed_date', 'date', 'tags', 'tag', 'coord'])
    const entries = Object.entries(block.metadata)
      .filter(([key, value]) => {
        if (typeof value !== 'string' || value.trim().length === 0) return false
        if (consumedKeys.has(key.toLowerCase())) return false
        return true
      })
      .sort(([a], [b]) => a.localeCompare(b))

    for (const [key, value] of entries) {
      const label = key.replace(/_/g, ' ')
      metadataHtml.push(`
        <div class="arena-meta-item">
          <span class="arena-meta-label">${escapeHtml(label)}</span>
          <em class="arena-meta-value">${escapeHtml(value)}</em>
        </div>
      `)
    }
  }

  if (block.tags && block.tags.length > 0) {
    const tagsLabel = block.tags.length === 1 ? 'tag' : 'tags'
    const tagsList = block.tags
      .map(tag => `<span class="tag-link">${escapeHtml(tag)}</span>`)
      .join('')
    metadataHtml.push(`
      <div class="arena-meta-item">
        <span class="arena-meta-label">${tagsLabel}</span>
        <em class="arena-meta-value">
          <span class="arena-meta-taglist">${tagsList}</span>
        </em>
      </div>
    `)
  }

  const subItems = block.subItems ?? []
  const subItemsHtml =
    subItems.length > 0
      ? `
    <div class="arena-modal-connections">
      <div class="arena-modal-connections-header">
        <span class="arena-modal-connections-title">notes</span>
        <span class="arena-modal-connections-count">${subItems.length}</span>
      </div>
      <ul class="arena-modal-connections-list">
        ${subItems
          .map(
            subItem => `
          <li>${renderArenaSearchSubItemHtml(subItem)}</li>
        `,
          )
          .join('')}
      </ul>
    </div>
  `
      : ''

  const mapTitle = block.title || block.content || block.url || ''
  const mapHtml = block.coordinates
    ? `
        <div class="arena-modal-map-wrapper">
          <div
            class="arena-modal-map"
            data-map-lon="${String(block.coordinates.lon)}"
            data-map-lat="${String(block.coordinates.lat)}"${mapTitle.trim().length > 0 ? ` data-map-title="${escapeHtml(mapTitle)}"` : ''}
          ></div>
        </div>
      `
    : ''

  let mainContentHtml = ''
  if (block.embedHtml) {
    mainContentHtml = block.embedHtml
  } else if (block.url && SUBSTACK_POST_REGEX.test(block.url)) {
    const escapedUrl = escapeHtml(block.url)
    mainContentHtml = `
      <div class="arena-modal-embed arena-modal-embed-substack" data-substack-url="${escapedUrl}">
        <span class="arena-loading-spinner" role="status" aria-label="Loading Substack preview"></span>
      </div>
    `
  } else if (block.url && isArenaPdfUrl(block.url)) {
    mainContentHtml = renderPdfModalHtml(block.url)
  } else if (block.url) {
    const externalEmbedMode = block.embedMode ?? 'auto'
    mainContentHtml = renderExternalModalHtml(block, block.url, externalEmbedMode)
  } else if (block.internalSlug) {
    mainContentHtml = `
      <div
        class="arena-modal-internal-host"
        data-block-id="${escapeHtml(block.id)}"
        data-internal-slug="${escapeHtml(block.internalSlug)}"
        data-internal-href="${escapeHtml(block.internalHref || '')}"
        data-internal-hash="${escapeHtml(block.internalHash || '')}"
      >
        <div class="arena-modal-internal-preview grid"></div>
      </div>
    `
  } else {
    mainContentHtml = `<div class="arena-modal-text-preview">${block.blockHtml ?? escapeHtml(block.content)}</div>`
  }

  container.innerHTML = `
    <div class="arena-modal-layout">
      <div class="arena-modal-main">
        ${
          displayUrl
            ? `
          <div class="arena-modal-url-bar">
            <button type="button" class="arena-url-copy-button" data-url="${escapedDisplayUrl}" role="button" tabIndex="0" aria-label="Copy URL to clipboard">
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" class="copy-icon">
                <path d="M7.49996 1.80002C4.35194 1.80002 1.79996 4.352 1.79996 7.50002C1.79996 10.648 4.35194 13.2 7.49996 13.2C10.648 13.2 13.2 10.648 13.2 7.50002C13.2 4.352 10.648 1.80002 7.49996 1.80002ZM0.899963 7.50002C0.899963 3.85494 3.85488 0.900024 7.49996 0.900024C11.145 0.900024 14.1 3.85494 14.1 7.50002C14.1 11.1451 11.145 14.1 7.49996 14.1C3.85488 14.1 0.899963 11.1451 0.899963 7.50002Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M13.4999 7.89998H1.49994V7.09998H13.4999V7.89998Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.09991 13.5V1.5H7.89991V13.5H7.09991zM10.375 7.49998C10.375 5.32724 9.59364 3.17778 8.06183 1.75656L8.53793 1.24341C10.2396 2.82218 11.075 5.17273 11.075 7.49998 11.075 9.82724 10.2396 12.1778 8.53793 13.7566L8.06183 13.2434C9.59364 11.8222 10.375 9.67273 10.375 7.49998zM3.99969 7.5C3.99969 5.17611 4.80786 2.82678 6.45768 1.24719L6.94177 1.75281C5.4582 3.17323 4.69969 5.32389 4.69969 7.5 4.6997 9.67611 5.45822 11.8268 6.94179 13.2472L6.45769 13.7528C4.80788 12.1732 3.9997 9.8239 3.99969 7.5z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.49996 3.95801C9.66928 3.95801 11.8753 4.35915 13.3706 5.19448 13.5394 5.28875 13.5998 5.50197 13.5055 5.67073 13.4113 5.83948 13.198 5.89987 13.0293 5.8056 11.6794 5.05155 9.60799 4.65801 7.49996 4.65801 5.39192 4.65801 3.32052 5.05155 1.97064 5.8056 1.80188 5.89987 1.58866 5.83948 1.49439 5.67073 1.40013 5.50197 1.46051 5.28875 1.62927 5.19448 3.12466 4.35915 5.33063 3.95801 7.49996 3.95801zM7.49996 10.85C9.66928 10.85 11.8753 10.4488 13.3706 9.6135 13.5394 9.51924 13.5998 9.30601 13.5055 9.13726 13.4113 8.9685 13.198 8.90812 13.0293 9.00238 11.6794 9.75643 9.60799 10.15 7.49996 10.15 5.39192 10.15 3.32052 9.75643 1.97064 9.00239 1.80188 8.90812 1.58866 8.9685 1.49439 9.13726 1.40013 9.30601 1.46051 9.51924 1.62927 9.6135 3.12466 10.4488 5.33063 10.85 7.49996 10.85z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
              </svg>
              <svg width="15" height="15" viewBox="-2 -2 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" class="check-icon">
                <use href="#github-check"/>
              </svg>
            </button>
            ${
              block.url
                ? `
              <a href="${escapeHtml(block.url)}" target="_blank" rel="noopener noreferrer" class="arena-modal-link">
                <div class="arena-modal-link-text">${escapedDisplayUrl}</div>
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path fill-rule="evenodd" clip-rule="evenodd" d="${EXTERNAL_LINK_ICON_PATH}" fill="currentColor"/>
                </svg>
              </a>
            `
                : `
              <span class="arena-modal-link">
                <div class="arena-modal-link-text">${escapedDisplayUrl}</div>
              </span>
            `
            }
          </div>
        `
            : ''
        }
        ${mapHtml}
        <div class="arena-modal-main-content">
          ${mainContentHtml}
        </div>
      </div>
      <div class="arena-modal-sidebar">
        <div class="arena-modal-info">
          <h3 class="arena-modal-title">
            ${block.titleHtml ?? escapeHtml(block.title ?? '')}
          </h3>
          ${
            block.entryId
              ? `<a class="arena-entry-permalink internal" href="/${['arena', ...channelSlug.split('/'), block.entryId].map(encodeURIComponent).join('/')}" data-no-popover>open entry</a>`
              : ''
          }
          ${
            metadataHtml.length > 0
              ? `
            <div class="arena-modal-meta">
              ${metadataHtml.join('')}
            </div>
          `
              : ''
          }
        </div>
        ${subItemsHtml}
      </div>
    </div>
  `

  return container
}
