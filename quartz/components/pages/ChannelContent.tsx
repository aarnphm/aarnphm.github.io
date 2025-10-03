import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { ArenaChannel, ArenaBlock } from "../../plugins/transformers/arena"
import { classNames } from "../../util/lang"
import { htmlToJsx } from "../../util/jsx"
import { toHtml } from "hast-util-to-html"
import type { ElementContent, Root } from "hast"
import style from "../styles/arena.scss"
// @ts-ignore
import modalScript from "../scripts/arena.inline"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"

const substackPostRegex = /^https?:\/\/[^/]+\/p\/[^/]+/i

const escapeHtml = (value: string): string =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")

const nodeToHtml = (node?: ElementContent): string => {
  if (!node) {
    return ""
  }

  const root: Root = {
    type: "root",
    children: [node as ElementContent],
  }

  return toHtml(root, { allowDangerousHtml: true })
}

const buildSrcDoc = (block: ArenaBlock): string => {
  const body = block.htmlNode
    ? nodeToHtml(block.htmlNode)
    : `<p>${escapeHtml(block.title ?? block.content ?? "")}</p>`

  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><style>html,body{margin:0;padding:12px;font-family:inherit;font-size:16px;line-height:1.5;color:var(--text,#111);background:var(--light,#fff);overflow:hidden;}a{color:inherit;text-decoration:underline;}img,video{max-width:100%;height:auto;display:block;}p{margin:0 0 1em 0;}ul,ol{margin:0 0 1em 1.5em;padding:0;}code{font-family:inherit;background:rgba(0,0,0,0.05);padding:0.15em 0.3em;border-radius:4px;}</style></head><body>${body}</body></html>`
}

export default (() => {
  const ChannelContent: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const channel = fileData.arenaChannel as ArenaChannel | undefined

    if (!channel) {
      return <article class="arena-content">Channel not found</article>
    }

    const renderBlock = (block: ArenaBlock, blockIndex: number) => {
      const hasSubItems = block.subItems && block.subItems.length > 0
      const frameTitle = block.title ?? block.content ?? `Block ${blockIndex + 1}`
      const iframeSrcDoc = block.url ? undefined : buildSrcDoc(block)
      const embedHtml = block.embedHtml
      const isSubstackCandidate = block.url ? substackPostRegex.test(block.url) : false

      const convertFromText = (text: string) => {
        const tocAst = fromHtmlIsomorphic(text, { fragment: true })
        return htmlToJsx(fileData.filePath!, tocAst)
      }

      return (
        <div
          key={block.id}
          class={classNames(displayClass, "arena-block", block.highlighted ? "highlighted" : "")}
          data-block-id={block.id}
          data-block-index={blockIndex}
          data-channel-slug={channel.slug}
        >
          <div
            class="arena-block-clickable"
            role="button"
            tabIndex={0}
            aria-label="View block details"
          >
            <div class="arena-block-content">
              {block.titleHtmlNode
                ? htmlToJsx(fileData.filePath!, block.titleHtmlNode)
                : block.title || block.content}
            </div>
          </div>
          <div
            class="arena-block-modal-data"
            id={`arena-modal-data-${block.id}`}
            style="display: none;"
          >
            <div class="arena-modal-layout">
              <div class="arena-modal-main">
                {block.url && (
                  <div class="arena-modal-url-bar">
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
                  ) : isSubstackCandidate && block.url ? (
                    <div
                      class="arena-modal-embed arena-modal-embed-substack"
                      data-substack-url={block.url}
                    >
                      <p>Loading Substack preview...</p>
                    </div>
                  ) : (
                    <iframe
                      class="arena-modal-iframe"
                      title={`Embedded block: ${frameTitle}`}
                      loading="lazy"
                      data-block-id={block.id}
                      sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox"
                      {...(block.url ? { src: block.url } : { srcDoc: iframeSrcDoc })}
                    />
                  )}
                </div>
              </div>
              <div class="arena-modal-sidebar">
                <div class="arena-modal-info">
                  <h3 class="arena-modal-title">{block.title ?? ""}</h3>
                </div>
                {hasSubItems && (
                  <div class="arena-modal-connections">
                    <div class="arena-modal-connections-header">
                      <span class="arena-modal-connections-title">notes</span>
                      <span class="arena-modal-connections-count">{block.subItems!.length}</span>
                    </div>
                    <ul class="arena-modal-connections-list">
                      {block.subItems!.map((subItem) => (
                        <li key={subItem.id}>
                          {subItem.titleHtmlNode
                            ? htmlToJsx(fileData.filePath!, subItem.titleHtmlNode)
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
          {channel.blocks.map((block, idx) => renderBlock(block, idx))}
        </div>

        <div class="arena-block-modal" id="arena-modal">
          <div class="arena-modal-content">
            <div class="arena-modal-nav">
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
