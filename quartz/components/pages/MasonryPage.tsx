import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { classNames } from '../../util/lang'
// @ts-ignore
import script from '../scripts/masonry.inline'
import style from '../styles/masonry.scss'

export default (() => {
  const MasonryPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const jsonPath = fileData.masonryJsonPath
    const images = fileData.masonryImages || []

    const classes: string[] = fileData.frontmatter?.cssclasses ?? []

    if (images.length === 0) {
      return (
        <article
          class={classNames(
            displayClass,
            ...classes,
            'masonry-container',
            'all-col',
            'popover-hint',
          )}
        >
          <div class="masonry-empty all-col">no images found</div>
        </article>
      )
    }

    return (
      <article
        class={classNames(displayClass, ...classes, 'masonry-container', 'all-col', 'popover-hint')}
      >
        <button
          type="button"
          class="masonry-mode-toggle"
          id="masonry-mode-toggle"
          aria-label="Chaos mode"
          aria-pressed="false"
          aria-controls="masonry-grid"
          title="Switch to chaos mode"
          hidden
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <g class="masonry-tidy-dots">
              {[5, 12, 19].flatMap(y => [5, 12, 19].map(x => <circle cx={x} cy={y} r="1.5" />))}
            </g>
            <g class="masonry-chaos-dots">
              <circle cx="4" cy="6" r="1.5" />
              <circle cx="12" cy="3" r="2" />
              <circle cx="20" cy="7" r="1.5" />
              <circle cx="8" cy="13" r="2" />
              <circle cx="18" cy="14" r="1" />
              <circle cx="3" cy="20" r="1" />
              <circle cx="13" cy="21" r="1.5" />
              <circle cx="22" cy="21" r="1.5" />
            </g>
          </svg>
        </button>
        <div class="masonry-grid" id="masonry-grid" data-json-path={jsonPath}>
          {images.map(image => (
            <img
              class="masonry-image"
              src={image.src}
              alt={image.alt}
              width={image.width}
              height={image.height}
              style={{ aspectRatio: `${image.width} / ${image.height}` }}
              loading="lazy"
              decoding="async"
            />
          ))}
        </div>
        <div class="masonry-caption-modal" id="masonry-caption-modal"></div>
      </article>
    )
  }

  MasonryPage.css = style
  MasonryPage.afterDOMLoaded = script

  return MasonryPage
}) satisfies QuartzComponentConstructor
