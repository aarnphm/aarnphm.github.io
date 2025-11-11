import type { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/masonry.scss"
// @ts-ignore
import script from "../scripts/masonry.inline"

export default (() => {
  const MasonryPage: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData } = props

    // get images from the data passed by emitter
    const images = (fileData as any).masonryImages || []

    if (images.length === 0) {
      return (
        <article class="masonry-container all-col">
          <div class="masonry-empty">no images found</div>
        </article>
      )
    }

    return (
      <article class="masonry-container all-col">
        <div class="masonry-grid" id="masonry-grid">
          {images.map((img: { src: string; alt: string }, idx: number) => (
            <img
              data-src={img.src}
              data-caption={img.alt}
              data-index={idx}
              class="masonry-image"
              loading="lazy"
              alt={img.alt}
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
