import type { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/masonry.scss"
// @ts-ignore
import script from "../scripts/masonry.inline"

export default (() => {
  const MasonryPage: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    // get JSON path from emitter
    const jsonPath = fileData.masonryJsonPath
    const images = fileData.masonryImages || []

    if (images.length === 0) {
      return (
        <article class="masonry-container all-col">
          <div class="masonry-empty">no images found</div>
        </article>
      )
    }

    return (
      <article class="masonry-container all-col">
        <div class="masonry-grid" id="masonry-grid" data-json-path={jsonPath}></div>
        <div class="masonry-caption-modal" id="masonry-caption-modal"></div>
      </article>
    )
  }

  MasonryPage.css = style
  MasonryPage.afterDOMLoaded = script

  return MasonryPage
}) satisfies QuartzComponentConstructor
