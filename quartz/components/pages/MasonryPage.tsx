import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../../types/component"
import style from "../styles/masonry.scss"
// @ts-ignore
import script from "../scripts/masonry.inline"
import { classNames } from "../../util/lang"

export default (() => {
  const MasonryPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    // get JSON path from emitter
    const jsonPath = fileData.masonryJsonPath
    const images = fileData.masonryImages || []

    const classes: string[] = fileData.frontmatter?.cssclasses ?? []

    if (images.length === 0) {
      return (
        <article
          class={classNames(
            displayClass,
            ...classes,
            "masonry-container",
            "all-col",
            "grid",
            "popover-hint",
          )}
        >
          <div class="masonry-empty all-col">no images found</div>
        </article>
      )
    }

    return (
      <article
        class={classNames(
          displayClass,
          ...classes,
          "masonry-container",
          "all-col",
          "popover-hint",
          "grid",
        )}
      >
        <div class="masonry-grid all-col" id="masonry-grid" data-json-path={jsonPath}></div>
        <div class="masonry-caption-modal" id="masonry-caption-modal"></div>
      </article>
    )
  }

  MasonryPage.css = style
  MasonryPage.afterDOMLoaded = script

  return MasonryPage
}) satisfies QuartzComponentConstructor
