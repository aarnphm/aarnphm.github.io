import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

export default (() =>
  ({ fileData, displayClass }: QuartzComponentProps) => {
    const title = fileData.frontmatter?.title
    if (title) {
      return (
        <hgroup class={classNames(displayClass, "title-col")} data-article-title>
          <h1 class="article-title">{title}</h1>
          <p class="description">{fileData.description}</p>
        </hgroup>
      )
    }

    return <></>
  }) satisfies QuartzComponentConstructor
