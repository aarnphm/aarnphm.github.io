import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

export default (() => {
  const ArticleTitle: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const title = fileData.frontmatter?.title
    if (title) {
      return (
        <hgroup class="title-col">
          <h1 class={classNames(displayClass, "article-title")}>{title}</h1>
          <p class="description">{fileData.frontmatter?.description && fileData.description}</p>
        </hgroup>
      )
    } else {
      return <></>
    }
  }

  return ArticleTitle
}) satisfies QuartzComponentConstructor
