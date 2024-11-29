import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

const ArticleTitle: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
  const title = fileData.frontmatter?.title
  if (title) {
    return (
      <hgroup>
        <h1 class={classNames(displayClass, "article-title")}>{title}</h1>
        <p class="description">
          {fileData.description ?? i18n(cfg.locale).propertyDefaults.description}
        </p>
      </hgroup>
    )
  } else {
    return <></>
  }
}

ArticleTitle.css = `
.article-title {
  margin: 2rem 0 0 0;
}
`

export default (() => ArticleTitle) satisfies QuartzComponentConstructor
