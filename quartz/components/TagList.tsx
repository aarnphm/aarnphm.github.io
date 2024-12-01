import { pathToRoot, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

const TagList: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const tags = fileData.frontmatter?.tags
  const baseDir = pathToRoot(fileData.slug!)
  if (tags && tags.length > 0) {
    return (
      <div class={classNames(displayClass, "tags")}>
        <h3>étiquette</h3>
        <ul>
          {tags.map((tag) => {
            const linkDest = baseDir + `/tags/${slugTag(tag)}`
            return (
              <li>
                <a href={linkDest} class="internal tag-link">
                  {tag}
                </a>
              </li>
            )
          })}
        </ul>
      </div>
    )
  } else {
    return null
  }
}

TagList.css = `
ul.tags,
.tags > ul {
  list-style: none;
  display: flex;
  padding-left: 0;
  gap: 0.4rem;
  margin: 0;
  flex-wrap: wrap;
}

ul.tags {
  margin: 1rem 0;
}

.section-li > .section > .tags {
  justify-content: flex-end;
}

ul.tags > li {
  display: inline-block;
  white-space: nowrap;
  margin: 0;
  overflow-wrap: normal;
}
`

export default (() => TagList) satisfies QuartzComponentConstructor
