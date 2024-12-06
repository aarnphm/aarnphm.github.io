import { pathToRoot, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

const TagList: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
  const tags = fileData.frontmatter?.tags
  const baseDir = pathToRoot(fileData.slug!)
  if (tags && tags.length > 0) {
    return (
      <menu class={classNames(displayClass, "tags")}>
        <li>
          <h2>{i18n(cfg.locale).pages.tagContent.tag}</h2>
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
        </li>
        {fileData.frontmatter?.socials && (
          <li class="socials">
            <h2>média</h2>
            <ul>
              {Object.entries(fileData.frontmatter?.socials).map(([social, link]) => {
                return (
                  <li>
                    <address>
                      <a href={link} target="_blank" rel="noopener noreferrer" class="external">
                        {social}
                      </a>
                    </address>
                  </li>
                )
              })}
            </ul>
          </li>
        )}
      </menu>
    )
  } else {
    return null
  }
}

TagList.css = `
ul.tags,
menu.tags,
menu.tags > li > ul {
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

menu.tags {
  grid-column: 5/5;
}

li.socials > ul {
  flex-direction: column;
  gap: 0.8rem;
}

li.socials > ul > li > address {
  font-style: normal
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
