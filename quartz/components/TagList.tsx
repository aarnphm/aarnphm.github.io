import { pathToRoot, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"
import style from "./styles/tags.scss"

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

TagList.css = style

export default (() => TagList) satisfies QuartzComponentConstructor
