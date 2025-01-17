import { pathToRoot, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"
import style from "./styles/tags.scss"

export default (() => {
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
                  const isInternal = link.startsWith("/")
                  return (
                    <li>
                      <address>
                        <a
                          href={link}
                          target={!isInternal ? "_blank" : ""}
                          rel={!isInternal ? "noopener noreferrer" : ""}
                          class={isInternal ? "internal" : "external"}
                          data-no-popover
                        >
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
  return TagList
}) satisfies QuartzComponentConstructor
