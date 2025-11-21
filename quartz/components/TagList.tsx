import { FullSlug, pathToRoot, resolveRelative, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"
import { stripWikilinkFormatting } from "../util/wikilinks"
import type { FrontmatterLink } from "../plugins/transformers/frontmatter"
import style from "./styles/tags.scss"

export default (() => {
  const TagList: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    const tags = fileData.frontmatter?.tags
    const linkLookup = fileData.frontmatterLinkLookup as Record<string, FrontmatterLink> | undefined
    const currentSlug = fileData.slug as FullSlug | undefined
    const baseDir = pathToRoot(fileData.slug!)

    const buildHref = (link: FrontmatterLink): string => {
      if (currentSlug) {
        return `${resolveRelative(currentSlug, link.slug)}${link.anchor ?? ""}`
      }

      return `/${link.slug}${link.anchor ?? ""}`
    }
    if (tags && tags.length > 0) {
      return (
        <menu class={classNames(displayClass, "tags")}>
          <li>
            <h2>{i18n(cfg.locale).pages.tagContent.tag}</h2>
            <ul>
              {tags.map((tag) => {
                const wikiLink = linkLookup?.[tag]

                if (wikiLink) {
                  const href = buildHref(wikiLink)
                  const label = wikiLink.alias ?? stripWikilinkFormatting(tag)

                  return (
                    <li>
                      <a href={href} class="internal tag-link" data-slug={wikiLink.slug}>
                        {label}
                      </a>
                    </li>
                  )
                }

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
                  const linkValue = typeof link === "string" ? link : (link?.toString?.() ?? "")
                  const wikiLink = linkLookup?.[linkValue]
                  const isInternal = Boolean(wikiLink) || linkValue.startsWith("/")
                  const href = wikiLink ? buildHref(wikiLink) : linkValue
                  return (
                    <li>
                      <address>
                        <a
                          href={href}
                          target={!isInternal ? "_blank" : ""}
                          rel={!isInternal ? "noopener noreferrer" : ""}
                          class={isInternal ? "internal" : "external"}
                          data-slug={wikiLink?.slug}
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
    }
    return <></>
  }

  TagList.css = style
  return TagList
}) satisfies QuartzComponentConstructor
