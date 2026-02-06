import type { FrontmatterLink } from '../plugins/transformers/frontmatter'
import { i18n } from '../i18n'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
import { FullSlug, pathToRoot, resolveRelative, slugTag } from '../util/path'
import { stripWikilinkFormatting } from '../util/wikilinks'
import style from './styles/tags.scss'

export default (() => {
  const TagList: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    const tags = fileData.frontmatter?.tags
    const linkLookup = fileData.frontmatterLinkLookup as Record<string, FrontmatterLink> | undefined
    const currentSlug = fileData.slug as FullSlug | undefined
    const baseDir = pathToRoot(fileData.slug!)

    const buildHref = (link: FrontmatterLink): string => {
      if (currentSlug) {
        return `${resolveRelative(currentSlug, link.slug)}${link.anchor ?? ''}`
      }

      return `/${link.slug}${link.anchor ?? ''}`
    }

    const isRecord = (value: unknown): value is Record<string, unknown> =>
      typeof value === 'object' && value !== null

    const normalizeSocialLink = (value: unknown): string => {
      if (typeof value === 'string') return value
      if (typeof value === 'number' || typeof value === 'boolean') return String(value)
      if (value && typeof value === 'object' && 'toString' in value) {
        const candidate = value.toString
        if (typeof candidate === 'function') {
          return candidate.call(value)
        }
      }
      return ''
    }

    const socialsEntries = isRecord(fileData.frontmatter?.socials)
      ? Object.entries(fileData.frontmatter.socials)
      : []
    if (tags && tags.length > 0) {
      return (
        <menu class={classNames(displayClass, 'tags')}>
          <li>
            <h2>{i18n(cfg.locale).pages.tagContent.tag}</h2>
            <ul>
              {tags.map(tag => {
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
          {socialsEntries.length > 0 && (
            <li class="socials">
              <h2>média</h2>
              <ul>
                {socialsEntries.map(([social, link]) => {
                  const linkValue = normalizeSocialLink(link)
                  const wikiLink = linkLookup?.[linkValue]
                  const isInternal = Boolean(wikiLink) || linkValue.startsWith('/')
                  const href = wikiLink ? buildHref(wikiLink) : linkValue
                  return (
                    <li>
                      <address>
                        <a
                          href={href}
                          target={!isInternal ? '_blank' : ''}
                          rel={!isInternal ? 'noopener noreferrer' : ''}
                          class={isInternal ? 'internal' : 'external'}
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
