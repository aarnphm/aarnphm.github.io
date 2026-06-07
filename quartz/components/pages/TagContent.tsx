import { Root } from 'hast'
import { i18n } from '../../i18n'
import { QuartzPluginData } from '../../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { inheritComponentSourceNames } from '../../util/component-source'
import { htmlToJsx } from '../../util/jsx'
import { FullSlug, getAllSegmentPrefixes, simplifySlug } from '../../util/path'
import { concatenateResources } from '../../util/resources'
import PageListConstructor, { SortFn } from '../PageList'
import PageListSearchConstructor from '../PageListSearch'
import SeeAlsoComponent from '../SeeAlso'
import style from '../styles/listPage.scss'

interface TagContentOptions {
  sort?: SortFn
  numPages: number
}

const defaultOptions: TagContentOptions = { numPages: 10 }

function collectTagIndex(allFiles: QuartzPluginData[]): {
  tagItemMap: Map<string, QuartzPluginData[]>
  tagContentMap: Map<string, QuartzPluginData>
} {
  const tagItemMap = new Map<string, QuartzPluginData[]>()
  const tagContentMap = new Map<string, QuartzPluginData>()
  for (const file of allFiles) {
    if (file.slug?.startsWith('tags/')) {
      tagContentMap.set(file.slug.slice('tags/'.length), file)
    }

    for (const tag of (file.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes)) {
      const pages = tagItemMap.get(tag)
      if (pages) {
        pages.push(file)
      } else {
        tagItemMap.set(tag, [file])
      }
    }
  }
  return { tagItemMap, tagContentMap }
}

function tagIndexData(
  value: unknown,
):
  | { tagItemMap: Map<string, QuartzPluginData[]>; tagContentMap: Map<string, QuartzPluginData> }
  | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  if (!('tagItemMap' in value) || !('tagContentMap' in value)) return undefined
  if (!(value.tagItemMap instanceof Map) || !(value.tagContentMap instanceof Map)) return undefined
  return { tagItemMap: value.tagItemMap, tagContentMap: value.tagContentMap }
}

function pagesWithTag(allFiles: QuartzPluginData[], tag: string): QuartzPluginData[] {
  const pages: QuartzPluginData[] = []
  for (const file of allFiles) {
    for (const candidate of file.frontmatter?.tags ?? []) {
      if (getAllSegmentPrefixes(candidate).includes(tag)) {
        pages.push(file)
        break
      }
    }
  }
  return pages
}

function explicitTagPageFiles(value: unknown): QuartzPluginData[] | undefined {
  return Array.isArray(value)
    ? value.filter((file): file is QuartzPluginData => Boolean(file))
    : undefined
}

export default ((opts?: Partial<TagContentOptions>) => {
  const options: TagContentOptions = { ...defaultOptions, ...opts }

  const PageList = PageListConstructor()
  const PageListSearch = PageListSearchConstructor()
  const SeeAlso = SeeAlsoComponent()

  const TagContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles, cfg } = props
    const slug = fileData.slug

    if (!(slug?.startsWith('tags/') || slug === 'tags')) {
      throw new Error(`Component "TagContent" tried to render a non-tag page: ${slug}`)
    }

    const tag = simplifySlug(slug.slice('tags/'.length) as FullSlug)

    const content =
      (tree as Root).children.length === 0
        ? fileData.description
        : htmlToJsx(fileData.filePath!, tree)
    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const classes = ['popover-hint', 'full-col', ...cssClasses].join(' ')
    if (tag === '/') {
      const providedTagIndexData = tagIndexData(props.tagIndexData)
      const { tagItemMap, tagContentMap } = providedTagIndexData ?? collectTagIndex(allFiles)
      const tags = [...tagItemMap.keys()].sort((a, b) => a.localeCompare(b))
      return (
        <div class={classes}>
          <article>
            <p>{content}</p>
          </article>
          <SeeAlso {...props} />
          <p>{i18n(cfg.locale).pages.tagContent.totalTags({ count: tags.length })}</p>
          <PageListSearch {...props} allTags />
          <div>
            {tags.map(tag => {
              const pages = tagItemMap.get(tag)!
              const listProps = { ...props, allFiles: pages }

              const contentPage = tagContentMap.get(tag)

              const root = contentPage?.htmlAst
              const content =
                !root || root?.children.length === 0
                  ? contentPage?.description
                  : htmlToJsx(contentPage.filePath!, root)

              return (
                <div>
                  <h2>
                    <a class="internal tag-link" data-no-popover href={`../tags/${tag}`}>
                      {tag}
                    </a>
                  </h2>
                  {content && <p>{content}</p>}
                  <div class="page-listing">
                    <p>
                      {i18n(cfg.locale).pages.tagContent.itemsUnderTag({ count: pages.length })}
                      {pages.length > options.numPages && (
                        <>
                          {' '}
                          <span>
                            {i18n(cfg.locale).pages.tagContent.showingFirst({
                              count: options.numPages,
                            })}
                          </span>
                        </>
                      )}
                    </p>
                    <PageList
                      limit={options.numPages}
                      {...listProps}
                      sort={opts?.sort}
                      presorted={Boolean(providedTagIndexData)}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    } else {
      const pages = explicitTagPageFiles(props.tagPageFiles) ?? pagesWithTag(allFiles, tag)
      const listProps = { ...props, allFiles: pages }

      return (
        <div class={classes} data-pagelist>
          <article>{content}</article>
          <SeeAlso {...props} />
          <div class="page-listing">
            <p>{i18n(cfg.locale).pages.tagContent.itemsUnderTag({ count: pages.length })}</p>
            <PageListSearch {...props} />
            <div>
              <PageList {...listProps} sort={opts?.sort} presorted={Boolean(props.tagPageFiles)} />
            </div>
          </div>
        </div>
      )
    }
  }

  TagContent.css = concatenateResources(style, PageListSearch.css, SeeAlso.css)
  TagContent.sourceNames = inheritComponentSourceNames('TagContent', [
    PageList,
    PageListSearch,
    SeeAlso,
  ])
  TagContent.afterDOMLoaded = PageListSearch.afterDOMLoaded
  return TagContent
}) satisfies QuartzComponentConstructor
