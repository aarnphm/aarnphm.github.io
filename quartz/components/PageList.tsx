import { GlobalConfiguration } from '../cfg'
import { QuartzPluginData } from '../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentProps,
  QuartzComponentConstructor,
} from '../types/component'
import { resolveRelative } from '../util/path'
import { Date as DateComponent, getDate } from './Date'

export type SortFn = (f1: QuartzPluginData, f2: QuartzPluginData) => number
type LocaleConfig = Pick<GlobalConfiguration, 'locale'>

const collators = new Map<string, Intl.Collator>()

function collatorFor(cfg: LocaleConfig): Intl.Collator {
  let collator = collators.get(cfg.locale)
  if (!collator) {
    collator = new Intl.Collator(cfg.locale, { numeric: true, sensitivity: 'base' })
    collators.set(cfg.locale, collator)
  }
  return collator
}

function titleFor(file: QuartzPluginData): string {
  return file.frontmatter?.title ?? ''
}

function naturalSlugFor(file: QuartzPluginData): string {
  return (file.slug ?? '').replace(/\/index$/, '').replace(/[-_]+/g, ' ')
}

export function byTitleAlphabetical(cfg: LocaleConfig): SortFn {
  const collator = collatorFor(cfg)
  return (f1, f2) => collator.compare(titleFor(f1), titleFor(f2))
}

export function byNaturalSlug(cfg: LocaleConfig): SortFn {
  const collator = collatorFor(cfg)
  return (f1, f2) => {
    const slugComparison = collator.compare(naturalSlugFor(f1), naturalSlugFor(f2))
    if (slugComparison !== 0) return slugComparison
    return collator.compare(titleFor(f1), titleFor(f2))
  }
}

export function byDateAndAlphabetical(cfg: GlobalConfiguration): SortFn {
  return (f1, f2) => {
    // Sort by date/alphabetical
    if (f1.dates && f2.dates) {
      // sort descending
      return getDate(cfg, f2)!.getTime() - getDate(cfg, f1)!.getTime()
    } else if (f1.dates && !f2.dates) {
      // prioritize files with dates
      return -1
    } else if (!f1.dates && f2.dates) {
      return 1
    }

    // otherwise, sort lexicographically by title
    const f1Title = f1.frontmatter?.title.toLowerCase() ?? ''
    const f2Title = f2.frontmatter?.title.toLowerCase() ?? ''
    return f1Title.localeCompare(f2Title)
  }
}

type Props = { limit?: number; sort?: SortFn } & QuartzComponentProps

interface Options {
  highlightTags: string[]
}

const defaultOptions: Options = { highlightTags: [] }

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }

  const PageList: QuartzComponent = ({ cfg, fileData, allFiles, limit, sort }: Props) => {
    const sorter = sort ?? byDateAndAlphabetical(cfg)
    let list = allFiles.sort(sorter)
    if (limit) {
      list = list.slice(0, limit)
    }

    return (
      <ul class="section-ul">
        {list.map((page, idx) => {
          const title = page.frontmatter?.title
          const tags = page.frontmatter?.tags ?? []
          const hiTags = opts.highlightTags.filter(v => tags.includes(v))
          const date = new Date(0)

          return (
            <li
              class="section-li"
              data-index={idx}
              data-title={title}
              data-tags={JSON.stringify(tags)}
            >
              <a
                class="note-link"
                href={resolveRelative(fileData.slug!, page.slug!)}
                data-list={true}
                data-tags={tags.join(',')}
              >
                <div class="note-grid">
                  {page.dates ? (
                    <div class="meta">
                      <DateComponent date={getDate(cfg, page)!} locale={cfg.locale} />
                    </div>
                  ) : (
                    <div class="meta">
                      <DateComponent date={date} locale={cfg.locale} />
                    </div>
                  )}
                  <div class="desc">
                    {title}
                    {tags.includes('folder') && <span>/</span>}
                  </div>
                  {hiTags.length > 0 ? (
                    <menu class="tag-highlights">
                      {hiTags.map(el => (
                        <li class="tag" data-tag={el}>
                          {el}
                        </li>
                      ))}
                    </menu>
                  ) : (
                    <></>
                  )}
                </div>
              </a>
            </li>
          )
        })}
      </ul>
    )
  }

  return PageList
}) satisfies QuartzComponentConstructor
