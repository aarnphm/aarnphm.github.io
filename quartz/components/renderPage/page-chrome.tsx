import { Element, ElementContent, Root } from 'hast'
import { JSX } from 'preact'
import { visit } from 'unist-util-visit'
import { i18n } from '../../i18n'
import { collectHtmlTocData } from '../../plugins/transformers/toc'
import { QuartzPluginData } from '../../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { classNames } from '../../util/lang'
import { resolveRelative, SimpleSlug } from '../../util/path'
import { Date as DateComponent, getDate } from '../Date'
import FooterConstructor from '../Footer'
import HeaderConstructor from '../Header'
import HeadingsConstructor from '../Headings'
import { byDateAndAlphabetical } from '../PageList'
import ContentConstructor from '../pages/Content'
import PressKitConstructor from '../PressKit'

export interface RenderComponents {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  afterBody: QuartzComponent[]
  sidebar: QuartzComponent[]
  footer: QuartzComponent
}

export const Header = HeaderConstructor()
export const Headings = HeadingsConstructor()
export const PressKit = PressKitConstructor()
export const PoetryFooter = FooterConstructor({ layout: 'poetry' })
export const MenuFooter = FooterConstructor({ layout: 'menu' })

export function updateTocDataFromTree(root: Root, componentData: QuartzComponentProps): void {
  const opts = componentData.fileData.tocOptions
  if (!opts) return

  if (!opts.display) {
    delete componentData.fileData.toc
    return
  }

  const toc = collectHtmlTocData(root, opts)
  componentData.fileData.tocOptions = { ...opts, sourceEntries: toc.sourceEntries }
  if (toc.entries.length > 0) {
    componentData.fileData.toc = toc.entries
  } else {
    delete componentData.fileData.toc
  }
}

type AliasLinkProp = {
  name?: string
  url?: string
  isInternal?: boolean
  newTab?: boolean | ((name: string) => boolean)
  enablePopover?: boolean
  classes?: string[]
  children?: JSX.Element | JSX.Element[]
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, enablePopover: true, ...props }
  const className = ['landing-links']
  if (opts.isInternal) className.push('internal')
  if (opts.classes) className.push(...opts.classes)
  return (
    <a
      href={opts.url}
      target={opts.newTab ? '_blank' : '_self'}
      rel="noopener noreferrer"
      className={className.join(' ')}
      data-no-popover={!opts.enablePopover}
      data-skip-icons
    >
      {opts.name}
      {opts.children}
    </a>
  )
}

const NotesComponent = ((opts?: { slug: SimpleSlug; numLimits?: number; header?: string }) => {
  const Notes: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { allFiles, fileData, cfg } = componentData
    const noteSlug = opts?.slug
    const pages = allFiles
      .filter((f: QuartzPluginData) => {
        const slug = f.slug
        if (noteSlug && slug?.startsWith(noteSlug)) {
          return (
            !['university', 'tags', 'library', 'index', ...cfg.ignorePatterns].some(it =>
              slug.includes(it),
            ) && !f.frontmatter?.noindex
          )
        }
        return false
      })
      .sort((a: QuartzPluginData, b: QuartzPluginData): number => {
        const afm = a.frontmatter!
        const bfm = b.frontmatter!
        if (afm.priority && bfm.priority) {
          return afm.priority - bfm.priority
        } else if (afm.priority && !bfm.priority) {
          return -1
        } else if (!afm.priority && bfm.priority) {
          return 1
        }
        return byDateAndAlphabetical(cfg)(a, b)
      })

    const remaining = Math.max(0, pages.length - opts!.numLimits!)
    const classes = ['min-links', 'internal'].join(' ')
    return (
      <section id={`note-item-${opts!.header}`} data-note style={{ marginTop: '1.2em' }}>
        <em>{opts!.header}</em>
        <div class="notes-container">
          <div class="recent-links">
            <ul class="landing-notes">
              {pages.slice(0, opts!.numLimits).map(page => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
                return (
                  <li>
                    <a
                      data-no-popover
                      href={resolveRelative(fileData.slug!, page.slug!)}
                      class={classes}
                    >
                      <div class="landing-meta">
                        <span class="landing-mspan">
                          <DateComponent date={getDate(cfg, page)!} locale={cfg.locale} />
                        </span>
                        <span class="landing-mtitle">{title}</span>
                      </div>
                    </a>
                  </li>
                )
              })}
            </ul>
            {remaining > 0 && (
              <p style={{ marginTop: '0' }}>
                <a
                  data-no-popover
                  href={resolveRelative(fileData.slug!, opts!.slug)}
                  class={classNames(undefined, classes, 'see-more')}
                  style={{ fontSize: '0.9em', textDecoration: 'underline' }}
                >
                  {i18n(cfg.locale).components.recentNotes.seeRemainingMore({ remaining })}
                </a>
              </p>
            )}
          </div>
        </div>
      </section>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const HyperlinksComponent = ((props?: { children: JSX.Element[] }) => {
  const { children } = props ?? { children: [] }

  const Hyperlink: QuartzComponent = () => <section class="hyperlinks">{children}</section>
  return Hyperlink
}) satisfies QuartzComponentConstructor

export const ElementComponent = ((enableRecents: boolean = false) => {
  const Content = ContentConstructor()
  const RecentNotes = NotesComponent({
    header: 'récentes',
    slug: 'thoughts/' as SimpleSlug,
    numLimits: 9,
  })
  const RecentPosts = NotesComponent({
    header: 'écriture',
    slug: 'posts/' as SimpleSlug,
    numLimits: 6,
  })

  const Element: QuartzComponent = (componentData: QuartzComponentProps) => {
    const Hyperlink = HyperlinksComponent({
      children: [
        <section class="homepage-board-garden">
          <div class="clickable-container">
            <AliasLink isInternal enablePopover={false} name="stream" url="/stream" />
            <AliasLink isInternal enablePopover={false} name="are.na" url="/arena" />
            <AliasLink isInternal enablePopover={false} name="notes" url="/thoughts/" />
            <AliasLink isInternal enablePopover={false} name="writings" url="/posts/" />
          </div>
        </section>,
        <section class="boring-legal">
          <div class="clickable-container">
            <em class="homepage-board-license">
              <AliasLink
                newTab
                classes={['external']}
                name="CC BY-NC-SA"
                url="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en"
              />
            </em>
            <AliasLink isInternal enablePopover={false} name="colophon" url="/colophon" />
            <AliasLink isInternal enablePopover={false} name="friends" url="/friends" />
          </div>
        </section>,
      ],
    })

    return (
      <div class="content-container homepage-board">
        <span class="homepage-board-hangers" aria-hidden="true" />
        <nav class="homepage-board-links" aria-label="garden and site information">
          <Hyperlink {...componentData} />
        </nav>
        <div class="homepage-board-copy">
          <Content {...componentData} />
          {enableRecents && (
            <section class="notes-outer">
              <RecentNotes {...componentData} />
              <RecentPosts {...componentData} />
            </section>
          )}
        </div>
      </div>
    )
  }

  return Element
}) satisfies QuartzComponentConstructor

// Menu components

export function Functions({ displayClass }: QuartzComponentProps) {
  return (
    <section class={classNames(displayClass, 'menu', 'side-col')} data-function={true}>
      <a href="../atelier-with-friends" class="internal alias" data-no-popover={true}>
        atelier with friends.
      </a>
    </section>
  )
}

export type RenderPageOptions = {
  forEmail?: boolean
  skipProtected?: boolean
  skipSearch?: boolean
}

export function updateStreamDataFromTree(tree: Root, componentData: QuartzComponentProps): void {
  const fileData = componentData.fileData
  const isStreamRoute =
    fileData.slug === 'stream' || /^stream\/on\/\d{4}\/\d{2}\/\d{2}$/.test(fileData.slug ?? '')
  if (!isStreamRoute) return

  const streamData = fileData.streamData
  if (!streamData) return

  type StreamMarker = { node: ElementContent; index: number }
  const nodeBuckets = new Map<string, StreamMarker[]>()

  visit(tree, 'element', (node: Element) => {
    const data = node.data as Record<string, unknown> | undefined
    if (!data) return

    const entryId = data.streamEntryId
    if (typeof entryId !== 'string') return

    const rawIndex = data.streamEntryContentIndex
    const index = typeof rawIndex === 'number' ? rawIndex : Number.POSITIVE_INFINITY

    const bucket = nodeBuckets.get(entryId)
    if (bucket) {
      bucket.push({ node, index })
    } else {
      nodeBuckets.set(entryId, [{ node, index }])
    }
  })

  const entries = streamData.entries.map(entry => {
    const bucket = nodeBuckets.get(entry.id)
    if (!bucket || bucket.length === 0) return entry

    bucket.sort((a, b) => a.index - b.index)
    return { ...entry, content: bucket.map(({ node }) => node) }
  })
  fileData.streamData = { ...streamData, entries }
}
