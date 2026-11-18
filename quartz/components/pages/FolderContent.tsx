import path from 'path'
import type { ReferenceCatalog } from '../../util/references/catalog'
import { GlobalConfiguration } from '../../cfg'
import { i18n } from '../../i18n'
import { QuartzPluginData } from '../../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { inheritComponentSourceNames } from '../../util/component-source'
import { courseDirOf } from '../../util/course'
import { htmlToJsx } from '../../util/jsx'
import { stripSlashes, simplifySlug, joinSegments, FullSlug, SimpleSlug } from '../../util/path'
import { referenceCatalogFor, referenceSourcePath } from '../../util/references/context'
import { parseLiteralReference, resolveLiteralReference } from '../../util/references/literal'
import { concatenateResources } from '../../util/resources'
import { universityPlaceOf } from '../../util/university'
import { CourseCalendar, courseCalendarScript } from '../CourseCalendar'
import CourseHomeConstructor from '../CourseHome'
import CourseResourceConstructor, { CoursePageHead, coursePageOf } from '../CourseResource'
import CoursesIndexConstructor from '../CoursesIndex'
import CourseStatementsConstructor from '../CourseStatements'
import CourseStudyLogConstructor from '../CourseStudy'
import EvergreenConstructor, { AllTags, EvergreenPermanentNotes } from '../Evergreen'
import FrHomeConstructor from '../FrHome'
import PageListConstructor, {
  byDateAndAlphabetical,
  byNaturalSlug,
  byTitleAlphabetical,
  SortFn,
} from '../PageList'
import PageListSearchConstructor from '../PageListSearch'
import SeeAlsoComponent from '../SeeAlso'
import style from '../styles/listPage.scss'
import UniversityCourseConstructor from '../UniversityCourse'
import UniversityIndexConstructor from '../UniversityIndex'
import { buildFolderContentIndex } from './folder-content'

interface FolderContentOptions {
  /**
   * Sort function for the pages
   */
  sort?: SortFn
  /**
   * File extensions to include (e.g., [".md", ".pdf", ".ipynb"])
   * If not provided, defaults to showing all files
   */
  include?: (string | RegExp)[]
  /**
   * File extensions to exclude
   * If not provided, no extensions are excluded
   */
  exclude?: (string | RegExp)[]
  lg: string[]
  sm: string[]
  tags: string[]
}

function extensionFilterFn(opts: FolderContentOptions): (filePath: string) => boolean {
  const matchesPattern = (filePath: string, pattern: string | RegExp): boolean => {
    if (pattern instanceof RegExp) {
      return pattern.test(filePath)
    }
    // For string patterns, treat them as exact matches (could be file extensions or exact names)
    if (pattern.startsWith('.')) {
      // If it starts with a dot, treat as extension
      return filePath.toLowerCase().endsWith(pattern.toLowerCase())
    }
    return filePath === pattern
  }

  return (filePath: string): boolean => {
    if (!opts.include && !opts.exclude) return true
    if (opts.exclude?.some(pattern => matchesPattern(filePath, pattern))) return false
    return opts.include?.some(pattern => matchesPattern(filePath, pattern)) ?? true
  }
}

const defaultOptions: FolderContentOptions = {
  include: undefined,
  exclude: undefined,
  lg: [],
  sm: [],
  tags: [],
}

const Layout = {
  defn: 'L->EAT',
  etas: 'L->ET|A',
  alsp: 'A|L',
  lovp: 'L',
  course: 'COURSE',
  courses: 'COURSES',
  fr: 'FR',
  university: 'UNIVERSITY',
  resource: 'RESOURCE',
  studyLog: 'STUDY-LOG',
  statements: 'STATEMENTS',
  latent: 'LATENT',
} as const

type FolderLayout = (typeof Layout)[keyof typeof Layout]
type FolderSortMode = 'date' | 'natural' | 'title'

const folderSortAliases: Record<string, FolderSortMode> = {
  alphabetical: 'title',
  alphabetic: 'title',
  title: 'title',
  date: 'date',
  chronological: 'date',
  created: 'date',
  modified: 'date',
  published: 'date',
  natural: 'natural',
  grammar: 'natural',
  grammatical: 'natural',
  lexical: 'natural',
  lexicographic: 'natural',
  path: 'natural',
  file: 'natural',
  filename: 'natural',
}

const parseFolderLayout = (input: string): FolderLayout => {
  if (Array.isArray(input)) return parseFolderLayout(input[0])

  const normalized = input.trim().toUpperCase() as FolderLayout
  const valid = Object.values(Layout)
  return valid.includes(normalized) ? normalized : Layout.defn
}

const parseFolderSortMode = (input: unknown): FolderSortMode | undefined => {
  if (Array.isArray(input)) return parseFolderSortMode(input[0])
  if (typeof input !== 'string') return undefined
  return folderSortAliases[input.trim().toLowerCase()]
}

const sortForFolderMode = (mode: FolderSortMode, cfg: GlobalConfiguration): SortFn => {
  switch (mode) {
    case 'date':
      return byDateAndAlphabetical(cfg)
    case 'natural':
      return byNaturalSlug(cfg)
    case 'title':
      return byTitleAlphabetical(cfg)
  }
}

function normalizePath(
  pathEntry: string,
  currentSlug: FullSlug,
  catalog: ReferenceCatalog,
  origin: string,
): SimpleSlug[] {
  const raw = pathEntry.trim()
  const link = parseLiteralReference(raw) ?? parseLiteralReference(`[[${raw}]]`)
  if (!link) return []
  const resolved = resolveLiteralReference(link, currentSlug, { catalog, origin })
  return resolved ? [simplifySlug(resolved.slug)] : []
}

export default ((opts?: Partial<FolderContentOptions>) => {
  const options: FolderContentOptions = { ...defaultOptions, ...opts }

  let folderContentIndex: ReturnType<typeof buildFolderContentIndex> | undefined

  const shouldIncludeFile = extensionFilterFn(options)

  const { tags } = options

  const PageList = PageListConstructor({ highlightTags: tags })
  const Evergreen = EvergreenConstructor({ lg: [], sm: [], tags })
  const PageListSearch = PageListSearchConstructor()
  const SeeAlso = SeeAlsoComponent()
  const CourseHome = CourseHomeConstructor()
  const CoursesIndex = CoursesIndexConstructor()
  const FrHome = FrHomeConstructor()
  const UniversityIndex = UniversityIndexConstructor()
  const UniversityCourse = UniversityCourseConstructor()
  const CourseResource = CourseResourceConstructor()
  const CourseStudyLog = CourseStudyLogConstructor()
  const CourseStatements = CourseStatementsConstructor()

  const FolderContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles, ctx, cfg } = props
    if (!fileData.slug) return null
    const catalog = referenceCatalogFor(ctx)
    const origin = referenceSourcePath(fileData)
    const currentSlug = fileData.slug
    const curated = {
      lg: options.lg.flatMap(entry => normalizePath(entry, currentSlug, catalog, origin)),
      sm: options.sm.flatMap(entry => normalizePath(entry, currentSlug, catalog, origin)),
      tags,
    }
    const CuratedEvergreen = EvergreenConstructor(curated)
    const PermanentNotes = EvergreenPermanentNotes(curated)
    if (
      !folderContentIndex ||
      folderContentIndex.allFiles !== allFiles ||
      folderContentIndex.allPaths !== ctx.allFiles
    ) {
      folderContentIndex = buildFolderContentIndex(ctx, allFiles, cfg)
    }
    const {
      aliases,
      descendantsByFolder,
      firstDescendantWithBasename,
      firstDescendantByFolder,
      fullTrie,
      immediateMarkdownByFolder,
      mdBySlug,
      navigableFolders,
      pdfTitles,
    } = folderContentIndex

    const folderSlug = stripSlashes(simplifySlug(fileData.slug!))
    const entries: QuartzPluginData[] = []
    const processed = new Set<string>()

    const folderNode = fullTrie.findNode(folderSlug.split(path.posix.sep))
    const isImagesPath = (slug: string) => slug.split('/').includes('images')

    // Compute a sensible date for the current folder (used as fallback for children)
    const folderIndexMd = mdBySlug.get(folderSlug)
    const filesUnderCurrent = descendantsByFolder.get(folderSlug) ?? []
    const defaultDate = { created: new Date(0), modified: new Date(0), published: new Date(0) }
    const currentFolderDates =
      filesUnderCurrent.length > 0
        ? firstDescendantByFolder.get(folderSlug)?.dates
        : (folderIndexMd?.dates ?? fileData?.dates)

    const pushFileEntry = (fileSlug: string, filePathStr: string) => {
      if (processed.has(fileSlug)) return
      const ext = path.extname(filePathStr)
      const baseFileName = path.basename(filePathStr, ext)
      if (!shouldIncludeFile(filePathStr)) return
      if (isImagesPath(fileSlug)) return

      // If this slug corresponds to a markdown page we know about, just use it directly
      const md = mdBySlug.get(fileSlug)
      if (md) {
        if (md.frontmatter?.noindex === true) return
        // Augment missing dates so PageList can render consistently
        const folderFallback = currentFolderDates || fileData.dates
        const augmentedDates = {
          created: md.dates?.created ?? folderFallback?.created ?? defaultDate.created,
          modified: md.dates?.modified ?? folderFallback?.modified ?? defaultDate.modified,
          published: md.dates?.published ?? folderFallback?.published ?? defaultDate.published,
        }
        entries.push({ ...md, dates: augmentedDates })
        processed.add(fileSlug)
        return
      }

      // Pull dates (prefer markdown companion if present), else fallback to current folder date
      const associatedFirst = firstDescendantWithBasename(folderSlug, baseFileName)

      const dates =
        ext.toLowerCase() === '.pdf'
          ? undefined
          : associatedFirst?.dates || currentFolderDates || fileData.dates || defaultDate

      entries.push({
        slug: (fileSlug as FullSlug) ?? (joinSegments(folderSlug, baseFileName) as FullSlug),
        frontmatter: {
          title: pdfTitles.get(fileSlug) ?? baseFileName,
          tags: [ext.replace('.', '') || 'file'],
          pageLayout: 'default',
        },
        dates,
      })
      processed.add(fileSlug)
    }

    if (folderNode) {
      const subfolders = folderNode.children.filter(child => child.isFolder)
      for (const sub of subfolders) {
        const subfolderSlugWithIndex = stripSlashes(sub.slug)
        const subfolderSimple = stripSlashes(simplifySlug(sub.slug))
        if (isImagesPath(subfolderSimple) || isImagesPath(subfolderSlugWithIndex)) continue
        if (!navigableFolders.has(subfolderSimple)) continue

        const folderIndex = mdBySlug.get(subfolderSimple)
        if (folderIndex?.frontmatter?.noindex === true) continue

        const filesInSubfolder = descendantsByFolder.get(subfolderSimple) ?? []
        const subfolderDates =
          filesInSubfolder.length > 0
            ? firstDescendantByFolder.get(subfolderSimple)?.dates
            : (folderIndex?.dates ?? fileData?.dates)

        entries.push({
          ...folderIndex,
          slug: sub.slug,
          frontmatter: {
            ...folderIndex?.frontmatter,
            title: folderIndex?.frontmatter?.title ?? sub.displayName,
            tags: [...new Set(['folder', ...(folderIndex?.frontmatter?.tags ?? [])])],
            pageLayout: folderIndex?.frontmatter?.pageLayout ?? 'default',
          },
          dates: folderIndex?.dates ?? subfolderDates,
        })
        processed.add(subfolderSlugWithIndex)
        processed.add(subfolderSimple)
      }

      for (const child of folderNode.children) {
        if (child.isFolder || !child.data) continue
        const fileSlug = stripSlashes(child.slug)
        if (aliases.has(stripSlashes(simplifySlug(child.slug)))) continue
        pushFileEntry(fileSlug, child.data.filePath)
      }
    }

    // Fallback: ensure immediate markdown children are present
    for (const file of immediateMarkdownByFolder.get(folderSlug) ?? []) {
      const fileSlug = stripSlashes(simplifySlug(file.slug!))
      if (!processed.has(fileSlug)) {
        if (file.frontmatter?.noindex === true) continue
        const folderFallback = currentFolderDates || fileData.dates
        const augmentedDates = {
          created: file.dates?.created ?? folderFallback?.created ?? defaultDate.created,
          modified: file.dates?.modified ?? folderFallback?.modified ?? defaultDate.modified,
          published: file.dates?.published ?? folderFallback?.published ?? defaultDate.published,
        }
        entries.push({ ...file, dates: augmentedDates })
        processed.add(fileSlug)
      }
    }

    const layout = parseFolderLayout(
      fileData.frontmatter ? fileData.frontmatter.pageLayout! : 'L->ET|A',
    )

    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const courseDir = courseDirOf(fileData.slug ?? '')
    const isCourseHome =
      courseDir !== null &&
      fileData.slug !== undefined &&
      stripSlashes(simplifySlug(fileData.slug)) === `courses/${courseDir}`
    const baseClassList = ['popover-hint', 'notes-list', 'side-col', ...cssClasses]
    const baseListClass = baseClassList.join(' ')
    const listClassName =
      layout === Layout.etas ? `${baseListClass} folder-layout--list` : baseListClass
    const content = htmlToJsx(fileData.filePath!, tree)

    const sortMode = parseFolderSortMode(
      fileData.frontmatter?.folderSort ??
        fileData.frontmatter?.sort ??
        fileData.frontmatter?.sortBy,
    )
    const sort = sortMode ? sortForFolderMode(sortMode, cfg) : options.sort
    const listProps = { ...props, sort, content, allFiles: entries, vaults: allFiles }

    // a university course home or one of its subfolders, over the folder's own files
    if (universityPlaceOf(fileData.slug ?? '')?.isFolder) return <UniversityCourse {...props} />

    // vendored OCW pages (syllabus, calendar, …) are articles under a course, never listings
    if (coursePageOf(fileData.slug)) {
      return (
        <article class="folder-layout--article course-page popover-hint">
          <CoursePageHead {...props} />
          <CourseCalendar props={props}>{content}</CourseCalendar>
        </article>
      )
    }

    switch (layout) {
      // `layout: latent`: the embeddings explorer (quartz/plugins/transformers/latent.ts) is the
      // listing; it takes the page's width, so no evergreen column and no file list
      case Layout.latent:
        return (
          <div class="folder-layout folder-layout--latent" data-pagelist>
            <article class="folder-layout--article">{content}</article>
          </div>
        )

      case Layout.etas:
        return (
          <div class="folder-layout folder-layout--et-a" data-pagelist>
            <section class={listClassName}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
            <div class="notes-evergreen folder-layout--evergreen">
              <PermanentNotes {...listProps} />
              <AllTags {...listProps} opts />
            </div>
            <article class="folder-layout--article">
              {content}
              <SeeAlso {...props} />
              <p class="listing-count">
                {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
                  count: listProps.allFiles.length,
                })}
              </p>
            </article>
          </div>
        )

      case Layout.alsp:
        return (
          <div
            class={`folder-layout folder-layout--a-l${isCourseHome ? ' course-home--folder' : ''}`}
            data-pagelist
          >
            <article class="folder-layout--article">
              {content}
              <SeeAlso {...props} />
              <p class="listing-count">
                {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
                  count: listProps.allFiles.length,
                })}
              </p>
            </article>
            <section class={listClassName}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
          </div>
        )

      // the course home renders its metadata from frontmatter; the vendored body is not shown
      case Layout.course:
        return <CourseHome {...props} />

      case Layout.courses:
        return <CoursesIndex {...props} />

      // `fr/index.md`: the French folder as one course, seasons and calendar
      case Layout.fr:
        return <FrHome {...props} />

      // `thoughts/university/index.md`: archived courses by term, no study state
      case Layout.university:
        return <UniversityIndex {...props} />

      // vendored OCW resource pages pick their view from `resource.kind`
      case Layout.resource:
        return <CourseResource {...props} />

      // `lectures/index.md`: the authored study log of a course
      case Layout.studyLog:
        return <CourseStudyLog {...props} />

      // `statements/index.md`: theorems, lemmas and definitions of a course with their cards
      case Layout.statements:
        return <CourseStatements {...props} />

      case Layout.lovp: {
        const hasBodyContent = Boolean(
          tree &&
          'children' in tree &&
          Array.isArray((tree as { children?: unknown[] }).children) &&
          (tree as { children: Array<{ type?: string; value?: string }> }).children.some(
            child =>
              child.type === 'element' ||
              (child.type === 'text' &&
                typeof child.value === 'string' &&
                child.value.trim().length > 0),
          ),
        )
        const hasSeeAlso = Boolean(
          fileData.frontmatter?.seealso &&
          Array.isArray(fileData.frontmatter.seealso) &&
          fileData.frontmatter.seealso.length > 0,
        )
        const showArticle = hasBodyContent || hasSeeAlso
        const showList = listProps.allFiles.length > 0

        if (!showArticle && !showList) {
          return (
            <>
              <section class={baseListClass}>
                <PageListSearch {...props} />
                <PageList {...listProps} />
              </section>
              <aside class="notes-evergreen">
                <CuratedEvergreen {...listProps} />
              </aside>
            </>
          )
        }

        return (
          <div class="folder-layout folder-layout--l" data-pagelist>
            {showArticle && (
              <article class="folder-layout--article">
                {hasBodyContent ? content : null}
                <SeeAlso {...props} />
              </article>
            )}
            {showList && (
              <section class={listClassName}>
                <PageListSearch {...props} />
                <PageList {...listProps} />
              </section>
            )}
          </div>
        )
      }

      default:
        return (
          <>
            <section class={baseListClass}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
            <aside class="notes-evergreen">
              <CuratedEvergreen {...listProps} />
            </aside>
          </>
        )
    }
  }

  FolderContent.css = concatenateResources(
    style,
    Evergreen.css,
    PageListSearch.css,
    SeeAlso.css,
    CourseHome.css,
    FrHome.css,
    CourseResource.css,
    CourseStudyLog.css,
    CourseStatements.css,
    UniversityIndex.css,
    UniversityCourse.css,
  )
  FolderContent.sourceNames = inheritComponentSourceNames('FolderContent', [
    Evergreen,
    PageList,
    PageListSearch,
    SeeAlso,
    CourseHome,
    CoursesIndex,
    FrHome,
    UniversityIndex,
    UniversityCourse,
    CourseResource,
    CourseStudyLog,
    CourseStatements,
  ])
  FolderContent.afterDOMLoaded = concatenateResources(
    Evergreen.afterDOMLoaded,
    PageListSearch.afterDOMLoaded,
    CourseHome.afterDOMLoaded,
    FrHome.afterDOMLoaded,
    CourseResource.afterDOMLoaded,
    courseCalendarScript,
  )

  return FolderContent
}) satisfies QuartzComponentConstructor
