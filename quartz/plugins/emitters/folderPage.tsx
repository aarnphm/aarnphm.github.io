import path from 'path'
import { defaultListPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { FolderContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { i18n, TRANSLATIONS } from '../../i18n'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { pageListingChanged } from '../../util/listing-signature'
import {
  FilePath,
  FullSlug,
  SimpleSlug,
  stripSlashes,
  joinSegments,
  pathToRoot,
  simplifySlug,
  slugifyFilePath,
} from '../../util/path'
import { StaticResources } from '../../util/resources'
import { ProcessedContent, QuartzPluginData, defaultProcessedContent } from '../vfile'
import { write } from './helpers'
interface FolderPageOptions extends FullPageLayout {
  sort?: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

const folderPageSourceExtensions = new Set(['.md', '.base', '.canvas', '.ipynb'])

function isFolderPageSourcePath(fp: FilePath): boolean {
  return folderPageSourceExtensions.has(path.extname(fp))
}

function sourcePathSlug(fp: FilePath): FullSlug {
  return slugifyFilePath(fp, path.extname(fp) === '.ipynb')
}

async function* processFolderInfo(
  ctx: BuildCtx,
  folderInfo: Record<SimpleSlug, ProcessedContent>,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const entries = Object.entries(folderInfo) as [SimpleSlug, ProcessedContent][]
  const files = await mapConcurrent(entries, defaultIoConcurrency, ([folder, folderContent]) =>
    processFolderPage(ctx, folder, folderContent, allFiles, opts, resources),
  )
  yield* files
}

async function processFolderPage(
  ctx: BuildCtx,
  folder: SimpleSlug,
  folderContent: ProcessedContent,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath> {
  const slug = joinSegments(folder, 'index') as FullSlug
  const [tree, file] = folderContent
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: file.data,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(ctx, slug, componentData, opts, externalResources, true)
  return write({ ctx, content, slug, ext: '.html' })
}

function computeFolderInfo(
  folders: Set<SimpleSlug>,
  content: ProcessedContent[],
  locale: keyof typeof TRANSLATIONS,
): Record<SimpleSlug, ProcessedContent> {
  const folderInfo: Record<SimpleSlug, ProcessedContent> = Object.fromEntries(
    [...folders].map(folder => [
      folder,
      defaultProcessedContent({
        slug: joinSegments(folder, 'index') as FullSlug,
        frontmatter: {
          title: `${i18n(locale).pages.folderContent.folder}: ${folder}`,
          pageLayout: 'default',
          tags: [],
        },
      }),
    ]),
  )

  for (const [tree, file] of content) {
    const slug = stripSlashes(simplifySlug(file.data.slug!)) as SimpleSlug
    if (folders.has(slug)) {
      folderInfo[slug] = [tree, file]
    }
  }

  return folderInfo
}

function _getFolders(slug: FullSlug): SimpleSlug[] {
  var folderName = path.dirname(slug ?? '') as SimpleSlug
  const parentFolderNames = [folderName]

  while (folderName !== '.') {
    folderName = path.dirname(folderName ?? '') as SimpleSlug
    parentFolderNames.push(folderName)
  }
  return parentFolderNames
}

export const FolderPage: QuartzEmitterPlugin<Partial<FolderPageOptions>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: FolderContent({ sort: userOpts?.sort }),
    header: [...defaultListPageLayout.beforeBody],
    beforeBody: [],
    sidebar: [],
    afterBody: [],
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: 'FolderPage',
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const mdFiles = contentDataFor(content)
      const cfg = ctx.cfg.configuration

      const folders: Set<SimpleSlug> = new Set(
        ctx.allFiles
          .filter(isFolderPageSourcePath)
          .map(sourcePathSlug)
          .flatMap(slug =>
            _getFolders(slug).filter(folderName => folderName !== '.' && folderName !== 'tags'),
          ),
      )

      const folderInfo = computeFolderInfo(folders, content, cfg.locale)
      yield* processFolderInfo(ctx, folderInfo, mdFiles, opts, resources)
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)
      const cfg = ctx.cfg.configuration

      const affectedFolders: Set<SimpleSlug> = new Set()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (
          changeEvent.type === 'change' &&
          !pageListingChanged(changeEvent.file.data, changeEvent.previousFile?.data)
        ) {
          continue
        }
        const slug = changeEvent.file.data.slug!
        const folders = _getFolders(slug).filter(
          folderName => folderName !== '.' && folderName !== 'tags',
        )
        folders.forEach(folder => affectedFolders.add(folder))
      }

      if (affectedFolders.size > 0) {
        const folderInfo = computeFolderInfo(affectedFolders, content, cfg.locale)
        yield* processFolderInfo(ctx, folderInfo, allFiles, opts, resources)
      }
    },
  }
}
