import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'path'
import type { StaticResources } from '../../util/resources'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { onModuleRelease } from '../../util/module-caches'
import { FilePath, FullSlug, joinSegments } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import {
  buildFolderFeedMap,
  feedCacheContainsAnySlug,
  feedContainsAnySlug,
  hasFeedEntries,
  pageFolderFeedSlugs,
  removeFolderAtomFeed,
  writeAtomFeedStylesheet,
  writeFolderAtomFeed,
  writePatchedRootAtomFeed,
  writeRootAtomFeed,
  type AtomFeedCache,
} from './contentIndex/atom'
import { robotsTxt, writeSecurityFiles, writeSiteMap } from './contentIndex/discovery'
import {
  addGraphFilePages,
  buildLinkIndex,
  buildSearchIndex,
  type ContentIndexMap,
  type Options,
} from './contentIndex/link-index'
import {
  buildSearchIndexPatch,
  planPartialContentIndex,
  titlePatchContentIndexUpdates,
} from './contentIndex/partial'
import {
  writeContentIndexJson,
  writePatchedSearchIndexChunks,
  writePatchedSearchIndexJson,
  writePatchedSerializedIndexJson,
  writeSearchIndexJson,
  type SerializedIndexCache,
} from './contentIndex/search-index'
import { write } from './helpers'
export type { ContentDetails, ContentLayout } from './contentIndex/link-index'
export { generateSiteMap, robotsTxt } from './contentIndex/discovery'

const defaultOptions: Options = {
  enableSiteMap: true,
  enableAtom: true,
  atomLimit: 10,
  includeEmptyFiles: true,
  enableSecurity: true,
}

export const ContentIndex: QuartzEmitterPlugin<Partial<Options>> = userOpts => {
  const opts: Options = { ...defaultOptions, ...userOpts }
  let cachedLinkIndex: ContentIndexMap | undefined
  let cachedContentIndexJson: SerializedIndexCache | undefined
  let cachedRootAtomFeed: AtomFeedCache | undefined
  let cachedSearchIndexJson: SerializedIndexCache | undefined
  onModuleRelease(() => {
    cachedLinkIndex = undefined
    cachedContentIndexJson = undefined
    cachedRootAtomFeed = undefined
    cachedSearchIndexJson = undefined
  })

  return {
    name: 'ContentIndex',
    async *emit(ctx, content, _resources) {
      const cfg = ctx.cfg.configuration
      const writesAtom = opts.enableAtom && !ctx.argv.watch
      const linkIndex = buildLinkIndex(ctx, content, opts)
      cachedLinkIndex = new Map(linkIndex)

      yield write({
        ctx,
        content: robotsTxt(cfg.baseUrl ?? 'example.com'),
        slug: 'robots' as FullSlug,
        ext: '.txt',
      })

      if (opts.enableSecurity) {
        yield* writeSecurityFiles(ctx, cfg, linkIndex)
      }

      if (opts.enableSiteMap) {
        yield writeSiteMap(ctx, cfg, linkIndex)
      }

      if (writesAtom) {
        yield writeAtomFeedStylesheet(ctx)
        const rootFeedWrite = await writeRootAtomFeed(ctx, cfg, linkIndex, opts)
        cachedRootAtomFeed = rootFeedWrite.cache
        yield rootFeedWrite.file
        const folderFeedMap = buildFolderFeedMap(linkIndex)
        const sortedFolderFeeds = Array.from(folderFeedMap.entries()).sort(([a], [b]) =>
          a.localeCompare(b),
        )

        const folderFeedWrites = await mapConcurrent(
          sortedFolderFeeds,
          defaultIoConcurrency,
          async ([folderSlug, folderIndex]) => {
            const file = await writeFolderAtomFeed(
              ctx,
              cfg,
              linkIndex,
              folderSlug,
              folderIndex,
              opts,
            )
            return file
          },
        )

        for (const file of folderFeedWrites) {
          if (file) yield file
        }
      }

      addGraphFilePages(ctx, linkIndex)
      const contentIndexWrite = await writeContentIndexJson(ctx, linkIndex)
      cachedContentIndexJson = contentIndexWrite.cache
      yield contentIndexWrite.file
      const searchIndexWrite = await writeSearchIndexJson(ctx, buildSearchIndex(ctx, linkIndex))
      cachedSearchIndexJson = searchIndexWrite.cache
      yield* searchIndexWrite.files

      if (
        process.env.NODE_ENV === 'development' ||
        process.env.NODE_ENV === 'test' ||
        ctx.argv.watch
      ) {
        const slug = joinSegments('.well-known', 'appspecific', 'com.chrome.devtools') as FullSlug
        const root = path.resolve(path.dirname(ctx.argv.directory)) as FilePath
        const dir = path.dirname(joinSegments(ctx.argv.output, slug)) as FilePath
        await fs.mkdir(dir, { recursive: true })
        yield write({
          ctx,
          content: JSON.stringify({
            workspace: {
              root,
              uuid: crypto
                .createHash('sha256')
                .update(root)
                .digest('hex')
                .replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, '$1-$2-$3-$4-$5')
                .substring(0, 36),
            },
          }),
          slug,
          ext: '.json',
        })
      }
    },
    async *partialEmit(ctx, content, _resources, changeEvents) {
      const writesAtom = opts.enableAtom && !ctx.argv.watch
      const plan = planPartialContentIndex(changeEvents)
      if (!plan.content && !plan.graph && !plan.search && !plan.security) {
        return
      }

      if (
        plan.content &&
        plan.search &&
        !plan.graph &&
        !plan.security &&
        !plan.sitemap &&
        plan.affectedFolders.size === 0 &&
        cachedLinkIndex &&
        cachedContentIndexJson &&
        cachedSearchIndexJson
      ) {
        const titleUpdate = titlePatchContentIndexUpdates(cachedLinkIndex, changeEvents)
        if (titleUpdate) {
          const perf = new PerfTimer()
          const cfg = ctx.cfg.configuration
          const files: FilePath[] = []
          cachedLinkIndex = new Map(titleUpdate.linkIndex)
          if (writesAtom) {
            const writesRootFeed =
              feedContainsAnySlug(
                titleUpdate.linkIndex,
                titleUpdate.changedSlugs,
                opts.atomLimit,
              ) || feedCacheContainsAnySlug(cachedRootAtomFeed, titleUpdate.changedSlugs)
            if (writesRootFeed) {
              let rootFeedWrite = cachedRootAtomFeed
                ? await writePatchedRootAtomFeed(
                    ctx,
                    cfg,
                    titleUpdate.linkIndex,
                    cachedRootAtomFeed,
                    titleUpdate.changedSlugs,
                  )
                : undefined
              if (!rootFeedWrite) {
                rootFeedWrite = await writeRootAtomFeed(
                  ctx,
                  cfg,
                  titleUpdate.linkIndex,
                  opts,
                  cachedRootAtomFeed,
                  titleUpdate.changedSlugs,
                  true,
                )
              }
              cachedRootAtomFeed = rootFeedWrite.cache
              files.push(rootFeedWrite.file)
            }
          }
          let contentIndexWrite = await writePatchedSerializedIndexJson(
            ctx,
            joinSegments('static', 'contentIndex') as FullSlug,
            cachedContentIndexJson,
            titleUpdate.contentUpdates,
          )
          if (!contentIndexWrite) {
            contentIndexWrite = await writeContentIndexJson(
              ctx,
              titleUpdate.linkIndex,
              cachedContentIndexJson,
              titleUpdate.changedSlugs,
              true,
            )
          }
          cachedContentIndexJson = contentIndexWrite.cache
          files.push(contentIndexWrite.file)
          let searchIndexWrite = await writePatchedSearchIndexChunks(
            ctx,
            cachedSearchIndexJson,
            titleUpdate.searchUpdates,
          )
          if (!searchIndexWrite) {
            searchIndexWrite = await writePatchedSearchIndexJson(
              ctx,
              cachedSearchIndexJson,
              titleUpdate.searchUpdates,
              new Set(),
              true,
            )
          }
          cachedSearchIndexJson = searchIndexWrite.cache
          files.push(...searchIndexWrite.files)
          logBuildSpan(
            ctx.argv,
            'contentIndex:titlePatch',
            `${files.length} files`,
            perf.elapsedMs(),
          )
          yield* files
          return
        }
      }

      if (!plan.content && !plan.graph && plan.search && cachedSearchIndexJson) {
        const patch = buildSearchIndexPatch(ctx, content, opts, changeEvents)
        if (patch) {
          const searchIndexWrite = await writePatchedSearchIndexJson(
            ctx,
            cachedSearchIndexJson,
            patch.updates,
            patch.deletes,
          )
          cachedSearchIndexJson = searchIndexWrite.cache
          cachedLinkIndex = undefined
          yield* searchIndexWrite.files
          return
        }
      }

      const cfg = ctx.cfg.configuration
      const linkIndex = cachedLinkIndex
        ? buildLinkIndex(ctx, content, opts, { previous: cachedLinkIndex, changeEvents })
        : buildLinkIndex(ctx, content, opts)
      cachedLinkIndex = new Map(linkIndex)

      if (opts.enableSecurity && plan.security) {
        yield* writeSecurityFiles(ctx, cfg, linkIndex)
      }

      if (plan.content) {
        if (opts.enableSiteMap && plan.sitemap) {
          yield writeSiteMap(ctx, cfg, linkIndex)
        }

        const writesRootFeed =
          writesAtom &&
          (feedContainsAnySlug(linkIndex, plan.changedSlugs, opts.atomLimit) ||
            feedCacheContainsAnySlug(cachedRootAtomFeed, plan.changedSlugs))

        if (writesAtom && writesRootFeed) {
          const rootFeedWrite = await writeRootAtomFeed(
            ctx,
            cfg,
            linkIndex,
            opts,
            cachedRootAtomFeed,
            plan.changedSlugs,
          )
          cachedRootAtomFeed = rootFeedWrite.cache
          yield rootFeedWrite.file
          if (plan.affectedFolders.size > 0) {
            const folderFeedMap = buildFolderFeedMap(linkIndex)

            const affectedFeeds = Array.from(plan.affectedFolders).sort((a, b) =>
              a.localeCompare(b),
            )
            const folderWrites = await mapConcurrent(
              affectedFeeds,
              defaultIoConcurrency,
              async folderSlug => {
                const folderIndex = folderFeedMap.get(folderSlug)
                if (!folderIndex) {
                  await removeFolderAtomFeed(ctx, folderSlug)
                  return undefined
                }

                const file = await writeFolderAtomFeed(
                  ctx,
                  cfg,
                  linkIndex,
                  folderSlug,
                  folderIndex,
                  opts,
                )
                if (file) return file
                await removeFolderAtomFeed(ctx, folderSlug)
                return undefined
              },
            )
            for (const file of folderWrites) {
              if (file) yield file
            }
          }
        }

        if (writesAtom && !writesRootFeed) {
          const folderFeedMap = buildFolderFeedMap(linkIndex)
          const affectedFeeds = Array.from(plan.affectedFolders).sort((a, b) => a.localeCompare(b))
          const folderWrites = await mapConcurrent(
            affectedFeeds,
            defaultIoConcurrency,
            async folderSlug => {
              const folderIndex = folderFeedMap.get(folderSlug)
              if (
                !folderIndex ||
                !feedContainsAnySlug(folderIndex, plan.changedSlugs, opts.atomLimit)
              ) {
                return undefined
              }

              const file = await writeFolderAtomFeed(
                ctx,
                cfg,
                linkIndex,
                folderSlug,
                folderIndex,
                opts,
              )
              if (file) return file
              await removeFolderAtomFeed(ctx, folderSlug)
              return undefined
            },
          )
          for (const file of folderWrites) {
            if (file) yield file
          }
        }
      }

      const writesContentIndex = plan.content || plan.graph
      let searchSource = linkIndex
      if (writesContentIndex) {
        searchSource = new Map(linkIndex)
        addGraphFilePages(ctx, searchSource)
        const contentIndexWrite = await writeContentIndexJson(
          ctx,
          searchSource,
          cachedContentIndexJson,
          plan.changedSlugs,
        )
        cachedContentIndexJson = contentIndexWrite.cache
        yield contentIndexWrite.file
      }

      if (plan.search || plan.graph) {
        let patchedSearchIndex = false
        if (!plan.graph && cachedSearchIndexJson) {
          const patch = buildSearchIndexPatch(ctx, content, opts, changeEvents)
          if (patch) {
            const searchIndexWrite = await writePatchedSearchIndexJson(
              ctx,
              cachedSearchIndexJson,
              patch.updates,
              patch.deletes,
            )
            cachedSearchIndexJson = searchIndexWrite.cache
            yield* searchIndexWrite.files
            patchedSearchIndex = true
          }
        }

        if (!patchedSearchIndex) {
          const incrementalSearchSlugs =
            cachedSearchIndexJson && !plan.content && !plan.graph ? plan.changedSlugs : undefined
          const searchIndexWrite = await writeSearchIndexJson(
            ctx,
            buildSearchIndex(ctx, searchSource),
            cachedSearchIndexJson,
            incrementalSearchSlugs,
          )
          cachedSearchIndexJson = searchIndexWrite.cache
          yield* searchIndexWrite.files
        }
      }
    },
    externalResources: (ctx, content = []) => {
      const { argv, cfg } = ctx
      const additionalHead: StaticResources['additionalHead'] = [
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"
          media="print"
          // @ts-ignore
          onload="this.media='all'"
        />,
      ]

      if (opts.enableAtom && !argv.watch) {
        // Capture feed eligibility before concurrent emitters render page heads.
        const folderFeedSlugs = new Set(
          Array.from(
            buildFolderFeedMap(buildLinkIndex(ctx, content, { ...opts, enableAtom: false })),
          )
            .filter(([, folderIndex]) => hasFeedEntries(folderIndex))
            .map(([folderSlug]) => folderSlug),
        )
        additionalHead.push(
          <link
            rel="alternate"
            type="application/atom+xml"
            title="atom feed"
            href={`https://${cfg.configuration.baseUrl}/index.xml`}
          />,
        )
        additionalHead.push(fileData => (
          <>
            {pageFolderFeedSlugs(fileData, folderFeedSlugs).map(folderSlug => (
              <link
                key={`atom-${folderSlug}`}
                rel="alternate"
                type="application/atom+xml"
                title={`atom feed for ${folderSlug}`}
                href={`https://${cfg.configuration.baseUrl}/${folderSlug}/index.xml`}
              />
            ))}
          </>
        ))
      }

      return { additionalHead }
    },
  }
}
