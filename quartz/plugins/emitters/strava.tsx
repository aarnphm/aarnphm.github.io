import fs from 'node:fs/promises'
import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { TriathlonPage } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FilePath, FullSlug, joinSegments, pathToRoot, QUARTZ } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { buildPayload, StravaPayload, StravaRawCache } from '../stores/strava'
import { buildAnalytics } from '../stores/strava-analytics'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')

async function readCache(): Promise<StravaRawCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as StravaRawCache
  } catch {
    return null
  }
}

const isTriathlon = (data: QuartzPluginData): boolean => data.frontmatter?.layout === 'triathlon'

export const Strava: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    header: [],
    beforeBody: [],
    afterBody: [],
    sidebar: [],
    pageBody: TriathlonPage(),
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  async function emitAll(
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
  ): Promise<FilePath[]> {
    const cache = await readCache()
    const files: FilePath[] = []

    const allFiles = contentDataFor(content)
    for (const [tree, file] of content) {
      if (!isTriathlon(file.data)) continue
      const since = file.data.frontmatter?.['strava']
      const payload = buildPayload(cache, typeof since === 'string' ? since : undefined)
      files.push(
        await write({
          ctx,
          slug: 'static/strava-detail' as FullSlug,
          ext: '.json',
          content: JSON.stringify(payload.details),
        }),
      )
      files.push(
        await write({
          ctx,
          slug: 'static/strava-analytics' as FullSlug,
          ext: '.json',
          content: JSON.stringify(buildAnalytics(cache)),
        }),
      )
      const slug = file.data.slug!
      const externalResources = pageResources(pathToRoot(slug), resources, ctx)
      const componentData: QuartzComponentProps = {
        ctx,
        fileData: { ...file.data, stravaPayload: payload },
        externalResources,
        cfg: ctx.cfg.configuration,
        children: [],
        tree: tree as Node,
        allFiles,
      }
      const html = renderPage(ctx, slug, componentData, opts, externalResources, false)
      files.push(await write({ ctx, content: html, slug, ext: '.html' }))
    }
    return files
  }

  return {
    name: 'Strava',
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      yield* await emitAll(ctx, content, resources)
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const touched = changeEvents.some(event => event.file && isTriathlon(event.file.data))
      if (!touched) return
      yield* await emitAll(ctx, content, resources)
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    stravaPayload: StravaPayload
  }
}
