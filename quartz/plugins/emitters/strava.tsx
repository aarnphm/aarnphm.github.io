import fs from 'node:fs/promises'
import { Node } from 'unist'
import type { GarminCache } from '../stores/garmin'
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
import { buildAnalytics } from '../stores/analytics'
import { AppleCache } from '../stores/apple'
import { OuraCache } from '../stores/oura'
import { buildPayload, StravaPayload, StravaRawCache } from '../stores/strava'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const ouraCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'oura.json')
const garminCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
const appleCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')

async function readCache(): Promise<StravaRawCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as StravaRawCache
  } catch {
    return null
  }
}

async function readOura(): Promise<OuraCache | null> {
  try {
    return JSON.parse(await fs.readFile(ouraCacheFile, 'utf8')) as OuraCache
  } catch {
    return null
  }
}

async function readGarmin(): Promise<GarminCache | null> {
  try {
    return JSON.parse(await fs.readFile(garminCacheFile, 'utf8')) as GarminCache
  } catch {
    return null
  }
}

async function readApple(): Promise<AppleCache | null> {
  try {
    return JSON.parse(await fs.readFile(appleCacheFile, 'utf8')) as AppleCache
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
    const oura = await readOura()
    const garmin = await readGarmin()
    const apple = await readApple()
    const files: FilePath[] = []

    const allFiles = contentDataFor(content)
    for (const [tree, file] of content) {
      if (!isTriathlon(file.data)) continue
      const since = file.data.frontmatter?.['strava']
      const payload = buildPayload(
        cache,
        oura,
        garmin,
        typeof since === 'string' ? since : undefined,
      )
      files.push(
        await write({
          ctx,
          slug: 'static/strava-detail' as FullSlug,
          ext: '.json',
          content: JSON.stringify({
            details: payload.details,
            health: payload.health,
            zones: payload.zones,
            powerCurveRef: payload.powerCurveRef,
          }),
        }),
      )
      const tracking = file.data.tracking
      files.push(
        await write({
          ctx,
          slug: 'static/analytics' as FullSlug,
          ext: '.json',
          content: JSON.stringify(
            buildAnalytics(cache, {
              oura,
              apple,
              weights: tracking?.days,
              events: tracking?.races,
              since: typeof since === 'string' ? since : undefined,
            }),
          ),
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
