import type { Root as HtmlRoot } from 'hast'
import fs from 'node:fs/promises'
import { Node } from 'unist'
import type { TriView } from '../../components/pages/triathlon-panels'
import type { GarminCache } from '../stores/garmin'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { TriathlonPage } from '../../components'
import HeaderConstructor from '../../components/Header'
import { TriathlonSubPage } from '../../components/pages/TriathlonSubPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FilePath, FullSlug, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import {
  appleCachePath,
  enrichSwimStrokes,
  garminCachePath,
  ouraCachePath,
  stravaCachePath,
  weatherCachePath,
} from '../../util/strava-payload'
import { buildFeedMarkdown } from '../../util/triathlon-feed'
import {
  ATHLETE,
  buildAnalytics,
  buildDataFeed,
  hrZoneUppers,
  parseVo2Lab,
} from '../stores/analytics'
import { AppleCache } from '../stores/apple'
import { OuraCache } from '../stores/oura'
import { buildPayload, emptyHealth, StravaPayload, StravaRawCache } from '../stores/strava'
import { parseTrainingPlans } from '../stores/training'
import { parseWeatherCache, WeatherCache } from '../stores/weather'
import { defaultProcessedContent, ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

const TRI_SUBVIEWS: TriView[] = ['tools', 'calc', 'analytics', 'maps', 'training', 'feed']

async function readCache(): Promise<StravaRawCache | null> {
  try {
    return JSON.parse(await fs.readFile(stravaCachePath, 'utf8')) as StravaRawCache
  } catch {
    return null
  }
}

async function readOura(): Promise<OuraCache | null> {
  try {
    return JSON.parse(await fs.readFile(ouraCachePath, 'utf8')) as OuraCache
  } catch {
    return null
  }
}

async function readGarmin(): Promise<GarminCache | null> {
  try {
    return JSON.parse(await fs.readFile(garminCachePath, 'utf8')) as GarminCache
  } catch {
    return null
  }
}

async function readApple(): Promise<AppleCache | null> {
  try {
    return JSON.parse(await fs.readFile(appleCachePath, 'utf8')) as AppleCache
  } catch {
    return null
  }
}

async function readWeather(): Promise<WeatherCache | null> {
  try {
    return parseWeatherCache(JSON.parse(await fs.readFile(weatherCachePath, 'utf8')))
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
    const weather = await readWeather()
    const files: FilePath[] = []

    const allFiles = contentDataFor(content)
    for (const [tree, file] of content) {
      if (!isTriathlon(file.data)) continue
      const since = file.data.frontmatter?.['strava']
      const tracking = file.data.tracking
      const vo2labs = parseVo2Lab(file.data.frontmatter?.['vo2max'])
      const latestVo2 = vo2labs.length ? vo2labs[vo2labs.length - 1] : null
      const hrBoundsOverride = latestVo2 ? hrZoneUppers(latestVo2) : null
      const payload = buildPayload(
        cache,
        oura,
        garmin,
        typeof since === 'string' ? since : undefined,
        weather,
        ATHLETE.ftp,
        hrBoundsOverride ?? undefined,
      )
      for (const t of tracking?.days ?? [])
        if (t.windKph != null) {
          const h = payload.health[t.date] ?? emptyHealth()
          payload.health[t.date] = { ...h, windKph: t.windKph, windDir: t.windDir ?? h.windDir }
        }
      enrichSwimStrokes(payload, apple)
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
            ftp: ATHLETE.ftp,
            goalFtp: ATHLETE.goalFTP,
            vt1Hr: latestVo2?.vt1Hr ?? null,
          }),
        }),
      )
      const analytics = buildAnalytics(cache, {
        oura,
        apple,
        garmin,
        weather,
        weights: tracking?.days,
        events: tracking?.races,
        dexa: file.data.frontmatter?.['dexa'],
        vo2labs: file.data.frontmatter?.['vo2max'],
        ftp: ATHLETE.ftp,
        since: typeof since === 'string' ? since : undefined,
      })
      files.push(
        await write({
          ctx,
          slug: 'static/analytics' as FullSlug,
          ext: '.json',
          content: JSON.stringify(analytics),
        }),
      )
      files.push(
        await write({
          ctx,
          slug: 'static/training' as FullSlug,
          ext: '.json',
          content: JSON.stringify({ plans: parseTrainingPlans(tree as unknown as HtmlRoot) }),
        }),
      )
      const dataFeed = buildDataFeed(cache, analytics, {
        oura,
        apple,
        weather,
        garmin,
        weights: tracking?.days,
        zones: payload.zones,
      })
      files.push(
        await write({ ctx, slug: 'triathlon/data' as FullSlug, ext: '.jsonl', content: dataFeed }),
      )
      files.push(
        await write({
          ctx,
          slug: 'triathlon/feed' as FullSlug,
          ext: '.md',
          content: buildFeedMarkdown(dataFeed, analytics, {
            details: payload.details,
            baseUrl: ctx.cfg.configuration.baseUrl,
          }),
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

      for (const view of TRI_SUBVIEWS) {
        const subSlug = `triathlon/${view}` as FullSlug
        const [subTree, subFile] = defaultProcessedContent({
          slug: subSlug,
          frontmatter: { title: `triathlon · ${view}`, pageLayout: 'default', tags: [] },
        })
        const subResources = pageResources(pathToRoot(subSlug), resources, ctx)
        const subData: QuartzComponentProps = {
          ctx,
          fileData: { ...subFile.data, stravaPayload: payload },
          externalResources: subResources,
          cfg: ctx.cfg.configuration,
          children: [],
          tree: subTree as Node,
          allFiles,
        }
        const subHtml = renderPage(
          ctx,
          subSlug,
          subData,
          { ...opts, pageBody: TriathlonSubPage(view) },
          subResources,
          true,
        )
        files.push(await write({ ctx, content: subHtml, slug: subSlug, ext: '.html' }))
      }
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
