import type { Root as HtmlRoot } from 'hast'
import fs from 'node:fs/promises'
import { Node } from 'unist'
import type { GarminCache } from '../stores/garmin'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { TriathlonPage } from '../../components'
import HeaderConstructor from '../../components/Header'
import { CONVERSIONS, GEAR, type TriView } from '../../components/pages/triathlon-panels'
import TriathlonDayPage from '../../components/pages/TriathlonDayPage'
import { TriathlonSubPage } from '../../components/pages/TriathlonSubPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FilePath, FullSlug, joinSegments, pathToRoot, QUARTZ } from '../../util/path'
import { StaticResources } from '../../util/resources'
import {
  appleCachePath,
  coreBodyTemperatureCachePath,
  enrichCoreBodyTemperature,
  enrichRunDynamics,
  enrichSwimMetrics,
  garminCachePath,
  ouraCachePath,
  stravaCachePath,
  weatherCachePath,
} from '../../util/strava-payload'
import {
  triathlonActivityDates,
  triathlonActivityFeedRoutes,
  triathlonDaySlug,
  triathlonFeedScopeFromSlug,
} from '../../util/triathlon-date-route'
import { buildTriathlonMarkdown } from '../../util/triathlon-markdown'
import {
  ATHLETE,
  buildAnalytics,
  buildDataFeed,
  hrZoneUppers,
  parseVo2Lab,
} from '../stores/analytics'
import { AppleCache } from '../stores/apple'
import {
  parseCoreBodyTemperatureCache,
  type CoreBodyTemperatureCache,
} from '../stores/core-body-temperature'
import { buildMatchedRides, emptyMatchedRides } from '../stores/matched-rides'
import { buildMatchedRuns, emptyMatchedRuns } from '../stores/matched-runs'
import { OuraCache } from '../stores/oura'
import {
  applyManualFueling,
  applyManualStrength,
  buildPayload,
  emptyHealth,
  StravaPayload,
  StravaRawCache,
} from '../stores/strava'
import { parseTrainingPlans } from '../stores/training'
import { parseWeatherCache, WeatherCache } from '../stores/weather'
import { defaultProcessedContent, ProcessedContent, QuartzPluginData } from '../vfile'
import { removeWritten, write } from './helpers'
import { createOgImageGenerator } from './ogImage'

const TRI_SUBVIEWS: TriView[] = ['tools', 'calc', 'analytics', 'maps', 'training', 'feed', 'on']
const TRIATHLON_DATA_CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache', 'triathlon')
const TRIATHLON_DATA_CACHE_PATH = joinSegments(TRIATHLON_DATA_CACHE_DIR, 'data.jsonl')

async function cacheDataFeed(content: string): Promise<void> {
  await fs.mkdir(TRIATHLON_DATA_CACHE_DIR, { recursive: true })
  await fs.writeFile(TRIATHLON_DATA_CACHE_PATH, content)
}

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

async function readCoreBodyTemperature(): Promise<CoreBodyTemperatureCache | null> {
  try {
    return parseCoreBodyTemperatureCache(
      JSON.parse(await fs.readFile(coreBodyTemperatureCachePath, 'utf8')),
    )
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

const triathlonDescription = (
  view: TriView | 'day',
  activityCount: number,
  planCount: number,
  scopePrefix = '',
): string => {
  switch (view) {
    case 'tools':
      return 'Triathlon gear, fuel, pace conversions, and race-distance reference data.'
    case 'calc':
      return 'Triathlon finish-time calculator inputs, current calibration, thresholds, and projections.'
    case 'analytics':
      return `Generated training analytics across ${activityCount} activities.`
    case 'maps':
      return `Generated route geometry and map metrics across ${activityCount} activities.`
    case 'training':
      return `${planCount} generated triathlon training plans with their complete note content.`
    case 'feed':
      return `Generated training feed across ${activityCount} activities.`
    case 'on':
      return `Generated training log for ${scopePrefix || 'the complete activity window'}.`
    case 'day':
      return `Generated activity data and telemetry for ${scopePrefix}.`
  }
}

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
  const DayPage = TriathlonDayPage()
  let emittedTemporalSlugs = new Set<FullSlug>()

  async function emitAll(
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
  ): Promise<FilePath[]> {
    const cache = await readCache()
    const oura = await readOura()
    const garmin = await readGarmin()
    const apple = await readApple()
    const core = await readCoreBodyTemperature()
    const weather = await readWeather()
    const files: FilePath[] = []
    const nextTemporalSlugs = new Set<FullSlug>()
    const generateOgImage =
      ctx.argv.watch && !ctx.argv.force ? null : await createOgImageGenerator(ctx)
    let ogImageQueue = Promise.resolve()
    const queueOgImage = generateOgImage
      ? (data: QuartzPluginData) => {
          const output = ogImageQueue.then(() => generateOgImage(data))
          ogImageQueue = output.then(() => undefined)
          return output
        }
      : null

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
      applyManualFueling(payload, tracking?.fueling ?? [])
      applyManualStrength(payload, tracking?.strength ?? [])
      for (const t of tracking?.days ?? [])
        if (t.windKph != null) {
          const h = payload.health[t.date] ?? emptyHealth()
          payload.health[t.date] = { ...h, windKph: t.windKph, windDir: t.windDir ?? h.windDir }
        }
      enrichSwimMetrics(payload, apple)
      enrichRunDynamics(payload, apple)
      enrichCoreBodyTemperature(payload, core)
      const detailActivityIds = new Set(Object.keys(payload.details))
      const matchedActivities = cache
        ? Object.values(cache.activities).filter(activity =>
            detailActivityIds.has(String(activity.id)),
          )
        : []
      const matchedRuns = cache
        ? buildMatchedRuns(matchedActivities, cache.streams ?? {})
        : emptyMatchedRuns()
      const matchedRides = cache
        ? buildMatchedRides(matchedActivities, cache.streams ?? {})
        : emptyMatchedRides()
      files.push(
        await write({
          ctx,
          slug: 'static/strava-detail' as FullSlug,
          ext: '.json',
          content: JSON.stringify({
            details: payload.details,
            swimTrend: payload.swimTrend,
            health: payload.health,
            zones: payload.zones,
            powerCurveRef: payload.powerCurveRef,
            powerCurveYearRef: payload.powerCurveYearRef,
            powerCurveYear: payload.powerCurveYear,
            ftp: ATHLETE.ftp,
            goalFtp: ATHLETE.goalFTP,
            vt1Hr: latestVo2?.vt1Hr ?? null,
            matchedRuns,
            matchedRides,
          }),
        }),
      )
      const analytics = buildAnalytics(cache, {
        oura,
        apple,
        core,
        garmin,
        weather,
        weights: tracking?.days,
        events: tracking?.races,
        dexa: file.data.frontmatter?.['dexa'],
        vo2labs: file.data.frontmatter?.['vo2max'],
        ftp: ATHLETE.ftp,
        powerCurve: {
          sixWeeks: payload.powerCurveRef,
          year: payload.powerCurveYearRef,
          yearLabel: payload.powerCurveYear,
          ftp: ATHLETE.ftp,
          goalFtp: ATHLETE.goalFTP,
        },
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
          slug: 'static/oura-detail' as FullSlug,
          ext: '.json',
          content: JSON.stringify(oura?.details ?? {}),
        }),
      )
      const plans = parseTrainingPlans(tree as unknown as HtmlRoot)
      files.push(
        await write({
          ctx,
          slug: 'static/training' as FullSlug,
          ext: '.json',
          content: JSON.stringify({ plans }),
        }),
      )
      const dataFeed = buildDataFeed(cache, analytics, {
        oura,
        apple,
        weather,
        garmin,
        weights: tracking?.days,
        trainingExclusions: tracking?.trainingExclusions,
        zones: payload.zones,
      })
      await cacheDataFeed(dataFeed)
      files.push(
        await write({
          ctx,
          slug: 'static/triathlon/data' as FullSlug,
          ext: '.jsonl',
          content: dataFeed,
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

      const renderSubPage = async (
        subSlug: FullSlug,
        view: TriView,
        title: string,
        scopePrefix = '',
      ): Promise<FilePath[]> => {
        const description = triathlonDescription(
          view,
          payload.totalCount,
          plans.length,
          scopePrefix,
        )
        const markdown = buildTriathlonMarkdown({
          view,
          slug: subSlug,
          title,
          description,
          baseUrl: ctx.cfg.configuration.baseUrl,
          scopePrefix,
          dataFeed,
          analytics,
          payload,
          plans,
          tools: { conversions: CONVERSIONS, gear: GEAR },
        })
        const [subTree, subFile] = defaultProcessedContent({
          slug: subSlug,
          text: description,
          rawDescription: description,
          frontmatter: {
            title,
            description,
            socialDescription: description,
            generatedSocialImage: true,
            pageLayout: 'default',
            tags: [],
          },
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
          { ...opts, pageBody: TriathlonSubPage(view, file.data.frontmatter?.['triathlon']) },
          subResources,
          true,
        )
        const outputs = await Promise.all([
          write({ ctx, content: subHtml, slug: subSlug, ext: '.html' }),
          write({ ctx, content: markdown, slug: subSlug, ext: '.md' }),
          ...(queueOgImage ? [queueOgImage(subFile.data)] : []),
        ])
        return outputs
      }
      files.push(
        ...(
          await Promise.all(
            TRI_SUBVIEWS.map(view =>
              renderSubPage(`triathlon/${view}` as FullSlug, view, `triathlon · ${view}`),
            ),
          )
        ).flat(),
      )

      const feedRoutes = triathlonActivityFeedRoutes(payload.details)
      for (const { slug: feedSlug } of feedRoutes) nextTemporalSlugs.add(feedSlug)
      files.push(
        ...(
          await Promise.all(
            feedRoutes.map(({ slug: feedSlug, title }) =>
              renderSubPage(
                feedSlug,
                'on',
                title,
                triathlonFeedScopeFromSlug(feedSlug)?.prefix ?? '',
              ),
            ),
          )
        ).flat(),
      )

      const dayRoutes = triathlonActivityDates(payload.details).flatMap(date => {
        const slug = triathlonDaySlug(date)
        return slug ? [{ date, slug }] : []
      })
      for (const { slug: daySlug } of dayRoutes) nextTemporalSlugs.add(daySlug)
      const location = file.data.frontmatter?.['location']
      files.push(
        ...(
          await Promise.all(
            dayRoutes.map(async ({ date, slug: daySlug }) => {
              const title = `triathlon · ${date}`
              const description = triathlonDescription(
                'day',
                payload.totalCount,
                plans.length,
                date,
              )
              const markdown = buildTriathlonMarkdown({
                view: 'day',
                slug: daySlug,
                title,
                description,
                baseUrl: ctx.cfg.configuration.baseUrl,
                scopePrefix: date,
                dataFeed,
                analytics,
                payload,
                plans,
                tools: { conversions: CONVERSIONS, gear: GEAR },
              })
              const [dayTree, dayFile] = defaultProcessedContent({
                slug: daySlug,
                filePath: file.data.filePath,
                tracking: file.data.tracking,
                text: description,
                rawDescription: description,
                frontmatter: {
                  title,
                  description,
                  socialDescription: description,
                  generatedSocialImage: true,
                  pageLayout: 'default',
                  tags: [],
                  ...(typeof location === 'string' ? { location } : {}),
                },
              })
              const dayResources = pageResources(pathToRoot(daySlug), resources, ctx)
              const dayData: QuartzComponentProps = {
                ctx,
                fileData: { ...dayFile.data, stravaPayload: payload },
                externalResources: dayResources,
                cfg: ctx.cfg.configuration,
                children: [],
                tree: dayTree as Node,
                allFiles,
              }
              const dayHtml = renderPage(
                ctx,
                daySlug,
                dayData,
                { ...opts, pageBody: DayPage },
                dayResources,
                true,
              )
              return Promise.all([
                write({ ctx, content: dayHtml, slug: daySlug, ext: '.html' }),
                write({ ctx, content: markdown, slug: daySlug, ext: '.md' }),
                ...(queueOgImage ? [queueOgImage(dayFile.data)] : []),
              ])
            }),
          )
        ).flat(),
      )
    }
    await Promise.all(
      [...emittedTemporalSlugs]
        .filter(temporalSlug => !nextTemporalSlugs.has(temporalSlug))
        .flatMap(temporalSlug => [
          removeWritten(ctx, temporalSlug, '.html'),
          removeWritten(ctx, temporalSlug, '.md'),
          removeWritten(ctx, `${temporalSlug}-og-image`, '.webp'),
        ]),
    )
    emittedTemporalSlugs = nextTemporalSlugs
    return files
  }

  return {
    name: 'Strava',
    getQuartzComponents() {
      return [
        Head,
        Header,
        ...header,
        ...beforeBody,
        pageBody,
        DayPage,
        ...afterBody,
        ...sidebar,
        Footer,
      ]
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
