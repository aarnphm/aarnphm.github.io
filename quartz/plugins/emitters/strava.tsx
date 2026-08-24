import type { Root as HtmlRoot } from 'hast'
import fs from 'node:fs/promises'
import { Node } from 'unist'
import type { TriathlonRenderData } from '../../components/triathlon/render-data'
import type { GarminCache } from '../stores/garmin'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { Cursor, TriathlonPage } from '../../components'
import HeaderConstructor from '../../components/Header'
import { CONVERSIONS, GEAR, type TriView } from '../../components/pages/triathlon-panels'
import TriathlonDayPage from '../../components/pages/TriathlonDayPage'
import { TriathlonSubPage } from '../../components/pages/TriathlonSubPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import {
  defaultEmitterConcurrency,
  defaultIoConcurrency,
  mapConcurrent,
} from '../../util/async-pool'
import { hashContent } from '../../util/content-hash'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FilePath, FullSlug, joinSegments, pathToRoot, QUARTZ } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { serializeStravaDetails, type StravaDetailPayload } from '../../util/strava-detail'
import {
  appleCachePath,
  coreBodyTemperatureCachePath,
  enrichCalculatedExerciseLoads,
  enrichCalculatedIntensityFactors,
  enrichCalculatedTrainingEffects,
  enrichCoreBodyTemperature,
  enrichRouteLessHeartRate,
  enrichRunPaceZones,
  enrichRunDynamics,
  enrichSwimMetrics,
  garminCachePath,
  type LoadedStravaPayload,
  ouraCachePath,
  stravaCachePath,
  weatherCachePath,
} from '../../util/strava-payload'
import {
  triathlonActivityFeedRoutes,
  triathlonDaySlug,
  triathlonFeedScopeFromSlug,
} from '../../util/triathlon-date-route'
import { buildTriathlonDailyAnalytics } from '../../util/triathlon-day-analytics'
import {
  parseTriathlonMaintenance,
  type TriathlonMaintenance,
} from '../../util/triathlon-maintenance'
import { buildTriathlonMarkdown } from '../../util/triathlon-markdown'
import { buildStravaActivityIndex } from '../../util/triathlon-shortcut'
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
  applyManualSauna,
  applyManualStrength,
  buildPayload,
  emptyHealth,
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
    header: [Cursor()],
    beforeBody: [],
    afterBody: [],
    sidebar: [],
    pageBody: TriathlonPage(),
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()
  const DayPage = TriathlonDayPage()
  let emittedTemporalSlugs = new Set<FullSlug>()
  const dayRenderCache = new Map<FullSlug, string>()

  async function* emitAll(
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
  ): AsyncGenerator<FilePath> {
    const cache = await readCache()
    const oura = await readOura()
    const garmin = await readGarmin()
    const apple = await readApple()
    const core = await readCoreBodyTemperature()
    const weather = await readWeather()
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
      const maintenance = parseTriathlonMaintenance(file.data.frontmatter?.['maintenance'])
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
      applyManualSauna(payload, tracking?.sauna ?? [], oura?.heartRate ?? [])
      for (const t of tracking?.days ?? [])
        if (t.windKph != null) {
          const h = payload.health[t.date] ?? emptyHealth()
          payload.health[t.date] = { ...h, windKph: t.windKph, windDir: t.windDir ?? h.windDir }
        }
      enrichSwimMetrics(payload, apple)
      enrichRunDynamics(payload, apple)
      enrichRouteLessHeartRate(payload, apple)
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
          criticalPower: payload.criticalPower,
          criticalPowerYear: payload.criticalPowerYear,
          ftp: ATHLETE.ftp,
          goalFtp: ATHLETE.goalFTP,
        },
        zones: payload.zones,
        activityDetails: payload.details,
        since: typeof since === 'string' ? since : undefined,
      })
      enrichRunPaceZones(payload, analytics.distributions)
      enrichCalculatedIntensityFactors(payload, analytics.activities, ATHLETE.ftp, ATHLETE.lt)
      enrichCalculatedExerciseLoads(payload)
      enrichCalculatedTrainingEffects(payload)
      const dailyAnalytics = buildTriathlonDailyAnalytics(analytics, oura?.details, payload.details)
      const renderPayload: LoadedStravaPayload = { ...payload, dailyAnalytics }
      const detailPayload: StravaDetailPayload = {
        details: payload.details,
        swimTrend: payload.swimTrend,
        health: payload.health,
        dailyAnalytics,
        zones: payload.zones,
        powerCurveRef: payload.powerCurveRef,
        powerCurveYearRef: payload.powerCurveYearRef,
        powerCurveYear: payload.powerCurveYear,
        criticalPower: payload.criticalPower,
        criticalPowerYear: payload.criticalPowerYear,
        ftp: ATHLETE.ftp,
        goalFtp: ATHLETE.goalFTP,
        vt1Hr: latestVo2?.vt1Hr ?? null,
        matchedRuns,
        matchedRides,
      }
      const detailArtifacts = serializeStravaDetails(detailPayload)
      const plans = parseTrainingPlans(tree as unknown as HtmlRoot)
      const triathlonRenderData: TriathlonRenderData = {
        analytics,
        plans,
        weather: weather?.current ?? null,
      }
      const dataFeed = buildDataFeed(cache, analytics, {
        oura,
        apple,
        weather,
        garmin,
        weights: tracking?.days,
        trainingExclusions: tracking?.trainingExclusions,
        zones: payload.zones,
        activityDetails: payload.details,
      })
      await cacheDataFeed(dataFeed)
      yield* await Promise.all([
        ...detailArtifacts.shards.map(shard =>
          write({
            ctx,
            slug: `static/strava-detail/${shard.name}`,
            ext: '.json',
            content: shard.content,
          }),
        ),
        write({
          ctx,
          slug: 'static/strava-detail' as FullSlug,
          ext: '.json',
          content: detailArtifacts.manifest,
        }),
        write({
          ctx,
          slug: 'static/strava-activity-index' as FullSlug,
          ext: '.json',
          content: JSON.stringify(buildStravaActivityIndex(payload.details)),
        }),
        write({
          ctx,
          slug: 'static/analytics' as FullSlug,
          ext: '.json',
          content: JSON.stringify(analytics),
        }),
        write({
          ctx,
          slug: 'static/oura-detail' as FullSlug,
          ext: '.json',
          content: JSON.stringify(oura?.details ?? {}),
        }),
        write({
          ctx,
          slug: 'static/training' as FullSlug,
          ext: '.json',
          content: JSON.stringify({ plans }),
        }),
        write({ ctx, slug: 'static/triathlon/data' as FullSlug, ext: '.jsonl', content: dataFeed }),
      ])
      const slug = file.data.slug!
      const externalResources = pageResources(pathToRoot(slug), resources, ctx)
      const componentData: QuartzComponentProps = {
        ctx,
        fileData: {
          ...file.data,
          stravaPayload: renderPayload,
          triathlonMaintenance: maintenance,
          triathlonRenderData,
        },
        externalResources,
        cfg: ctx.cfg.configuration,
        children: [],
        tree: tree as Node,
        allFiles,
      }
      const html = renderPage(ctx, slug, componentData, opts, externalResources, false)
      yield await write({ ctx, content: html, slug, ext: '.html' })

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
          tools: { conversions: CONVERSIONS, gear: GEAR, maintenance },
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
            pageLayout: 'triathlon',
            tags: [],
          },
        })
        const subResources = pageResources(pathToRoot(subSlug), resources, ctx)
        const subData: QuartzComponentProps = {
          ctx,
          fileData: {
            ...subFile.data,
            stravaPayload: renderPayload,
            triathlonMaintenance: maintenance,
            triathlonRenderData,
          },
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
      yield* (
        await mapConcurrent(TRI_SUBVIEWS, defaultIoConcurrency, view =>
          renderSubPage(`triathlon/${view}` as FullSlug, view, `triathlon · ${view}`),
        )
      ).flat()

      const feedRoutes = triathlonActivityFeedRoutes(
        Object.fromEntries(payload.days.map(day => [day.date, { date: day.date }])),
      )
      for (const { slug: feedSlug } of feedRoutes) nextTemporalSlugs.add(feedSlug)
      yield* (
        await mapConcurrent(feedRoutes, defaultIoConcurrency, ({ slug: feedSlug, title }) =>
          renderSubPage(feedSlug, 'on', title, triathlonFeedScopeFromSlug(feedSlug)?.prefix ?? ''),
        )
      ).flat()

      const dayRoutes = payload.days.flatMap(day => {
        const slug = triathlonDaySlug(day.date)
        return slug ? [{ date: day.date, slug }] : []
      })
      for (const { slug: daySlug } of dayRoutes) nextTemporalSlugs.add(daySlug)
      const location = file.data.frontmatter?.['location']
      const dayRenderCacheEligible = ctx.argv.watch
      const globalDayKey = dayRenderCacheEligible
        ? hashContent({
            payload: { ...payload, details: undefined, health: undefined },
            maintenance,
            location,
            today: new Date().toISOString().slice(0, 10),
          })
        : ''
      yield* (
        await mapConcurrent(
          dayRoutes,
          defaultEmitterConcurrency,
          async ({ date, slug: daySlug }) => {
            if (dayRenderCacheEligible) {
              const dayKey = hashContent({
                activities: Object.entries(payload.details).filter(([, d]) => d.date === date),
                health: payload.health[date] ?? null,
                dailyAnalytics: dailyAnalytics[date] ?? null,
                tracking: tracking?.days?.find(t => t.date === date) ?? null,
                global: globalDayKey,
              })
              if (dayRenderCache.get(daySlug) === dayKey) return []
              dayRenderCache.set(daySlug, dayKey)
            }
            const title = `triathlon · ${date}`
            const description = triathlonDescription('day', payload.totalCount, plans.length, date)
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
              tools: { conversions: CONVERSIONS, gear: GEAR, maintenance },
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
                pageLayout: 'triathlon',
                tags: [],
                ...(typeof location === 'string' ? { location } : {}),
              },
            })
            const dayResources = pageResources(pathToRoot(daySlug), resources, ctx)
            const dayData: QuartzComponentProps = {
              ctx,
              fileData: { ...dayFile.data, stravaPayload: renderPayload },
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
          },
        )
      ).flat()
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
      yield* emitAll(ctx, content, resources)
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const touched = changeEvents.some(event => event.file && isTriathlon(event.file.data))
      if (!touched) return
      yield* emitAll(ctx, content, resources)
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    stravaPayload: LoadedStravaPayload
    triathlonMaintenance: TriathlonMaintenance | null
    triathlonRenderData: TriathlonRenderData
  }
}
