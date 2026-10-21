import fs from 'node:fs/promises'
import type { TriView } from '../../components/pages/triathlon-panels'
import type { StravaDetailPayload } from '../../util/strava-detail'
import type { LoadedStravaPayload } from '../../util/strava-payload'
import type { QuartzPluginData } from '../vfile'
import { joinSegments, QUARTZ } from '../../util/path'
import { ATHLETE } from '../stores/analytics'

export const TRI_SUBVIEWS: TriView[] = ['tools', 'calc', 'analytics', 'maps', 'calendar', 'on']

// The feed lives on /triathlon/on behind its log · feed switch; old feed links open it there.
export const TRIATHLON_FEED_REDIRECT = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>triathlon · feed</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="on?view=feed">
<script>location.replace(location.pathname.replace(/\\/feed\\/?$/, "/on") + "?view=feed" + location.hash)</script>
<meta http-equiv="refresh" content="0; url=on?view=feed">
</head>
</html>
`

const TRIATHLON_DATA_CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache', 'triathlon')
const TRIATHLON_DATA_CACHE_PATH = joinSegments(TRIATHLON_DATA_CACHE_DIR, 'data.jsonl')

export async function cacheDataFeed(content: string): Promise<void> {
  await fs.mkdir(TRIATHLON_DATA_CACHE_DIR, { recursive: true })
  await fs.writeFile(TRIATHLON_DATA_CACHE_PATH, content)
}

export const isTriathlon = (data: QuartzPluginData): boolean =>
  data.frontmatter?.layout === 'triathlon'

export const triathlonDescription = (
  view: TriView | 'feed' | 'day',
  activityCount: number,
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
    case 'calendar':
      return 'Race dates and the TrainingPeaks training calendar, with weekly planned and completed sessions.'
    case 'feed':
      return `Generated training feed across ${activityCount} activities.`
    case 'on':
      return `Generated training log for ${scopePrefix || 'the complete activity window'}.`
    case 'day':
      return `Generated activity data and telemetry for ${scopePrefix}.`
  }
}

export const stravaDetailPayload = (
  payload: LoadedStravaPayload,
  extra: Pick<StravaDetailPayload, 'vt1Hr' | 'matchedRuns' | 'matchedRides'>,
): StravaDetailPayload => ({
  details: payload.details,
  swimTrend: payload.swimTrend,
  health: payload.health,
  dailyAnalytics: payload.dailyAnalytics,
  zones: payload.zones,
  powerCurveRef: payload.powerCurveRef,
  powerCurveYearRef: payload.powerCurveYearRef,
  runPowerCurveRef: payload.runPowerCurveRef,
  runPowerCurveYearRef: payload.runPowerCurveYearRef,
  powerCurveYear: payload.powerCurveYear,
  criticalPower: payload.criticalPower,
  criticalPowerYear: payload.criticalPowerYear,
  ftp: ATHLETE.ftp,
  goalFtp: ATHLETE.goalFTP,
  ...extra,
})
