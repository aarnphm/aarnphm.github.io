import { decodeCalcShare } from '../../../util/triathlon-calculator'
import { decodeActivityComparisonAnchor } from '../../../util/triathlon-comparison'
import { TRI_TIRE_PRESSURE_OPEN_EVENT } from '../../../util/triathlon-tire-pressure'
import { setupActivityComparisonEmbeds } from '../activity/embeds'
import { setupDayEmbeds } from '../activity/embeds'
import { setupMatchedActivities } from '../activity/matched'
import { setupPowerCurveActivityLinks } from '../activity/power-links'
import { createDayCardFacade, type DayCardFacade } from '../activity/public-api'
import { setupChartScrub } from '../activity/scrub'
import { setupAnalytics } from '../analytics/controller'
import { setupMap } from '../maps/controller'
import { setupAxisLabels } from '../shell/axis-labels'
import { setupCommandPalette } from '../shell/command-palette'
import { setupDropdown } from '../shell/dropdown'
import { setupFeed } from '../shell/feed'
import { setupGloss } from '../shell/glossary'
import { setupDistanceUnits, setupI18n, setupPaceUnit } from '../shell/presentation'
import { setupShortcuts } from '../shell/shortcuts'
import { setup } from '../shell/timeline'
import { setupCalc } from '../tools/calculator-controller'
import { setupCheat } from '../tools/cheat'
import { setupGearRatios } from '../tools/gear-ratios'
import { setupPaceForecast } from '../tools/pace-forecast'
import { setupTirePressure } from '../tools/tire-pressure'
import { setupTraining } from '../training/controller'
import { createTriathlonContext } from './context'
import { readTriPanelsFullscreen } from './preferences'

export interface MountedTriathlon {
  cleanup(): void
  dayCard: DayCardFacade
}

export const mountTriathlon = (signal: AbortSignal): MountedTriathlon => {
  const context = createTriathlonContext(signal)
  const cleanups: (() => void)[] = []
  const addCleanup = (cleanup: (() => void) | null | undefined): void => {
    if (cleanup) cleanups.push(cleanup)
  }
  cleanups.push(() => context.preferences.dispose())
  const root = context.root
  if (root) {
    if (!root.dataset.triView)
      root.classList.toggle('tri-panels-fullscreen', readTriPanelsFullscreen())
  }
  addCleanup(setupDayEmbeds(context))
  addCleanup(setupActivityComparisonEmbeds(context))
  addCleanup(setupPowerCurveActivityLinks(document.body, context))
  addCleanup(setupChartScrub(document.body, () => context.presentation))
  addCleanup(setupMatchedActivities(document.body))
  addCleanup(setupGloss(document.body, () => context.presentation.locale))
  if (root) {
    addCleanup(setupI18n(root, context))
    addCleanup(setupDistanceUnits(root, context))
    addCleanup(setupTirePressure(root))
    addCleanup(setupCommandPalette(root, context))
    addCleanup(setup(root, context))
    addCleanup(setupCalc(root, context))
    addCleanup(setupPaceForecast(root, context))
    addCleanup(setupGearRatios(root, context))
    addCleanup(setupDropdown(root, '.tri-gear-wrap', '.tri-gear-btn', '.tri-gear', 'tri-gear-open'))
    addCleanup(setupDropdown(root, '.tri-pace-wrap', '.tri-pace-btn', '.tri-pace', 'tri-pace-open'))
    addCleanup(setupPaceUnit(root, context))
    addCleanup(setupCheat(root, context))
    addCleanup(setupAnalytics(root, context))
    addCleanup(setupFeed(root, context))
    addCleanup(setupTraining(root, context))
    addCleanup(setupMap(root, context))
    addCleanup(setupAxisLabels(root))
    addCleanup(setupShortcuts(root))
    const hashDate = /^#(\d{4}-\d{2}-\d{2})$/.exec(window.location.hash)?.[1]
    if (hashDate)
      window.dispatchEvent(new CustomEvent('tri:focus-day', { detail: { date: hashDate } }))
    const calcHash = /^#(calculator-[a-z0-9-]+)$/.exec(window.location.hash)?.[1]
    const calcShare = calcHash ? decodeCalcShare(calcHash) : null
    if (calcShare)
      window.dispatchEvent(new CustomEvent('tri:calc-fill', { detail: { share: calcShare } }))
    if (decodeActivityComparisonAnchor(window.location.hash))
      window.dispatchEvent(
        new CustomEvent('tri:comparison-fill', { detail: { anchor: window.location.hash } }),
      )
    if (window.location.hash === '#tire-pressure')
      root.dispatchEvent(new CustomEvent(TRI_TIRE_PRESSURE_OPEN_EVENT))
  }
  let active = true
  const cleanup = (): void => {
    if (!active) return
    active = false
    for (let i = cleanups.length - 1; i >= 0; i--) cleanups[i]()
  }
  signal.addEventListener('abort', cleanup, { once: true })
  return { cleanup, dayCard: createDayCardFacade(context) }
}
