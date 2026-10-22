import type {
  ActivityComparisonScrubContext,
  ActivityComparisonScrubState,
} from './comparison-graph'
import { activityGearRatioDistribution } from '../../../util/triathlon-card'
import { activityPowerDistributionPercentages } from '../../../util/triathlon-card'
import { activityZonePercentages } from '../../../util/triathlon-card'
import { bindActivityComparisonGraph } from './comparison-graph'
import { positionActivityComparisonCursor } from './comparison-graph'

/** Scrubs for the binned comparison charts: 25 W power bins, gear ratios and zones. */
export const wireComparisonBinnedCharts = (
  scrub: ActivityComparisonScrubContext,
): (() => void)[] => {
  const { activities, charts, text } = scrub
  const { setReadout, showChartCursors, activate, render, deactivate } = scrub
  const cleanups: (() => void)[] = []
  const distributionChart = charts.find(
    chart => chart.dataset.compareChart === 'power-distribution',
  )
  const distributionGraph = distributionChart?.querySelector<SVGElement>(
    '.tri-compare-distribution-graph',
  )
  const distributions = activities.map(activity =>
    activityPowerDistributionPercentages(activity.powerHist),
  )
  const binCount = Math.max(0, ...distributions.map(values => values.length))
  if (distributionChart && distributionGraph && binCount >= 2) {
    const distributionState: ActivityComparisonScrubState = { fraction: 0 }
    const showDistribution = (fraction: number) => {
      distributionState.fraction = Math.min(1, Math.max(0, fraction))
      const index = Math.round(distributionState.fraction * (binCount - 1))
      const selectedFraction = index / (binCount - 1)
      const startWatts = index * 25
      const range = `${startWatts}–${startWatts + 24} W`
      positionActivityComparisonCursor(distributionGraph, selectedFraction)
      const values = activities.map((activity, activityIndex) => {
        const distribution = distributions[activityIndex]
        const value = distribution.length === 0 ? null : (distribution[index] ?? 0)
        return { activity, value: value == null ? '–' : `${value.toFixed(1)}%` }
      })
      showChartCursors([distributionChart])
      setReadout(
        'power-distribution',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '–' })),
        range,
      )
      distributionGraph.setAttribute('aria-valuenow', `${startWatts}`)
      distributionGraph.setAttribute(
        'aria-valuetext',
        `${range}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    cleanups.push(
      bindActivityComparisonGraph(
        distributionGraph,
        distributionState,
        showDistribution,
        activate,
        render,
        deactivate,
        1 / (binCount - 1),
      ),
    )
  }

  const gearRatioChart = charts.find(
    chart => chart.dataset.compareChart === 'gear-ratio-distribution',
  )
  const gearRatioGraph = gearRatioChart?.querySelector<SVGElement>(
    '.tri-compare-distribution-graph',
  )
  const gearRatioDistributions = activities.map(activity => activityGearRatioDistribution(activity))
  const gearRatios = [
    ...new Set(gearRatioDistributions.flatMap(points => points.map(point => point.ratio))),
  ].sort((left, right) => left - right)
  if (gearRatioChart && gearRatioGraph && gearRatios.length > 0) {
    const gearRatioState: ActivityComparisonScrubState = { fraction: 0 }
    const showGearRatio = (fraction: number) => {
      const index = Math.round(Math.min(1, Math.max(0, fraction)) * (gearRatios.length - 1))
      const selectedFraction = gearRatios.length <= 1 ? 0.5 : index / (gearRatios.length - 1)
      gearRatioState.fraction = selectedFraction
      const ratio = gearRatios[index]
      positionActivityComparisonCursor(gearRatioGraph, selectedFraction)
      const values = activities.map((activity, activityIndex) => {
        const distribution = gearRatioDistributions[activityIndex]
        const point = distribution.find(point => point.ratio === ratio)
        const pairings = point?.pairings
          .map(({ frontTeeth, rearTeeth }) => `${frontTeeth}×${rearTeeth}T`)
          .join(' / ')
        const percentage = `${(point?.percentage ?? 0).toFixed(1)}%`
        const value =
          distribution.length === 0 ? '–' : pairings ? `${pairings} · ${percentage}` : percentage
        return { activity, value }
      })
      showChartCursors([gearRatioChart])
      setReadout(
        'gear-ratio-distribution',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '–' })),
        `${ratio.toFixed(2)}×`,
      )
      gearRatioGraph.setAttribute('aria-valuenow', `${index}`)
      gearRatioGraph.setAttribute(
        'aria-valuetext',
        `${ratio.toFixed(2)}×; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    cleanups.push(
      bindActivityComparisonGraph(
        gearRatioGraph,
        gearRatioState,
        showGearRatio,
        activate,
        render,
        deactivate,
        gearRatios.length <= 1 ? 1 : 1 / (gearRatios.length - 1),
      ),
    )
  }

  const zoneKinds: readonly ('hr' | 'power')[] = ['hr', 'power']
  for (const kind of zoneKinds) {
    const chart = charts.find(candidate => candidate.dataset.compareChart === `${kind}-zones`)
    const graph = chart?.querySelector<SVGElement>('.tri-compare-graph')
    const zones = activities.map(activity =>
      activityZonePercentages(kind === 'hr' ? activity.hrZones : activity.powerZones),
    )
    const zoneCount = Math.max(0, ...zones.map(values => values.length))
    if (!chart || !graph || zoneCount === 0) continue
    const zoneState: ActivityComparisonScrubState = { fraction: 0 }
    const showZone = (fraction: number) => {
      // Zones are bands: the pointer picks the band under it and the state rests at its centre.
      const index = Math.min(
        zoneCount - 1,
        Math.floor(Math.min(1, Math.max(0, fraction)) * zoneCount),
      )
      zoneState.fraction = (index + 0.5) / zoneCount
      graph
        .querySelector('.tri-compare-zone-focus')
        ?.setAttribute('x', ((index * 100) / zoneCount).toFixed(2))
      const values = activities.map((activity, activityIndex) => {
        const value = zones[activityIndex][index]
        return { activity, value: value == null ? '–' : `${value.toFixed(1)}%` }
      })
      showChartCursors([chart])
      setReadout(
        `${kind}-zones`,
        values.map(({ activity, value }) => ({ activity, value, missing: value === '–' })),
        `Z${index + 1}`,
      )
      graph.setAttribute('aria-valuenow', `${index + 1}`)
      graph.setAttribute(
        'aria-valuetext',
        `Z${index + 1}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    const keyboardStep = 1 / zoneCount
    cleanups.push(
      bindActivityComparisonGraph(
        graph,
        zoneState,
        showZone,
        activate,
        render,
        deactivate,
        keyboardStep,
      ),
    )
  }
  return cleanups
}
