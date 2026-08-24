import type { Code, Root } from 'mdast'
import { remove } from 'unist-util-remove'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import {
  ManualFuelingEntry,
  ManualSaunaEntry,
  ManualStrengthEntry,
  parseTrackingBlock,
  RaceEvent,
  TrackEntry,
  TrackingData,
  TrainingExclusion,
} from '../stores/tracking'

export const Tracking: QuartzTransformerPlugin = () => ({
  name: 'Tracking',
  markdownPlugins() {
    return [
      () => (tree: Root, file) => {
        const days: TrackEntry[] = []
        const fueling: ManualFuelingEntry[] = []
        const strength: ManualStrengthEntry[] = []
        const sauna: ManualSaunaEntry[] = []
        const trainingExclusions: TrainingExclusion[] = []
        visit(tree, 'code', (node: Code) => {
          if (node.lang !== 'tracking') return
          const entry = parseTrackingBlock(node.meta, node.value ?? '')
          if (!entry) return
          days.push(entry.day)
          if (entry.fueling) fueling.push(entry.fueling)
          if (entry.strength) strength.push(entry.strength)
          if (entry.sauna) sauna.push(entry.sauna)
          if (entry.trainingExclusion) trainingExclusions.push(entry.trainingExclusion)
        })
        if (days.length === 0) return
        days.sort((a, b) => a.date.localeCompare(b.date))
        const races: RaceEvent[] = days
          .filter(d => d.race || d.event != null)
          .map(d => ({ date: d.date, event: d.event }))
        const data: TrackingData = { days, races, fueling, strength, sauna, trainingExclusions }
        file.data.tracking = data
        remove(tree, node => node.type === 'code' && (node as Code).lang === 'tracking')
      },
    ]
  },
})

declare module 'vfile' {
  interface DataMap {
    tracking: TrackingData
  }
}
