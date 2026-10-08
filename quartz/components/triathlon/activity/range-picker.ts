import type { ActivityAnalysisRange, StravaActivityDetail } from '../../../plugins/stores/strava'
import { cyclingWorkoutLaps, zoneClock } from '../../../util/triathlon-card'
import { buildSelect, type SelectGroup, type SelectOption } from '../../controls/select'

type Range = ActivityAnalysisRange | null

export const buildActivityRangePicker = (
  activity: StravaActivityDetail,
  text: (key: string) => string,
  select: (range: Range) => void,
): { element: HTMLElement; dispose: () => void } => {
  const lapNumbers = new Map(cyclingWorkoutLaps(activity).map(lap => [lap.range.id, lap.index]))
  const option = (range: ActivityAnalysisRange): SelectOption<Range> => {
    const number = range.kind === 'lap' ? lapNumbers.get(range.id) : undefined
    const name = number === undefined ? range.label : `${text('lap')} ${number}`
    return { value: range, label: `${name} · ${zoneClock(range.durationS)}` }
  }
  const groups: SelectGroup<Range>[] = [
    { options: [{ value: null, label: text('entire activity') }] },
  ]
  for (const [kind, label] of [
    ['lap', 'Laps'],
    ['climb', 'Summit Freeride'],
    ['segment', 'Segments'],
  ]) {
    const ranges = activity.analysisRanges.filter(
      range => range.kind === kind && range.endElapsedS > range.startElapsedS,
    )
    if (ranges.length) groups.push({ label: text(label), options: ranges.map(option) })
  }
  const picker = buildSelect<Range>({
    id: `tri-workspace-ranges-${activity.id}`,
    label: text('activity range'),
    groups,
    selected: null,
    onSelect: select,
    className: 'tri-workspace-range',
  })
  return { element: picker.element, dispose: picker.mount() }
}
