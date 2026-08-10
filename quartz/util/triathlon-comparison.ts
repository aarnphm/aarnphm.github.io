export const ACTIVITY_COMPARISON_ANCHOR_PREFIX = 'comparison-'

const activityComparisonIds = (activityIds: readonly string[]): string[] | null => {
  if (activityIds.length < 2) return null
  if (activityIds.some(activityId => !/^[1-9]\d*$/.test(activityId))) return null
  if (new Set(activityIds).size !== activityIds.length) return null
  return [...activityIds]
}

export const encodeActivityComparisonAnchor = (activityIds: readonly string[]): string | null => {
  const ids = activityComparisonIds(activityIds)
  return ids ? `${ACTIVITY_COMPARISON_ANCHOR_PREFIX}${ids.join('-')}` : null
}

export const decodeActivityComparisonAnchor = (value: string): string[] | null => {
  const anchor = value.startsWith('#') ? value.slice(1) : value
  if (!anchor.startsWith(ACTIVITY_COMPARISON_ANCHOR_PREFIX)) return null
  return activityComparisonIds(anchor.slice(ACTIVITY_COMPARISON_ANCHOR_PREFIX.length).split('-'))
}

export const activityComparisonEmbed = (activityIds: readonly string[]): string | null => {
  const anchor = encodeActivityComparisonAnchor(activityIds)
  return anchor ? `![[triathlon#${anchor}]]` : null
}
