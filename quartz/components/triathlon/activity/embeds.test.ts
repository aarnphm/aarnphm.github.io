import assert from 'node:assert/strict'
import test from 'node:test'
import { dayCardActivitiesExpanded } from '../../../util/triathlon-card'
import { alignedActivitySectionMargins } from './activity-layout'
import { dayExtrasFromDataset } from './embed-settings'

test('aligns training effects by their natural top within each activity row', () => {
  assert.deepEqual(
    alignedActivitySectionMargins([
      { activityTop: 100, sectionTop: 2_963.734375, marginTop: 0 },
      { activityTop: 100.0078125, sectionTop: 2_963.734375, marginTop: 0 },
      { activityTop: 100, sectionTop: 3_651.2109375, marginTop: 2_410.875 },
      { activityTop: 4_000, sectionTop: 4_500, marginTop: 0 },
    ]),
    [0, 0, 1_723.3984375, 0],
  )
})

test('restores activity selection, exclusions, and trace settings from the embed dataset', () => {
  const dataset: DOMStringMap = {
    triathlonActivityId: '19731411847',
    triathlonFilter: '19471122670&19476629599',
    triathlonSettings: 'matched-rides:false&power-balance:true&expanded:true',
    triathlonAnalytics: '1',
    triathlonEmbedded: '1',
  }

  assert.deepEqual(dayExtrasFromDataset(dataset), {
    location: undefined,
    event: undefined,
    sport: undefined,
    activityId: '19731411847',
    excludedActivityIds: ['19471122670', '19476629599'],
    settings: { 'matched-rides': false, 'power-balance': true, expanded: true },
    analytics: true,
    expanded: false,
    embedded: true,
    dateHref: undefined,
  })
  assert.equal(dayCardActivitiesExpanded(dayExtrasFromDataset(dataset)), true)
  assert.equal(
    dayCardActivitiesExpanded(
      dayExtrasFromDataset({ triathlonSport: 'bike', triathlonSettings: 'expanded:false' }),
    ),
    false,
  )
})

test('aligns environment sections without accumulating the previous layout offset', () => {
  const positions = [
    { activityTop: 100, sectionTop: 3308.625, marginTop: 0 },
    { activityTop: 100, sectionTop: 3480.9609375, marginTop: 0 },
  ]
  const margins = alignedActivitySectionMargins(positions)
  assert.deepEqual(margins, [172.3359375, 0])
  assert.deepEqual(
    alignedActivitySectionMargins(
      positions.map((position, index) => ({
        ...position,
        sectionTop: position.sectionTop + margins[index],
        marginTop: margins[index],
      })),
    ),
    margins,
  )
})

test('clears alignment when activity columns stack or only one section is visible', () => {
  assert.deepEqual(
    alignedActivitySectionMargins([
      { activityTop: 100, sectionTop: 3480.9609375, marginTop: 172.3359375 },
      { activityTop: 5000, sectionTop: 8308.625, marginTop: 0 },
    ]),
    [0, 0],
  )
  assert.deepEqual(
    alignedActivitySectionMargins([
      { activityTop: 100, sectionTop: 3480.9609375, marginTop: 172.3359375 },
    ]),
    [0],
  )
})
