import assert from 'node:assert/strict'
import { test } from 'node:test'
import { campSleeps } from '../../util/world-schedule'
import { campMayDepart, campMaySleep, campRoutine } from './camp-routine'

const minute = 60 * 1000
const hour = 60 * minute

test('camp schedules use hours and retain different passing and overnight stays', () => {
  const now = new Date(2026, 9, 11, 20).getTime()
  for (let seed = 0; seed < 24; seed++) {
    const camps = [campRoutine(seed, 0, now), campRoutine(seed, 1, now)]
    assert.equal(new Set(camps.map(camp => camp.variation)).size, 2)
    for (const camp of camps) {
      const minimum = camp.variation === 'overnight' ? 8 : 2
      assert.ok(camp.stay >= minimum * hour && camp.stay <= (minimum + 2) * hour)
      assert.equal(camp.leaveAt - now, camp.stay)
      assert.ok(camp.sleepAt - now >= 20 * minute && camp.sleepAt - now <= 40 * minute)
      assert.ok(camp.absence >= 30 * minute && camp.absence <= hour)
      assert.equal(campMayDepart(camp, now + 5 * minute, false), false)
      assert.equal(campMayDepart(camp, camp.leaveAt - 1, false), false)
      assert.equal(campMayDepart(camp, camp.leaveAt, false), true)
      assert.equal(campMayDepart(camp, camp.leaveAt, true), camp.variation === 'passing')
    }
  }
})

test('sleep waits for settling, the solar bedtime, and an overnight camp', () => {
  // The bedtime: two hours past civil dusk (sun near -20°) until civil dawn, by the visitor's sun.
  const sun = (alt: number, hour: number) => ({ sun: { alt, across: 0, bearing: 0 }, hour })
  assert.equal(campSleeps(sun(-12, 19)), false)
  assert.equal(campSleeps(sun(-24, 21)), true)
  assert.equal(campSleeps(sun(-30, 0.2)), true)
  assert.equal(campSleeps(sun(-4, 7)), false)
  assert.equal(campSleeps(sun(5, 21)), false)

  const camp = campRoutine(1, 0, new Date(2026, 9, 11, 21).getTime())
  assert.equal(camp.variation, 'overnight')
  assert.equal(campMaySleep(camp, camp.sleepAt - 1, true), false)
  assert.equal(campMaySleep(camp, camp.sleepAt, true), true)
  assert.equal(campMaySleep(camp, camp.sleepAt, false), false)
  const passing = campRoutine(0, 0, new Date(2026, 9, 11, 21).getTime())
  assert.equal(campMaySleep(passing, new Date(2026, 9, 11, 23).getTime(), true), false)
})
