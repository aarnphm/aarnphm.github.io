import { load } from 'js-yaml'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { promisify } from 'node:util'
import type { RawStravaActivity, StravaRawCache } from '../plugins/stores/strava'
import { parseTriathlonMaintenance } from './triathlon-maintenance'
import { isRecord } from './type-guards'

const componentNames = ['OSPW', 'bottom bracket', 'brake pads', 'chainrings', 'saddle', 'wheelset']
const fixture = `---
title: 'maintenance fixture' # preserve this comment
strava: '2026-11-10'
maintenance:
${componentNames
  .map(
    name => `  ${name}:
    - - type: current ${name}
      - bikeId: b18037537
      - distance: 999.00 # miles
      - range: [{ start: '2026-11-12', end: null }]
    - - type: retired ${name}
      - bikeId: b18037537
      - distance: 999.00
      - range: [{ start: '2026-11-10', end: '2026-11-11' }]`,
  )
  .join('\n')}
  chain:
    '4':
      bikeId: b18037537
      since: '2026-11-10'
      distance: 999.00
      lubricant: null
      waxed: true
      status: backup
      range:
        - { start: '2026-11-10', end: '2026-11-11' }
      cycles:
        - start: '2026-11-12'
          end: null
          distance: 999.00 # fresh wax, no rides yet
    '5':
      bikeId: b18037537
      since: '2026-11-12'
      distance: 999.00
      lubricant: null
      waxed: true
      status: active
      range: [{ start: '2026-11-12', end: null }]
      cycles:
        - { start: '2026-11-12', end: null }
  tires:
    front:
      tires:
        - - { type: old front, distance: 999.00, reason: null }
          - bikeId: b18037537
          - { start: '2026-11-10', end: '2026-11-12' }
        - - type: new front
            reason: null
          - bikeId: b18037537
          - range: [{ start: '2026-11-12', end: null }]
    rear:
      tires:
        - - { type: old rear, distance: 999.00, reason: null }
          - bikeId: b18037537
          - { start: '2026-11-10', end: '2026-11-11' }
        - - { type: new rear, distance: 999.00, reason: null }
          - bikeId: b18037537
          - { start: '2026-11-14', end: null }
      tube:
        - - type: rear tube
            reason: null
          - bikeId: b18037537
          - distance: 999.00
          - range: [{start: '2026-11-10', end: null}, {start: '2026-11-12', end: null}]
  service:
    soloist:
      - date: '2026-11-11'
        bikeId: b18037537
        place: workshop
        distance: 999.00
other:
  distance: 123.456 # unrelated distance
---

Keep the body, [[links]], and spacing exactly as authored.
`

const ride = (
  id: number,
  date: string,
  miles: number,
  sportType = 'Ride',
  gearId: string | null = 'b18037537',
): RawStravaActivity => ({
  id,
  gearId,
  name: `ride ${id}`,
  sportType,
  distance: miles * 1609.344,
  startDate: `${date}T02:00:00Z`,
  startDateLocal: `${date}T23:00:00`,
  movingTime: 100,
  elapsedTime: 100,
  totalElevationGain: 0,
  averageSpeed: 1,
})

const normalized = (source: string) => {
  const document: unknown = load(source.split('---')[1])
  assert.ok(isRecord(document))
  const maintenance = parseTriathlonMaintenance(document.maintenance)
  assert.ok(maintenance)
  return maintenance
}

// Exercise the same post-cache refresh used by Strava, with real files in a fresh process.
// Catch frozen component totals, backup mileage, repeated-sync double counts, deleted rides,
// date-boundary attribution, missing cycle distances, and unintended YAML/body rewrites.
test('sync refresh recalculates every maintenance category and preserves chain rotations', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'maintenance-sync-'))
  const notePath = join(directory, 'content/triathlon.md')
  const cachePath = join(directory, 'quartz/.quartz-cache/strava.json')
  const artifacts = join(
    process.cwd(),
    'docs/verification/triathlon-maintenance-auto-sync/artifacts',
  )
  const snapshots: Array<{ stage: string; maintenance: ReturnType<typeof normalized> }> = []
  const refresh = async () => {
    await promisify(execFile)(
      process.execPath,
      [
        '--import',
        import.meta.resolve('tsx'),
        '--input-type=module',
        '-e',
        `import { refreshTriathlonRouteSource } from ${JSON.stringify(new URL('./triathlon-cache.ts', import.meta.url).href)}; await refreshTriathlonRouteSource();`,
      ],
      { cwd: directory, timeout: 30_000 },
    )
    return readFile(notePath, 'utf8')
  }
  const saveRides = async (activities: RawStravaActivity[]) => {
    const cache: StravaRawCache = {
      athleteId: 1,
      auth: { refreshToken: '', obtainedAt: 0 },
      lastSync: Date.now(),
      lastActivityStart: 0,
      activities: Object.fromEntries(activities.map(activity => [String(activity.id), activity])),
    }
    await writeFile(cachePath, JSON.stringify(cache))
  }
  try {
    await mkdir(join(directory, 'content'), { recursive: true })
    await mkdir(join(directory, 'quartz/.quartz-cache'), { recursive: true })
    await writeFile(notePath, fixture)
    const rides = [
      ride(1, '2026-11-11', 1),
      ride(2, '2026-11-12', 2),
      ride(10, '2026-11-12', 100, 'VirtualRide'),
      ride(11, '2026-11-12', 100, 'Run'),
      ride(20, '2026-11-11', 30, 'Ride', 'b18595115'),
      ride(21, '2026-11-12', 40, 'Ride', 'b18595115'),
      ride(22, '2026-11-12', 100, 'Ride', null),
      { ...ride(23, '2026-11-12', 100), gearId: undefined },
      ride(24, '2026-11-12', 100, 'Ride', 'b999'),
    ]
    await saveRides(rides)
    const first = await refresh()
    const initial = normalized(first)
    assert.deepEqual(
      initial.components.map(component => component.distanceMiles),
      componentNames.flatMap(() => [2, 1]),
    )
    assert.equal(initial.chains.find(chain => chain.id === '4')?.distanceMiles, 1)
    assert.equal(initial.chains.find(chain => chain.id === '4')?.cycles?.[0].distanceMiles, 0)
    assert.equal(initial.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 2)
    assert.equal(initial.wheels.find(wheel => wheel.type === 'old front')?.distanceMiles, 1)
    assert.equal(initial.wheels.find(wheel => wheel.type === 'new front')?.distanceMiles, 2)
    assert.equal(initial.wheels.find(wheel => wheel.type === 'old rear')?.distanceMiles, 1)
    assert.equal(initial.wheels.find(wheel => wheel.type === 'new rear')?.distanceMiles, 0)
    assert.equal(initial.wheels.find(wheel => wheel.type === 'rear tube')?.distanceMiles, 3)
    assert.equal(initial.services[0].distanceMiles, 1)
    assert.ok(initial.components.every(component => component.bikeId === 'b18037537'))
    assert.ok(initial.chains.every(chain => chain.bikeId === 'b18037537'))
    assert.ok(initial.wheels.every(wheel => wheel.bikeId === 'b18037537'))
    assert.equal(initial.services[0].bikeId, 'b18037537')
    assert.ok(first.includes('distance: 0.00 # fresh wax, no rides yet'))
    assert.ok(first.includes('distance: 123.456 # unrelated distance'))
    assert.equal(first.split('---')[2], fixture.split('---')[2])
    assert.ok(first.startsWith("---\ntitle: 'maintenance fixture' # preserve this comment\n"))
    assert.equal(await refresh(), first, 'repeated sync must be idempotent')
    snapshots.push({ stage: 'initial', maintenance: initial })

    const multiBikeSource = first
      .replace(
        '  chain:',
        `    - - type: Speedmax wheelset
      - bikeId: b18595115
      - distance: 999.00
      - range: [{ start: '2026-11-10', end: null }]
  chain:
    '6':
      bikeId: b18595115
      since: '2026-11-10'
      distance: 999.00
      lubricant: null
      waxed: true
      range: [{ start: '2026-11-10', end: null }]
      cycles: [{ start: '2026-11-10', end: null, distance: 999.00 }]`,
      )
      .replace(
        '    rear:',
        `        - - type: Speedmax front tire
            reason: null
          - bikeId: b18595115
          - distance: 999.00
          - range: [{ start: '2026-11-10', end: null }]
    rear:`,
      )
      .replace(
        '  service:',
        `  service:
    speedmax:
      - date: '2026-11-11'
        bikeId: b18595115
        place: workshop
        distance: 999.00`,
      )
    await writeFile(notePath, multiBikeSource)
    const multiBike = normalized(await refresh())
    assert.equal(
      multiBike.components.find(component => component.type === 'Speedmax wheelset')?.distanceMiles,
      70,
    )
    assert.equal(
      multiBike.components.find(component => component.type === 'current wheelset')?.distanceMiles,
      2,
    )
    assert.equal(multiBike.chains.find(chain => chain.id === '6')?.distanceMiles, 70)
    assert.equal(multiBike.chains.find(chain => chain.id === '6')?.cycles?.[0].distanceMiles, 70)
    assert.equal(multiBike.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 2)
    assert.equal(multiBike.chains.find(chain => chain.id === '4')?.cycles?.[0].distanceMiles, 0)
    assert.equal(
      multiBike.wheels.find(wheel => wheel.type === 'Speedmax front tire')?.distanceMiles,
      70,
    )
    assert.equal(multiBike.wheels.find(wheel => wheel.type === 'new front')?.distanceMiles, 2)
    assert.equal(multiBike.services.find(service => service.bike === 'speedmax')?.distanceMiles, 30)
    assert.equal(multiBike.services.find(service => service.bike === 'soloist')?.distanceMiles, 1)
    snapshots.push({ stage: 'separate Soloist and Speedmax histories', maintenance: multiBike })

    await saveRides([
      ...rides.filter(activity => activity.id !== 2),
      ride(2, '2026-11-12', 2, 'Ride', 'b18595115'),
    ])
    const reassigned = normalized(await refresh())
    assert.equal(reassigned.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 0)
    assert.equal(reassigned.chains.find(chain => chain.id === '6')?.cycles?.[0].distanceMiles, 72)
    snapshots.push({ stage: 'ride reassigned to another bike', maintenance: reassigned })
    const unscoped = first.replace('      - bikeId: b18037537\n', '')
    await writeFile(notePath, unscoped)
    await assert.rejects(refresh(), /bikeId/)
    assert.equal(await readFile(notePath, 'utf8'), unscoped)
    const invalidBike = first.replace('bikeId: b18037537', 'bikeId: Soloist')
    await writeFile(notePath, invalidBike)
    await assert.rejects(refresh(), /bikeId/)
    assert.equal(await readFile(notePath, 'utf8'), invalidBike)
    await writeFile(notePath, first)

    await saveRides([...rides, ride(3, '2026-11-13', 3)])
    const added = normalized(await refresh())
    assert.ok(
      added.components
        .filter(component => component.type.startsWith('current'))
        .every(component => component.distanceMiles === 5),
    )
    assert.equal(added.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 5)
    assert.equal(added.chains.find(chain => chain.id === '4')?.cycles?.[0].distanceMiles, 0)
    snapshots.push({ stage: 'new ride', maintenance: added })

    await saveRides([rides[0], ride(2, '2026-11-12', 4)])
    const correctedSource = await refresh()
    const corrected = normalized(correctedSource)
    assert.equal(corrected.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 4)
    assert.ok(
      corrected.components
        .filter(component => component.type.startsWith('current'))
        .every(component => component.distanceMiles === 4),
    )
    snapshots.push({ stage: 'corrected and deleted rides', maintenance: corrected })

    await writeFile(
      notePath,
      correctedSource
        .replace(
          "        - { start: '2026-11-10', end: '2026-11-11' }",
          "        - { start: '2026-11-10', end: '2026-11-11' }\n        - { start: '2026-11-14', end: null }",
        )
        .replace(
          "      range: [{ start: '2026-11-12', end: null }]",
          "      range: [{ start: '2026-11-12', end: '2026-11-13' }]",
        ),
    )
    await saveRides([rides[0], ride(2, '2026-11-12', 4), ride(4, '2026-11-14', 5)])
    const rotated = normalized(await refresh())
    assert.equal(rotated.chains.find(chain => chain.id === '4')?.distanceMiles, 6)
    assert.equal(rotated.chains.find(chain => chain.id === '4')?.cycles?.[0].distanceMiles, 5)
    assert.equal(rotated.chains.find(chain => chain.id === '5')?.cycles?.[0].distanceMiles, 4)
    snapshots.push({ stage: 'backup used again', maintenance: rotated })

    const rotatedSource = await readFile(notePath, 'utf8')
    await writeFile(
      notePath,
      rotatedSource
        .replace(
          "        - start: '2026-11-12'\n          end: null\n          distance: 5.00",
          "        - start: '2026-11-12'\n          end: '2026-11-14'\n          distance: 5.00",
        )
        .replace(
          '          distance: 5.00 # fresh wax, no rides yet',
          "          distance: 5.00 # fresh wax, no rides yet\n        - { start: '2026-11-14', end: null }",
        ),
    )
    const rewaxed = normalized(await refresh())
    assert.deepEqual(
      rewaxed.chains.find(chain => chain.id === '4')?.cycles?.map(cycle => cycle.distanceMiles),
      [0, 5],
    )
    snapshots.push({ stage: 'wax change on the return-to-use date', maintenance: rewaxed })
    const ambiguousSource = rotatedSource.replace(
      "range: [{ start: '2026-11-12', end: null }]",
      "range: [{ start: '2026-11-10', end: null }]",
    )
    await writeFile(notePath, ambiguousSource)
    const ambiguous = normalized(await refresh())
    assert.ok(
      ambiguous.components
        .filter(component => component.component === 'OSPW')
        .every(component => component.distanceMiles === null),
    )
    assert.equal(ambiguous.chains.find(chain => chain.id === '4')?.cycles?.[0].distanceMiles, 5)
    snapshots.push({ stage: 'ambiguous component overlap', maintenance: ambiguous })

    const beforeInvalid = await readFile(notePath, 'utf8')
    await writeFile(cachePath, '{invalid')
    await assert.rejects(refresh(), /Strava cache.*invalid/)
    assert.equal(await readFile(notePath, 'utf8'), beforeInvalid)
    await rm(cachePath)
    assert.equal(await refresh(), beforeInvalid, 'missing cache must preserve saved values')
    await saveRides([ride(1, 'invalid', 1)])
    await assert.rejects(refresh(), /invalid.*(date|Ride)|Ride.*invalid/i)
    assert.equal(await readFile(notePath, 'utf8'), beforeInvalid)
    await saveRides(rides)
    await writeFile(notePath, '# No maintenance metadata\n')
    assert.equal(await refresh(), '# No maintenance metadata\n')

    await mkdir(artifacts, { recursive: true })
    await writeFile(join(artifacts, 'sync-stages.json'), JSON.stringify(snapshots, null, 2) + '\n')
    await writeFile(join(artifacts, 'initial.md'), first)
    await writeFile(join(artifacts, 'rotated.md'), rotatedSource)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
