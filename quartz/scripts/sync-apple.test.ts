import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleSwim } from '../plugins/stores/apple'
import { appleImportCandidates, expandAppleHealthPath, mergeAppleSwims } from './sync-apple'

const home = '/Users/aarnphm'
const iCloudFile =
  '/Users/aarnphm/Library/Mobile Documents/iCloud~xyz~aarnphm~healthexporter/Documents/apple-health-import.json'
const cloudDocsFile =
  '/Users/aarnphm/Library/Mobile Documents/com~apple~CloudDocs/HealthExporter/apple-health-import.json'

test('expandAppleHealthPath expands shell-style home prefixes', () => {
  assert.equal(expandAppleHealthPath(`"${iCloudFile}"`, home), iCloudFile)
  assert.equal(
    expandAppleHealthPath(
      '"$HOME/Library/Mobile Documents/iCloud~xyz~aarnphm~healthexporter/Documents/apple-health-import.json"',
      home,
    ),
    iCloudFile,
  )
  assert.equal(
    expandAppleHealthPath(
      "'${HOME}/Library/Mobile Documents/iCloud~xyz~aarnphm~healthexporter/Documents/apple-health-import.json'",
      home,
    ),
    iCloudFile,
  )
  assert.equal(
    expandAppleHealthPath(
      '~/Library/Mobile Documents/iCloud~xyz~aarnphm~healthexporter/Documents/apple-health-import.json',
      home,
    ),
    iCloudFile,
  )
  assert.equal(
    expandAppleHealthPath('iCloud Drive/HealthExporter/apple-health-import.json', home),
    iCloudFile,
  )
  assert.equal(
    expandAppleHealthPath('/tmp/apple-health-import.xml', home),
    '/tmp/apple-health-import.xml',
  )
})

test('appleImportCandidates reads HealthExporter iCloud output first by default', () => {
  assert.deepEqual(appleImportCandidates('$HOME/custom.json', home), ['/Users/aarnphm/custom.json'])
  assert.deepEqual(appleImportCandidates(undefined, home), [
    iCloudFile,
    cloudDocsFile,
    'quartz/.quartz-cache/apple-health-import.json',
    'quartz/.quartz-cache/apple-health-import.xml',
  ])
})

function swim(id: string | null, date = '2026-06-19'): AppleSwim {
  return {
    id,
    date,
    start: id ? `${date}T11:00:00Z` : null,
    end: id ? `${date}T12:00:00Z` : null,
    totalM: 1500,
    laps: 60,
    activeTimeS: id ? 1800 : null,
    strokeCount: id ? 900 : null,
    strokeTimeS: id ? 1700 : null,
    strokes: { freestyle: 1500 },
  }
}

test('mergeAppleSwims replaces a legacy day row with stable session ids', () => {
  const merged = mergeAppleSwims({ '2026-06-19': swim(null) }, [swim('morning'), swim('evening')])

  assert.deepEqual(Object.keys(merged).sort(), ['evening', 'morning'])
  assert.deepEqual(merged.morning, swim('morning'))
  assert.deepEqual(merged.evening, swim('evening'))
})

test('mergeAppleSwims does not restore a legacy row after session rows exist', () => {
  assert.deepEqual(mergeAppleSwims({ morning: swim('morning') }, [swim(null)]), {
    morning: swim('morning'),
  })
})

test('mergeAppleSwims retains legacy history outside dates upgraded to session rows', () => {
  const older = swim(null, '2026-05-17')
  const legacyCurrent = swim(null, '2026-06-19')
  const morning = swim('morning')

  assert.deepEqual(
    mergeAppleSwims({ '2026-05-17': older, '2026-06-19': legacyCurrent }, [morning]),
    { '2026-05-17': older, morning },
  )
})
