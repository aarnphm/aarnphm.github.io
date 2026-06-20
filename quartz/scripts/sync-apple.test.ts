import assert from 'node:assert/strict'
import test from 'node:test'
import { appleImportCandidates, expandAppleHealthPath } from './sync-apple'

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
