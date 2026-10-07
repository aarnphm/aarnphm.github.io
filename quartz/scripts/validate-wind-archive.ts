import { readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import { validateMyWindsockArchive } from '../util/mywindsock-archive'

const directory = process.argv[2] ?? 'content/triathlon/wind'
const filenames = (await readdir(directory)).filter(name => name.endsWith('.json')).sort()
let failures = 0
for (const filename of filenames) {
  try {
    const source = await readFile(path.join(directory, filename), 'utf8')
    const record = validateMyWindsockArchive(JSON.parse(source), filename.slice(0, -5))
    console.log(
      JSON.stringify({
        file: filename,
        state: record.ingestion.state,
        bytes: Buffer.byteLength(source),
        rawSeries: Object.keys(record.browserCapture?.runtime.series ?? {}).length,
        analysisSamples: record.browserCapture?.runtime.series.time?.length ?? 0,
        menuGraphs: record.graphCapture?.menu.length ?? 0,
        graphRecords: Object.keys(record.graphCapture?.graphs ?? {}).length,
        graphStates: Object.fromEntries(
          ['captured', 'empty', 'unavailable', 'failed'].map(state => [
            state,
            Object.values(record.graphCapture?.graphs ?? {}).filter(graph => graph.state === state)
              .length,
          ]),
        ),
        pending: record.ingestion.pending,
      }),
    )
  } catch (error) {
    failures += 1
    console.error(`${filename}: ${error instanceof Error ? error.message : String(error)}`)
  }
}
console.log(
  `${filenames.length - failures}/${filenames.length} records passed structural and capture checks`,
)
process.exitCode = failures > 0 ? 1 : 0
