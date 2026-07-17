import { createHash } from 'node:crypto'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { isRecord, readNumber, readString } from '../util/type-guards'
import { latestModelArchiveEntry } from './pace-model-archive'

type ModelFamily = 'pace' | 'hr'

const MODEL_FAMILIES: readonly ModelFamily[] = ['pace', 'hr']

export interface LocalModelSelection {
  family: ModelFamily
  archive: string
  version: number
}

function atomicCopy(source: string, destination: string): void {
  const temporary = `${destination}.${process.pid}.tmp`
  copyFileSync(source, temporary)
  renameSync(temporary, destination)
}

function atomicJson(destination: string, value: unknown): void {
  const temporary = `${destination}.${process.pid}.tmp`
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`)
  renameSync(temporary, destination)
}

export function syncLatestLocalPaceModels(
  archiveRoot = path.join(process.cwd(), '.models-archive'),
  publicRoot = path.join(process.cwd(), 'public'),
): LocalModelSelection[] {
  const selections: LocalModelSelection[] = []
  for (const family of MODEL_FAMILIES) {
    const entry = latestModelArchiveEntry(path.join(archiveRoot, family))
    if (!entry) continue
    const manifestPath = path.join(entry.path, 'manifest.json')
    const weightsPath = path.join(entry.path, 'model.safetensors')
    if (!existsSync(manifestPath) || !existsSync(weightsPath)) continue

    const manifest: unknown = JSON.parse(readFileSync(manifestPath, 'utf8'))
    if (!isRecord(manifest)) throw new Error(`${family}: local manifest is not an object`)
    if (readString(manifest, 'target') !== family)
      throw new Error(`${family}: local manifest target mismatch`)
    const sha256 = readString(manifest, 'sha256')
    if (!sha256) throw new Error(`${family}: local manifest is missing sha256`)
    const actualSha256 = createHash('sha256').update(readFileSync(weightsPath)).digest('hex')
    if (actualSha256 !== sha256) throw new Error(`${family}: local safetensors sha256 mismatch`)

    const version = readNumber(manifest, 'version') ?? entry.version
    const relativeDirectory = `models/${family}/local-v${entry.version}`
    const outputDirectory = path.join(publicRoot, relativeDirectory)
    mkdirSync(outputDirectory, { recursive: true })
    atomicCopy(manifestPath, path.join(outputDirectory, 'manifest.json'))
    atomicCopy(weightsPath, path.join(outputDirectory, 'model.safetensors'))

    const familyDirectory = path.join(publicRoot, 'models', family)
    mkdirSync(familyDirectory, { recursive: true })
    atomicJson(path.join(familyDirectory, 'latest.json'), {
      version,
      manifest: `${relativeDirectory}/manifest.json`,
      weights: `${relativeDirectory}/model.safetensors`,
      sha256,
      datasetHash: readString(manifest, 'datasetHash') ?? '',
      val: isRecord(manifest.val) ? manifest.val : {},
      createdAt: readString(manifest, 'createdAt') ?? '',
    })
    selections.push({ family, archive: entry.name, version })
  }
  return selections
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : ''
if (invokedPath === fileURLToPath(import.meta.url)) {
  for (const selection of syncLatestLocalPaceModels())
    process.stdout.write(
      `${selection.family}: local v${selection.version} from ${selection.archive}\n`,
    )
}
