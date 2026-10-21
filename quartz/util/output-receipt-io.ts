import { createHash, randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { defaultIoConcurrency, mapConcurrent } from './async-pool'
import {
  outputReceiptSchema,
  type OutputReceipt,
  type OutputReceiptTransaction,
} from './output-receipt-types'

export function outputReceiptPath(output: string): string {
  return path.join(
    path.dirname(path.resolve(output)),
    '.quartz-cache',
    `output-receipt-${path.basename(output)}.json`,
  )
}

export async function readOutputReceipt(output: string): Promise<OutputReceipt | undefined> {
  const text = await fs.readFile(outputReceiptPath(output), 'utf8').catch(() => undefined)
  if (text === undefined) return undefined
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    return undefined
  }
  const parsed = outputReceiptSchema.safeParse(value)
  if (!parsed.success || parsed.data.outputRoot !== path.resolve(output)) return undefined
  return parsed.data
}

export async function beginOutputReceipt(
  output: string,
  stage: string,
): Promise<OutputReceiptTransaction> {
  const previous = await readOutputReceipt(output)
  return {
    outputRoot: path.resolve(output),
    inputRoot: process.cwd(),
    stage,
    epoch: '',
    previous,
    owners: new Map(Object.entries(previous?.owners ?? {})),
    dependencySets: new Map(Object.entries(previous?.dependencySets ?? {})),
    contentDependencies: new WeakMap(),
    observed: new Set(),
    assets: new Map(),
  }
}

async function identity(file: string): Promise<string> {
  return fs.stat(file, { bigint: true }).then(
    stat => `${stat.ino}:${stat.size}:${stat.mtimeNs}:${stat.ctimeNs}:${stat.mode}`,
    error => {
      if (error.code === 'ENOENT') return 'missing'
      throw error
    },
  )
}

async function digest(file: string): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  return hash.digest('hex')
}

/** Commit only after the stage comparison and publication succeed. Receipts do not authorize reuse. */
export async function commitOutputReceipt(
  transaction: OutputReceiptTransaction,
  files: string[],
  changed: string[],
  removed: string[],
  preserved: readonly string[],
): Promise<void> {
  const current = new Set(files)
  const owners: OutputReceipt['owners'] = {}
  const byFile = new Map<string, string[]>()
  const dependencyKeys = new Set<string>()
  for (const [name, owner] of [...transaction.owners].sort(([a], [b]) => a.localeCompare(b))) {
    const outputs = owner.outputs.filter(file => current.has(file)).sort()
    owners[name] = {
      ...owner,
      outputs,
      dependencies: [...owner.dependencies].sort(),
      externalOutputs: [...owner.externalOutputs].sort(),
    }
    for (const key of owner.dependencies) dependencyKeys.add(key)
    for (const file of outputs) {
      const names = byFile.get(file) ?? []
      names.push(name)
      byFile.set(file, names)
    }
  }
  const entries = await mapConcurrent([...files].sort(), defaultIoConcurrency, async file => {
    const filename = path.join(transaction.outputRoot, file)
    const stat = await fs.stat(filename, { bigint: true })
    const signature = `${stat.ino}:${stat.size}:${stat.mtimeNs}:${stat.ctimeNs}:${stat.mode}`
    const previous = transaction.previous?.files[file]
    const asset = transaction.assets.get(file) ?? previous?.asset
    const sha256 = previous?.identity === signature ? previous.sha256 : await digest(filename)
    if (signature !== (await identity(filename)))
      throw new Error(`Output changed during receipt capture: ${file}`)
    return [
      file,
      {
        owners: byFile.get(file) ?? [],
        size: Number(stat.size),
        sha256,
        identity: signature,
        observed: transaction.observed.has(file),
        ...(asset ? { asset } : {}),
      },
    ] satisfies [string, OutputReceipt['files'][string]]
  })
  const dependencySets: OutputReceipt['dependencySets'] = {}
  const inputs = new Set<string>()
  for (const [, file] of entries) if (file.asset) inputs.add(file.asset.source)
  for (const key of [...dependencyKeys].sort()) {
    const set = transaction.dependencySets.get(key)
    if (!set) throw new Error(`Missing output dependency set: ${key}`)
    dependencySets[key] = set
    for (const file of set.files) inputs.add(file)
  }
  const inputIdentities = Object.fromEntries(
    await mapConcurrent(
      [...inputs].sort(),
      defaultIoConcurrency,
      async file =>
        [file, await identity(path.resolve(transaction.inputRoot, file))] satisfies [
          string,
          string,
        ],
    ),
  )
  const receipt: OutputReceipt = {
    version: 1,
    outputRoot: transaction.outputRoot,
    inputRoot: transaction.inputRoot,
    epoch: transaction.epoch,
    validation: 'full-tree-comparison',
    reusable: false,
    dependencyScope: 'conservative-build-inputs',
    dependencySets,
    inputIdentities,
    owners,
    files: Object.fromEntries(entries),
    changed: [...changed].sort(),
    removed: [...removed]
      .sort()
      .map(file => ({ path: file, owners: transaction.previous?.files[file]?.owners ?? [] })),
    unattributed: entries
      .filter(([, file]) => file.owners.length === 0 && !file.asset)
      .map(([name]) => name),
    overlaps: files.filter(file => (byFile.get(file)?.length ?? 0) > 1).sort(),
    preserved: [...preserved].sort(),
  }
  const target = outputReceiptPath(transaction.outputRoot)
  const temporary = `${target}.${randomUUID()}.tmp`
  try {
    await fs.writeFile(temporary, JSON.stringify(receipt) + '\n')
    await fs.rename(temporary, target)
  } finally {
    await fs.rm(temporary, { force: true })
  }
  console.log(
    `Output receipt: ${files.length} files, ${Object.keys(owners).length} owners, ${receipt.unattributed.length} unattributed; ${target}`,
  )
}
