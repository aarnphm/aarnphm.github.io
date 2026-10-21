import { AsyncLocalStorage } from 'node:async_hooks'
import { createHash } from 'node:crypto'
import path from 'node:path'
import type { ProcessedContent } from '../plugins/vfile'
import type { BuildCtx } from './ctx'
import type { OutputOwner, OutputReceiptTransaction } from './output-receipt-types'
import { transclusionDependencies } from './transclusion-dependencies'

type ReceiptContext = { transaction: OutputReceiptTransaction; owner?: OutputOwner }

declare global {
  var __quartzOutputReceiptContext: AsyncLocalStorage<ReceiptContext> | undefined
}

// Emitter bundles reload while helper modules can survive. Both must share the same async scope.
const context = (globalThis.__quartzOutputReceiptContext ??=
  new AsyncLocalStorage<ReceiptContext>())

export function outputRelativePath(root: string, file: string): string | undefined {
  const relative = path.relative(root, path.resolve(file))
  if (
    !relative ||
    relative === '..' ||
    relative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relative)
  )
    return undefined
  return relative.split(path.sep).join('/')
}

function inputPath(transaction: OutputReceiptTransaction, file: string): string {
  const absolute = path.resolve(transaction.inputRoot, file)
  const staged = outputRelativePath(transaction.stage, absolute)
  const logical = staged ? path.join(transaction.outputRoot, staged) : absolute
  return path.relative(transaction.inputRoot, logical).split(path.sep).join('/')
}

function dependencies(
  transaction: OutputReceiptTransaction,
  ctx: BuildCtx,
  content: ProcessedContent[],
): string {
  const cached = transaction.contentDependencies.get(content)
  if (cached) return cached
  const files = new Set([
    'quartz.config.ts',
    'quartz.layout.ts',
    'package.json',
    'pnpm-lock.yaml',
    '.env',
    ...(globalThis.__quartzBuildSourceInputs ?? []),
    ...ctx.cfg.plugins.transformers.flatMap(plugin => plugin.fileDependencies?.(ctx) ?? []),
  ])
  for (const [, file] of content) {
    if (file.data.filePath) files.add(file.data.filePath)
    for (const dependency of file.data.codeDependencies ?? []) {
      files.add(dependency)
      if (!path.isAbsolute(dependency)) files.add(path.resolve(ctx.argv.directory, dependency))
    }
  }
  const set = {
    files: [...new Set([...files].map(file => inputPath(transaction, file)))].sort(),
    transclusions: transclusionDependencies(content),
  }
  const key = createHash('sha256').update(JSON.stringify(set)).digest('hex')
  transaction.dependencySets.set(key, set)
  transaction.contentDependencies.set(content, key)
  return key
}

export function withOutputReceipt<T>(
  transaction: OutputReceiptTransaction,
  run: () => Promise<T>,
): Promise<T> {
  return context.run({ transaction }, run)
}

export function withOutputOwner<T>(
  ctx: BuildCtx,
  content: ProcessedContent[],
  name: string,
  mode: OutputOwner['mode'],
  run: () => Promise<T>,
): Promise<T> {
  const parent = context.getStore()
  if (!parent) return run()
  const { transaction } = parent
  transaction.epoch = ctx.buildId
  const previous = transaction.owners.get(name)
  const owner: OutputOwner = {
    epoch: ctx.buildId,
    mode,
    dependencies: [
      ...new Set([
        ...(mode === 'patch' ? (previous?.dependencies ?? []) : []),
        dependencies(transaction, ctx, content),
      ]),
    ].sort(),
    outputs: mode === 'patch' ? [...(previous?.outputs ?? [])] : [],
    externalOutputs: [],
  }
  transaction.owners.set(name, owner)
  return context.run({ transaction, owner }, run)
}

export function recordOutput(file: string, observed = true, source?: string): void {
  const active = context.getStore()
  if (!active) return
  const relative = outputRelativePath(active.transaction.stage, file)
  if (!relative) {
    if (active.owner && !active.owner.externalOutputs.includes(file))
      active.owner.externalOutputs.push(file)
    return
  }
  if (observed) active.transaction.observed.add(relative)
  if (active.owner && !active.owner.outputs.includes(relative)) active.owner.outputs.push(relative)
  if (source && active.owner) {
    const sourcePath = inputPath(active.transaction, source)
    const key = createHash('sha256').update(sourcePath).digest('hex')
    active.transaction.dependencySets.set(key, { files: [sourcePath], transclusions: [] })
    if (!active.owner.dependencies.includes(key)) active.owner.dependencies.push(key)
  }
}

export function recordOutputRemoval(file: string): void {
  const active = context.getStore()
  if (!active) return
  const relative = outputRelativePath(active.transaction.stage, file)
  if (!relative) return
  for (const owner of active.transaction.owners.values()) {
    owner.outputs = owner.outputs.filter(output => output !== relative)
  }
}

export function recordAssetOutput(file: string, owner: string, source: string): void {
  const active = context.getStore()
  if (!active) return
  const relative = outputRelativePath(active.transaction.stage, file)
  if (!relative) return
  const stagedSource = outputRelativePath(active.transaction.stage, source)
  const logicalSource = stagedSource
    ? path.join(active.transaction.outputRoot, stagedSource)
    : source
  active.transaction.assets.set(relative, {
    owner,
    source: inputPath(active.transaction, logicalSource),
  })
}
