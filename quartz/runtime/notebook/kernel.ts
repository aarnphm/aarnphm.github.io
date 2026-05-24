import type { CellId, ErrorOutput, Output } from '../../util/notebook/types'
import type { NotebookRuntimeOutput } from './types'

export type NotebookModule = {
  readonly name: string
  readonly sourcePath: string
  readonly source: string
}

export type RuntimeFileResult = {
  readonly ok: boolean
  readonly status: number
  readonly statusText: string
  readonly contentType: string
  readonly bytes?: ArrayBuffer
  readonly error?: string
}

export type RuntimeAssetRequest = {
  readonly runtimeId: string
  readonly cellId: CellId
  readonly assetId: string
  readonly url: string
}

export type RuntimeAssetResult = RuntimeFileResult & {
  readonly runtimeId: string
  readonly assetId: string
}

export type RuntimeDownload = {
  readonly runtimeId: string
  readonly cellId: CellId
  readonly filename: string
  readonly contentType: string
  readonly bytes: ArrayBuffer
}

export type KernelOutput = Output | NotebookRuntimeOutput

export type RuntimeEvent =
  | { readonly type: 'started'; readonly cellId: CellId }
  | {
      readonly type: 'stream'
      readonly cellId: CellId
      readonly name: 'stdout' | 'stderr'
      readonly text: string
    }
  | { readonly type: 'output'; readonly cellId: CellId; readonly output: KernelOutput }
  | { readonly type: 'error'; readonly cellId: CellId; readonly output: ErrorOutput }
  | {
      readonly type: 'asset'
      readonly cellId: CellId
      readonly assetId: string
      readonly url: string
    }
  | { readonly type: 'download'; readonly download: RuntimeDownload }
  | {
      readonly type: 'done'
      readonly cellId: CellId
      readonly executionCount: number | null
      readonly failed?: boolean
    }
  | { readonly type: 'interrupted'; readonly cellId: CellId }
  | { readonly type: 'status'; readonly text: string }

export type KernelInitOptions = { readonly signal: AbortSignal; readonly indexUrl?: string }

export type KernelExecuteOptions = {
  readonly modules?: readonly NotebookModule[]
  readonly debug?: boolean
}

export interface Kernel {
  readonly language: string
  init(opts: KernelInitOptions): Promise<void>
  execute(cellId: CellId, source: string, opts?: KernelExecuteOptions): AsyncIterable<RuntimeEvent>
  interrupt(): void
  reset(): Promise<void>
  dispose(): Promise<void>
}
