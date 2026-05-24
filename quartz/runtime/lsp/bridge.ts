import type { CellId } from '../../util/notebook/types'

export type LspCell = {
  readonly id: string
  readonly source: string
  readonly language: string
  readonly executionIndex: number | null
}

export type LspConfig = {
  readonly enabled: boolean
  readonly runtimeId: string
  readonly sourcePath: string
  readonly cellId: CellId | string
  readonly language: string
  readonly cells?: () => readonly LspCell[]
}

export interface LspBridge {
  extensions(config: LspConfig): Promise<readonly import('@codemirror/state').Extension[]>
}
