import type { Landscape } from './landscape'
import type { Clearing } from './repoussoir'
import type { ForestSettings } from './scene-settings'
import type { TreePlacement } from './scenery'

export type ForegroundTreeInput = { seed: number; clearings: Clearing[] }
export type PreparedForegroundTree = { id: string; bottom: number; shape: string; markup: string }
export type TreeWorkerRequest = {
  landscape: Landscape
  settings: ForestSettings
  trees: TreePlacement[]
  foreground?: ForegroundTreeInput
}

export type TreeWorkerResponse = { id: string; markup: string }[]
export type TreeWorkerMessage =
  | TreeWorkerResponse
  | { kind: 'foreground'; sprites: PreparedForegroundTree[] }
