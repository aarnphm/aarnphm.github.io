import type { Landscape } from './landscape'
import type { ForestSettings, TreePlacement } from './scenery'

export type TreeWorkerRequest = {
  landscape: Landscape
  settings: ForestSettings
  trees: TreePlacement[]
}

export type TreeWorkerResponse = { id: string; markup: string }[]
