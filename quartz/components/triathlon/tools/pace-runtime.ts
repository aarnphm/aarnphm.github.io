import type { PaceForecaster } from '../../../util/pace-forecast'

export interface PaceRuntime {
  forecaster: PaceForecaster | null
  unavailable: boolean
  sequence: number
}

export const createPaceRuntime = (): PaceRuntime => ({
  forecaster: null,
  unavailable: false,
  sequence: 0,
})
