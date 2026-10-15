import type { PlannedTrail, trailNavigationInput } from './obstacle-route'

export type NavigationWorkerRequest = (ReturnType<typeof trailNavigationInput> & { id: string })[]
export type NavigationWorkerResponse = { id: string; planned: PlannedTrail | null }
