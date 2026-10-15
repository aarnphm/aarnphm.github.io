import type { NavigationWorkerRequest, NavigationWorkerResponse } from './navigation-worker-types'
import { detourTrail } from './obstacle-route'

self.addEventListener('message', (event: MessageEvent<NavigationWorkerRequest>) => {
  for (const { id, road, radius, obstacles, bounds } of event.data) {
    const response: NavigationWorkerResponse = {
      id,
      planned: detourTrail(road, radius, obstacles, bounds),
    }
    self.postMessage(response)
  }
})
