import type { NavigationWorkerRequest, NavigationWorkerResponse } from './navigation-worker-types'
import { navigationSearch } from './navigation-search'

self.addEventListener('message', (event: MessageEvent<NavigationWorkerRequest>) => {
  const search = navigationSearch(event.data.query)
  let step = search.next()
  while (!step.done) step = search.next()
  const response: NavigationWorkerResponse = { id: event.data.id, planned: step.value }
  self.postMessage(response)
})
