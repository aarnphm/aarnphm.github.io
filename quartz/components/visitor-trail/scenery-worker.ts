import { paintSceneryPart, type SceneryMessage, type SceneryWorkerRequest } from './scenery-parts'

self.addEventListener('message', (event: MessageEvent<SceneryWorkerRequest>) => {
  for (const part of event.data.parts) {
    const message: SceneryMessage = { part, value: paintSceneryPart(part, event.data) }
    self.postMessage(message)
  }
})
