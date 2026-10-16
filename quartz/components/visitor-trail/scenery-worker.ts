import {
  paintSceneryPart,
  sceneryOrder,
  type SceneryMessage,
  type SceneryRequest,
} from './scenery-parts'

self.addEventListener('message', (event: MessageEvent<SceneryRequest>) => {
  for (const part of sceneryOrder) {
    const message = { part, value: paintSceneryPart(part, event.data) } as SceneryMessage
    self.postMessage(message)
  }
})
