import type { TreeWorkerRequest, TreeWorkerResponse } from './tree-worker-types'
import { createTreePainter, treeSprite } from './scenery'

const paint = createTreePainter()
self.addEventListener('message', (event: MessageEvent<TreeWorkerRequest>) => {
  const { landscape, settings, trees } = event.data
  for (const placement of trees) {
    const response: TreeWorkerResponse = [
      { id: placement.id, markup: treeSprite(landscape, placement, settings, paint) },
    ]
    self.postMessage(response)
  }
})
