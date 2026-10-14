import type { TreeWorkerRequest, TreeWorkerResponse } from './tree-worker-types'
import { createTreePainter, treeSprite } from './scenery'

const paint = createTreePainter()
self.addEventListener('message', (event: MessageEvent<TreeWorkerRequest>) => {
  const { landscape, settings, trees } = event.data
  const response: TreeWorkerResponse = trees.map(placement => ({
    id: placement.id,
    markup: treeSprite(landscape, placement, settings, paint),
  }))
  self.postMessage(response)
})
