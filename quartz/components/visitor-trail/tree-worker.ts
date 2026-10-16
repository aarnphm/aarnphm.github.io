import type { TreeWorkerRequest, TreeWorkerMessage } from './tree-worker-types'
import { repoussoirSprites } from './repoussoir'
import { createTreePainter, treeSprite } from './scenery'

const paint = createTreePainter()
self.addEventListener('message', (event: MessageEvent<TreeWorkerRequest>) => {
  const { landscape, settings, trees, foreground } = event.data
  if (foreground) {
    const response: TreeWorkerMessage = {
      kind: 'foreground',
      sprites: repoussoirSprites(landscape, foreground.seed, foreground.clearings).map(
        ({ render, ...sprite }) => ({ ...sprite, markup: render() }),
      ),
    }
    self.postMessage(response)
  }
  for (const placement of trees) {
    const response: TreeWorkerMessage = [
      { id: placement.id, markup: treeSprite(landscape, placement, settings, paint) },
    ]
    self.postMessage(response)
  }
})
