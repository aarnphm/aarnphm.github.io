import type { Cover } from './base-bookcase-artwork'
import type { BookcaseScene } from './base-bookcase/scene'
import { installCovers, mountCovers } from './base-bookcase/artwork'
import { mountCamera } from './base-bookcase/camera'
import { bindIndex } from './base-bookcase/index-dialog'
import { installLift, mountLift } from './base-bookcase/lift'
import { mountSelection } from './base-bookcase/selection'
import { mountArrangement, mountShelves } from './base-bookcase/shelves'
import { bindTopics } from './base-bookcase/topics'
import { bindViewport } from './base-bookcase/viewport'

export function initializeBookcase(root: HTMLElement, covers: Map<string, Cover>): void {
  if (root.dataset.bookcaseReady !== undefined) return
  const scene = root.querySelector<SVGSVGElement>('svg.bookcase-scene')
  const world = root.querySelector<HTMLElement>('.bookcase-world')
  const viewport = root.querySelector<HTMLElement>('.bookcase-viewport')
  const indexPane = root.querySelector<HTMLElement>('.bookcase-index')
  const dialog = root.querySelector<HTMLDialogElement>('.bookcase-preview')
  if (!scene || !world || !viewport || !indexPane || !dialog) return
  mountBookcase(root, covers, scene, world, viewport, indexPane, dialog)
}

function mountBookcase(
  root: HTMLElement,
  covers: Map<string, Cover>,
  scene: SVGSVGElement,
  world: HTMLElement,
  viewport: HTMLElement,
  indexPane: HTMLElement,
  dialog: HTMLDialogElement,
) {
  const s = { root, covers, scene, world, viewport, indexPane, dialog } as BookcaseScene
  installCovers(s)
  installLift(s)
  mountShelves(s)
  mountArrangement(s)
  mountCamera(s)
  mountCovers(s)
  mountLift(s)
  mountSelection(s)
  bindViewport(s)
  bindIndex(s)
  bindTopics(s)
}

export default ''
