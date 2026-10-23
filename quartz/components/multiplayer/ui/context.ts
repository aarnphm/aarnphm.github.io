import type { MultiplayerComment } from '../model'
import type { MultiplayerEvent, MultiplayerModel } from '../state'

export type UiDeps = {
  getState: () => MultiplayerModel
  dispatch: (event: MultiplayerEvent) => void
  canResolveComment?: (comment: MultiplayerComment) => boolean
}

// Shared mutable state of one comments UI. Every part reads and writes it through this object.
export type UiContext = UiDeps & {
  activeComposer: HTMLElement | null
  activeModal: HTMLElement | null
  activeActionsPopover: HTMLElement | null
  selectionHighlightLayer: HTMLElement | null
  activeModalCleanup: (() => void) | null
  activeComposerCleanup: (() => void) | null
  activePopoverCleanup: (() => void) | null
  threadOpenTimer: number
  composerVersion: number
  rendering: boolean
  renderAgain: boolean
  editCleanups: Set<() => void>
  renderedThreads: WeakMap<HTMLElement, MultiplayerComment[]>
}

export function createUiContext(deps: UiDeps): UiContext {
  return {
    ...deps,
    activeComposer: null,
    activeModal: null,
    activeActionsPopover: null,
    selectionHighlightLayer: null,
    activeModalCleanup: null,
    activeComposerCleanup: null,
    activePopoverCleanup: null,
    threadOpenTimer: 0,
    composerVersion: 0,
    rendering: false,
    renderAgain: false,
    editCleanups: new Set(),
    renderedThreads: new WeakMap(),
  }
}
