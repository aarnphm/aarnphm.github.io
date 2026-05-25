const notebookCellFrameSelector = '[data-notebook-cell-frame]'
const notebookActiveCellAttribute = 'data-notebook-active-cell'
const notebookActiveCellSelector = `${notebookCellFrameSelector}[${notebookActiveCellAttribute}]`

type NotebookCellRoot = Document | HTMLElement

export function notebookCellFrames(root: NotebookCellRoot): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(notebookCellFrameSelector))
}

export function activeNotebookCellFrame(root: NotebookCellRoot): HTMLElement | undefined {
  const frames = activeNotebookCellFrames(root)
  return frames.length === 0 ? undefined : frames[frames.length - 1]
}

export function notebookCellFrameFromElement(element: Element | null): HTMLElement | undefined {
  return element?.closest<HTMLElement>(notebookCellFrameSelector) ?? undefined
}

export function notebookCellFrameId(frame: HTMLElement | null | undefined): string | undefined {
  return frame?.dataset.notebookCellFrame
}

export function notebookCellFrameIsActive(frame: HTMLElement): boolean {
  return frame.hasAttribute(notebookActiveCellAttribute)
}

export function selectNotebookCellFrame(frame: HTMLElement) {
  for (const active of activeNotebookCellFrames(frame.ownerDocument)) {
    if (active !== frame) active.removeAttribute(notebookActiveCellAttribute)
  }
  frame.setAttribute(notebookActiveCellAttribute, '')
  if (!frame.hasAttribute('tabindex')) frame.tabIndex = -1
}

export function clearActiveNotebookCellFrames(root: NotebookCellRoot) {
  for (const active of activeNotebookCellFrames(root)) {
    active.removeAttribute(notebookActiveCellAttribute)
  }
}

function activeNotebookCellFrames(root: NotebookCellRoot): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(notebookActiveCellSelector))
}
