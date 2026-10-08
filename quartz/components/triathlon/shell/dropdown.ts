import { setupDropdown as setupSharedDropdown } from '../../controls/dropdown'

// The side panel covers the page, so its header panels get a close button there.
export const setupDropdown = (
  root: HTMLElement,
  wrapSel: string,
  btnSel: string,
  panelSel: string,
  openClass: string,
): (() => void) | null => {
  const trigger = root.querySelector<HTMLButtonElement>(btnSel)
  const wrap = root.querySelector<HTMLElement>(wrapSel)
  const panel = root.querySelector<HTMLElement>(panelSel)
  if (!trigger || !wrap || !panel) return null
  const inSidePanel = root.closest('.sidepanel-container') && !root.dataset.triView
  return setupSharedDropdown({
    wrap,
    trigger,
    panel,
    openClass,
    closeButton: inSidePanel
      ? {
          className: 'tri-dropdown-close',
          container: panel.querySelector<HTMLElement>('.tri-gear-scroll') ?? panel,
        }
      : undefined,
  })
}
