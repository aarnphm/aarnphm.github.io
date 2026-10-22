import { cleanupHydratedRoot } from '../root-lifecycle'

export const MAX_CACHED_POPOVERS = 8

const popoverAccessOrder = new Set<HTMLElement>()

export function blurPopoverFocus(popoverElement: HTMLElement): void {
  const activeElement = document.activeElement
  if (
    activeElement &&
    typeof (activeElement as HTMLElement).blur === 'function' &&
    typeof popoverElement.contains === 'function' &&
    popoverElement.contains(activeElement as Node)
  ) {
    ;(activeElement as HTMLElement).blur()
  }
}

export function touchPopoverElement(popoverElement: HTMLElement): void {
  popoverAccessOrder.delete(popoverElement)
  popoverAccessOrder.add(popoverElement)
}

export function removePopoverElement(popoverElement: HTMLElement): void {
  popoverAccessOrder.delete(popoverElement)
  blurPopoverFocus(popoverElement)
  cleanupHydratedRoot(popoverElement)
  popoverElement.remove()
}

export function evictInactivePopovers(except?: HTMLElement): void {
  const domPopovers = new Set<HTMLElement>(
    Array.from(document.querySelectorAll<HTMLElement>('.popover')),
  )
  for (const tracked of popoverAccessOrder) {
    if (!domPopovers.has(tracked)) {
      popoverAccessOrder.delete(tracked)
    }
  }
  for (const domPopover of domPopovers) {
    if (!popoverAccessOrder.has(domPopover)) {
      popoverAccessOrder.add(domPopover)
    }
  }

  const inactive: HTMLElement[] = []
  for (const popoverElement of popoverAccessOrder) {
    if (popoverElement === except) continue
    if (popoverElement.classList.contains('active-popover')) continue
    let hovered = false
    try {
      hovered = popoverElement.matches(':hover')
    } catch {
      hovered = false
    }
    if (hovered) continue
    inactive.push(popoverElement)
  }

  while (inactive.length > MAX_CACHED_POPOVERS) {
    const oldest = inactive.shift()
    if (!oldest) break
    removePopoverElement(oldest)
  }
}

export function clearAllPopovers(): void {
  const domNodes =
    typeof document !== 'undefined' && typeof document.querySelectorAll === 'function'
      ? Array.from(document.querySelectorAll<HTMLElement>('.popover'))
      : []
  const allPopovers = new Set<HTMLElement>([...domNodes, ...popoverAccessOrder])
  popoverAccessOrder.clear()
  for (const popoverElement of allPopovers) {
    popoverElement.classList.remove('active-popover')
    removePopoverElement(popoverElement)
  }
}
