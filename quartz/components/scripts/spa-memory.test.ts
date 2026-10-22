import assert from 'node:assert/strict'
import test from 'node:test'
import {
  clearAllPopovers,
  evictInactivePopovers,
  MAX_CACHED_POPOVERS,
  removePopoverElement,
  touchPopoverElement,
} from './popover/dom'
import { beginSidePanelRequest, disposeSidePanel } from './side-panel'

interface MockPopoverNode {
  id: string
  active: boolean
  hovered: boolean
  removed: boolean
  classList: { contains(cls: string): boolean; remove(cls: string): void; add(cls: string): void }
  matches(selector: string): boolean
  contains(other: unknown): boolean
  remove(): void
}

function createMockPopover(id: string, active = false, hovered = false): MockPopoverNode {
  const node: MockPopoverNode = {
    id,
    active,
    hovered,
    removed: false,
    classList: {
      contains(cls: string) {
        return cls === 'active-popover' ? node.active : cls === 'popover'
      },
      remove(cls: string) {
        if (cls === 'active-popover') node.active = false
      },
      add(cls: string) {
        if (cls === 'active-popover') node.active = true
      },
    },
    matches(selector: string) {
      if (selector === ':hover') return node.hovered
      return false
    },
    contains() {
      return false
    },
    remove() {
      node.removed = true
    },
  }
  return node
}

test('popover LRU cache evicts oldest inactive popovers above MAX_CACHED_POPOVERS and runs hydrated cleanup', t => {
  const prevWindow = (globalThis as { window?: unknown }).window
  const prevDocument = (globalThis as { document?: unknown }).document
  const cleanedPdfRoots: unknown[] = []
  const popovers: MockPopoverNode[] = []

  ;(globalThis as { window?: unknown }).window = {
    quartzPdfEmbeds: {
      cleanup(root: unknown) {
        cleanedPdfRoots.push(root)
      },
    },
  }
  ;(globalThis as { document?: unknown }).document = {
    activeElement: null,
    querySelectorAll(selector: string) {
      if (selector === '.popover') return popovers.filter(p => !p.removed)
      return []
    },
  }

  t.after(() => {
    clearAllPopovers()
    ;(globalThis as { window?: unknown }).window = prevWindow
    ;(globalThis as { document?: unknown }).document = prevDocument
  })

  // Create MAX_CACHED_POPOVERS + 3 popovers:
  // popover-0 is active, popover-1 is hovered, popover-2 is touched recently, so popover-3 and popover-4 should evict first
  for (let i = 0; i < MAX_CACHED_POPOVERS + 3; i++) {
    const p = createMockPopover(`popover-${i}`, i === 0, i === 1)
    popovers.push(p)
    touchPopoverElement(p as unknown as HTMLElement)
  }

  // Touch popover-2 so it becomes newer than popover-3 and popover-4
  touchPopoverElement(popovers[2] as unknown as HTMLElement)

  evictInactivePopovers()

  // Active (popover-0) and hovered (popover-1) are preserved; remaining inactive count must be <= MAX_CACHED_POPOVERS
  assert.equal(popovers[0].removed, false, 'active popover must not be evicted')
  assert.equal(popovers[1].removed, false, 'hovered popover must not be evicted')
  assert.equal(popovers[2].removed, false, 'recently touched popover-2 must not be evicted')
  assert.equal(popovers[3].removed, true, 'oldest inactive popover-3 must be evicted')
  assert.equal(
    cleanedPdfRoots.includes(popovers[3]),
    true,
    'evicted popover must run PDF/hydrated root cleanup before removal',
  )

  const remainingInactive = popovers.filter(p => !p.removed && !p.active && !p.hovered)
  assert.ok(
    remainingInactive.length <= MAX_CACHED_POPOVERS,
    `expected at most ${MAX_CACHED_POPOVERS} inactive popovers, got ${remainingInactive.length}`,
  )
})

test('clearAllPopovers removes every popover and cleans up hydrated PDF embeds on navigation teardown', t => {
  const prevWindow = (globalThis as { window?: unknown }).window
  const prevDocument = (globalThis as { document?: unknown }).document
  const cleanedPdfRoots: unknown[] = []
  const popovers = [
    createMockPopover('popover-a', true, false),
    createMockPopover('popover-b', false, false),
    createMockPopover('popover-c', false, true),
  ]

  ;(globalThis as { window?: unknown }).window = {
    quartzPdfEmbeds: {
      cleanup(root: unknown) {
        cleanedPdfRoots.push(root)
      },
    },
  }
  ;(globalThis as { document?: unknown }).document = {
    activeElement: null,
    querySelectorAll(selector: string) {
      if (selector === '.popover') return popovers.filter(p => !p.removed)
      return []
    },
  }

  t.after(() => {
    clearAllPopovers()
    ;(globalThis as { window?: unknown }).window = prevWindow
    ;(globalThis as { document?: unknown }).document = prevDocument
  })

  for (const p of popovers) touchPopoverElement(p as unknown as HTMLElement)
  clearAllPopovers()

  assert.equal(
    popovers.every(p => p.removed),
    true,
    'all popovers must be removed from DOM on navigation cleanup',
  )
  assert.equal(
    cleanedPdfRoots.length,
    3,
    'every removed popover must invoke hydrated/PDF embed cleanup',
  )

  // Calling removePopoverElement again is idempotent
  removePopoverElement(popovers[0] as unknown as HTMLElement)
})

test('disposeSidePanel aborts pending requests and cleans up detached sidepanel elements on navigation teardown', t => {
  const prevWindow = (globalThis as { window?: unknown }).window
  const prevDocument = (globalThis as { document?: unknown }).document
  const cleanedPdfRoots: unknown[] = []

  ;(globalThis as { window?: unknown }).window = {
    quartzPdfEmbeds: {
      cleanup(root: unknown) {
        cleanedPdfRoots.push(root)
      },
    },
  }
  // Simulate document where the <aside> has already been detached by micromorph
  ;(globalThis as { document?: unknown }).document = {
    querySelector() {
      return null
    },
  }

  t.after(() => {
    clearAllPopovers()
    ;(globalThis as { window?: unknown }).window = prevWindow
    ;(globalThis as { document?: unknown }).document = prevDocument
  })

  let childrenReplaced = false
  const detachedAside = {
    dataset: {},
    classList: { remove() {}, add() {} },
    style: { removeProperty() {}, setProperty() {} },
    replaceChildren() {
      childrenReplaced = true
    },
  } as unknown as HTMLElement

  const req = beginSidePanelRequest(detachedAside)
  assert.equal(req.signal.aborted, false)

  disposeSidePanel()

  assert.equal(req.signal.aborted, true, 'detached sidepanel pending request must be aborted')
  assert.equal(childrenReplaced, true, 'detached sidepanel children must be cleared')
  assert.equal(
    cleanedPdfRoots.includes(detachedAside),
    true,
    'detached sidepanel hydrated root must be cleaned up',
  )
})
