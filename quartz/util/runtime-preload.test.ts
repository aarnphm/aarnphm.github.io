import assert from 'node:assert/strict'
import test from 'node:test'
import { supportsEagerRuntimePreload } from './runtime-preload'

type Nav = { userAgent?: string; maxTouchPoints?: number }

function stubEnv(media: Record<string, boolean>, nav: Nav = {}) {
  const g = globalThis as Record<string, unknown>
  const winDesc = Object.getOwnPropertyDescriptor(g, 'window')
  const navDesc = Object.getOwnPropertyDescriptor(g, 'navigator')
  Object.defineProperty(g, 'window', {
    value: { matchMedia: (query: string) => ({ matches: media[query] ?? false }) },
    configurable: true,
    writable: true,
  })
  Object.defineProperty(g, 'navigator', {
    value: { userAgent: nav.userAgent ?? '', maxTouchPoints: nav.maxTouchPoints ?? 0 },
    configurable: true,
    writable: true,
  })
  return () => {
    if (winDesc) Object.defineProperty(g, 'window', winDesc)
    else delete g.window
    if (navDesc) Object.defineProperty(g, 'navigator', navDesc)
    else delete g.navigator
  }
}

const MAC_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
const IPAD_UA =
  'Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
const IPHONE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'

test('supportsEagerRuntimePreload allows fine-pointer desktops', () => {
  const restore = stubEnv({ '(pointer: fine)': true }, { userAgent: MAC_UA })
  try {
    assert.equal(supportsEagerRuntimePreload(), true)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload allows iPad reported as iPad', () => {
  const restore = stubEnv({ '(pointer: fine)': false }, { userAgent: IPAD_UA, maxTouchPoints: 5 })
  try {
    assert.equal(supportsEagerRuntimePreload(), true)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload allows iPadOS masquerading as macOS', () => {
  const restore = stubEnv({ '(pointer: fine)': false }, { userAgent: MAC_UA, maxTouchPoints: 5 })
  try {
    assert.equal(supportsEagerRuntimePreload(), true)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload skips iPhones', () => {
  const restore = stubEnv({ '(pointer: fine)': false }, { userAgent: IPHONE_UA, maxTouchPoints: 5 })
  try {
    assert.equal(supportsEagerRuntimePreload(), false)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload does not treat a touchless mac as iPad', () => {
  const restore = stubEnv({ '(pointer: fine)': false }, { userAgent: MAC_UA, maxTouchPoints: 0 })
  try {
    assert.equal(supportsEagerRuntimePreload(), false)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload respects reduced-data on every device', () => {
  const restore = stubEnv(
    { '(pointer: fine)': true, '(prefers-reduced-data: reduce)': true },
    { userAgent: IPAD_UA, maxTouchPoints: 5 },
  )
  try {
    assert.equal(supportsEagerRuntimePreload(), false)
  } finally {
    restore()
  }
})

test('supportsEagerRuntimePreload is false without a window', () => {
  const g = globalThis as Record<string, unknown>
  const winDesc = Object.getOwnPropertyDescriptor(g, 'window')
  delete g.window
  try {
    assert.equal(supportsEagerRuntimePreload(), false)
  } finally {
    if (winDesc) Object.defineProperty(g, 'window', winDesc)
  }
})
