import assert from 'node:assert'
import test, { describe } from 'node:test'
import { NOTEBOOK_VIM_MODE_STORAGE_KEY, readVimModeSetting, writeVimModeSetting } from './vim'

class MemoryStorage implements Storage {
  private store = new Map<string, string>()
  get length(): number {
    return this.store.size
  }
  clear(): void {
    this.store.clear()
  }
  getItem(key: string): string | null {
    return this.store.get(key) ?? null
  }
  setItem(key: string, value: string): void {
    this.store.set(key, value)
  }
  removeItem(key: string): void {
    this.store.delete(key)
  }
  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null
  }
}

describe('vim mode setting', () => {
  test('round-trips through storage', () => {
    const storage = new MemoryStorage()
    assert.strictEqual(readVimModeSetting(storage), false)
    writeVimModeSetting(true, storage)
    assert.strictEqual(storage.getItem(NOTEBOOK_VIM_MODE_STORAGE_KEY), 'true')
    assert.strictEqual(readVimModeSetting(storage), true)
    writeVimModeSetting(false, storage)
    assert.strictEqual(storage.getItem(NOTEBOOK_VIM_MODE_STORAGE_KEY), null)
    assert.strictEqual(readVimModeSetting(storage), false)
  })

  test('returns false when storage is unavailable', () => {
    assert.strictEqual(readVimModeSetting(undefined), false)
  })
})
