import assert from 'node:assert'
import test, { describe } from 'node:test'
import type { NotebookVimBindingsApi } from './code-editor'
import {
  notebookLeapMotionForKey,
  notebookLeapTargets,
  notebookSurroundingPairRange,
  notebookSurroundKeyPlan,
  notebookSurroundPair,
  notebookVimNoremaps,
  notebookWordRangeAt,
  registerNotebookSurroundBindings,
} from './code-editor'
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

describe('notebook vim bindings', () => {
  test('keeps semicolon available for character-search repeat', () => {
    const repeatKey: string = ';'
    const commandModeKey: string = ':'
    assert.strictEqual(
      notebookVimNoremaps.some(
        ([context, lhs, rhs]) =>
          context === 'normal' && lhs === repeatKey && rhs === commandModeKey,
      ),
      false,
    )
  })

  test('registers surround actions around CodeMirror vim', () => {
    type RawCommand = Parameters<NotebookVimBindingsApi['_mapCommand']>[0]
    const actions: string[] = []
    const rawCommands: RawCommand[] = []
    const mapCommands: {
      keys: string
      type: string
      name: string
      args: unknown
      extra: Record<string, unknown>
    }[] = []
    const Vim: NotebookVimBindingsApi = {
      defineAction(name) {
        actions.push(name)
      },
      _mapCommand(command) {
        rawCommands.push(command)
      },
      mapCommand(keys, type, name, args, extra) {
        mapCommands.push({ keys, type, name, args, extra })
      },
    }

    registerNotebookSurroundBindings(Vim)

    assert(actions.includes('notebookSurroundSelection'))
    assert(actions.includes('notebookSurroundWord'))
    assert(actions.includes('notebookSurroundLine'))
    assert(actions.includes('notebookDeleteSurround'))
    assert(actions.includes('notebookChangeSurround'))

    const rawKeys = rawCommands.map(command => command.keys)
    assert(rawKeys.includes('S<character>'))
    assert(rawKeys.includes('ysiw<character>'))
    assert(rawKeys.includes('ysw<character>'))
    assert(rawKeys.includes('yss<character>'))
    assert(rawKeys.includes('ds<character>'))

    assert.deepStrictEqual(mapCommands.find(command => command.keys === `cs"'`)?.args, {
      oldToken: '"',
      replacementToken: "'",
    })
    assert.deepStrictEqual(mapCommands.find(command => command.keys === 'cs)]')?.args, {
      oldToken: ')',
      replacementToken: ']',
    })
  })
})

describe('notebook surround ranges', () => {
  test('resolves common delimiter aliases', () => {
    assert.deepStrictEqual(notebookSurroundPair('b'), ['(', ')'])
    assert.deepStrictEqual(notebookSurroundPair('B'), ['{', '}'])
    assert.deepStrictEqual(notebookSurroundPair('>'), ['<', '>'])
    assert.deepStrictEqual(notebookSurroundPair('"'), ['"', '"'])
  })

  test('finds the vim word under the cursor', () => {
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 2), { from: 0, to: 5 })
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 5), { from: 0, to: 5 })
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 7), { from: 6, to: 10 })
    assert.strictEqual(notebookWordRangeAt('  ', 1), undefined)
  })

  test('finds bracket and quote surrounds around the cursor', () => {
    assert.deepStrictEqual(notebookSurroundingPairRange('call(foo)', 6, ')'), {
      openFrom: 4,
      openTo: 5,
      closeFrom: 8,
      closeTo: 9,
    })
    assert.deepStrictEqual(notebookSurroundingPairRange('call(foo)', 8, '('), {
      openFrom: 4,
      openTo: 5,
      closeFrom: 8,
      closeTo: 9,
    })
    assert.deepStrictEqual(notebookSurroundingPairRange('"foo"', 2, '"'), {
      openFrom: 0,
      openTo: 1,
      closeFrom: 4,
      closeTo: 5,
    })
    assert.deepStrictEqual(notebookSurroundingPairRange('"foo"', 4, '"'), {
      openFrom: 0,
      openTo: 1,
      closeFrom: 4,
      closeTo: 5,
    })
  })
})

describe('notebook surround key planning', () => {
  test('captures operator-prefixed surround sequences before vim consumes the operator', () => {
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'y', false), {
      kind: 'pending',
      buffer: 'y',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ysi', 'w', false), {
      kind: 'pending',
      buffer: 'ysiw',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ysiw', ')', false), {
      kind: 'surroundWord',
      token: ')',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('yss', 'B', false), {
      kind: 'surroundLine',
      token: 'B',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ds', '"', false), {
      kind: 'deleteSurround',
      token: '"',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('cs)', ']', false), {
      kind: 'changeSurround',
      oldToken: ')',
      replacementToken: ']',
    })
  })

  test('captures visual surround and flushes non-surround continuations', () => {
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'S', true), { kind: 'pending', buffer: 'S' })
    assert.deepStrictEqual(notebookSurroundKeyPlan('S', '}', true), {
      kind: 'surroundSelection',
      token: '}',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ys', 'x', false), { kind: 'flush' })
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'w', false), { kind: 'pass' })
  })
})

describe('notebook leap char motions', () => {
  test('maps f/F/t/T to the local leap traversal model', () => {
    assert.deepStrictEqual(notebookLeapMotionForKey('f'), {
      key: 'f',
      backward: false,
      offset: 0,
      forwardKey: 'f',
      backwardKey: 'F',
    })
    assert.deepStrictEqual(notebookLeapMotionForKey('F'), {
      key: 'F',
      backward: true,
      offset: 0,
      forwardKey: 'f',
      backwardKey: 'F',
    })
    assert.deepStrictEqual(notebookLeapMotionForKey('t'), {
      key: 't',
      backward: false,
      offset: -1,
      forwardKey: 't',
      backwardKey: 'T',
    })
    assert.deepStrictEqual(notebookLeapMotionForKey('T'), {
      key: 'T',
      backward: true,
      offset: 1,
      forwardKey: 't',
      backwardKey: 'T',
    })
    assert.strictEqual(notebookLeapMotionForKey('x'), undefined)
  })

  test('finds exact-case visible targets in traversal order', () => {
    const motion = notebookLeapMotionForKey('f')
    assert(motion)
    assert.deepStrictEqual(notebookLeapTargets('alpha beta banana', 0, 'a', motion), [
      { matchFrom: 4, matchTo: 5, target: 4 },
      { matchFrom: 9, matchTo: 10, target: 9 },
      { matchFrom: 12, matchTo: 13, target: 12 },
      { matchFrom: 14, matchTo: 15, target: 14 },
      { matchFrom: 16, matchTo: 17, target: 16 },
    ])
    assert.deepStrictEqual(notebookLeapTargets('Alpha alpha', 0, 'a', motion), [
      { matchFrom: 4, matchTo: 5, target: 4 },
      { matchFrom: 6, matchTo: 7, target: 6 },
      { matchFrom: 10, matchTo: 11, target: 10 },
    ])
  })

  test('applies backward and till offsets', () => {
    const backward = notebookLeapMotionForKey('F')
    const forwardTill = notebookLeapMotionForKey('t')
    const backwardTill = notebookLeapMotionForKey('T')
    assert(backward)
    assert(forwardTill)
    assert(backwardTill)

    assert.deepStrictEqual(notebookLeapTargets('abacad', 5, 'a', backward), [
      { matchFrom: 4, matchTo: 5, target: 4 },
      { matchFrom: 2, matchTo: 3, target: 2 },
      { matchFrom: 0, matchTo: 1, target: 0 },
    ])
    assert.deepStrictEqual(notebookLeapTargets('abacad', 0, 'a', forwardTill), [
      { matchFrom: 2, matchTo: 3, target: 1 },
      { matchFrom: 4, matchTo: 5, target: 3 },
    ])
    assert.deepStrictEqual(notebookLeapTargets('abacad', 5, 'a', backwardTill), [
      { matchFrom: 4, matchTo: 5, target: 5 },
      { matchFrom: 2, matchTo: 3, target: 3 },
      { matchFrom: 0, matchTo: 1, target: 1 },
    ])
  })

  test('keeps leap targets inside the visible editor ranges', () => {
    const motion = notebookLeapMotionForKey('f')
    assert(motion)
    assert.deepStrictEqual(
      notebookLeapTargets('alpha beta banana', 0, 'a', motion, [
        { from: 0, to: 6 },
        { from: 11, to: 15 },
      ]),
      [
        { matchFrom: 4, matchTo: 5, target: 4 },
        { matchFrom: 12, matchTo: 13, target: 12 },
        { matchFrom: 14, matchTo: 15, target: 14 },
      ],
    )
  })
})
