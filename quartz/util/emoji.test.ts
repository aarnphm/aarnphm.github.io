import assert from 'node:assert'
import test, { describe } from 'node:test'
import { getIconCode } from './emoji'
import { loadEmoji } from './emoji-node'

describe('emoji assets', () => {
  test('loads emoji image payloads from sharded assets', async () => {
    const payload = await loadEmoji(getIconCode('😀'))

    assert.match(payload, /^data:image\/png;base64,/)
    assert(payload.length > 1000)
  })

  test('normalizes emoji variation selectors before lookup', async () => {
    assert.strictEqual(getIconCode('❤️'), '2764')
  })
})
