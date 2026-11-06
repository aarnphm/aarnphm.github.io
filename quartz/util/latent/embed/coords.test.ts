import assert from 'node:assert/strict'
import test from 'node:test'
import { decodeCoords, encodeCoords } from './coords'

test('the coordinates asset round-trips within int16 precision, with levels and heat', () => {
  const coords = Float32Array.from([0.5, -1.25, 0, 2, 0.001, -2, 1, 1, 1])
  const kinds = Uint8Array.from([0, 1, 3])
  const lvl = Uint8Array.from([2, 0, 1])
  const heat = Uint8Array.from([0, 255, 17])
  const { bytes, reach } = encodeCoords(coords, kinds, lvl, heat)
  assert.equal(reach, 2)
  assert.equal(bytes.byteLength, 3 * 6 + 3 * 3)
  const back = decodeCoords(bytes.slice(), 3, reach, { lvl: true, heat: true })
  for (let i = 0; i < coords.length; i++) assert.ok(Math.abs(back.coords[i] - coords[i]) < 1e-4)
  assert.deepEqual([...back.kinds], [0, 1, 3])
  assert.deepEqual([...back.lvl!], [2, 0, 1])
  assert.deepEqual([...back.heat!], [0, 255, 17])
  const plain = encodeCoords(coords, kinds, null, null)
  const docs = decodeCoords(plain.bytes, 3, plain.reach, { lvl: false, heat: false })
  assert.equal(docs.lvl, null)
  assert.equal(docs.heat, null)
  assert.throws(() => decodeCoords(plain.bytes, 4, plain.reach, { lvl: false, heat: false }))
})
