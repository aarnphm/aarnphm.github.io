import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  aggregateCites,
  normaliseQuery,
  parseIntelligenceQuery,
  parseReadStream,
  splitHops,
  type ReadEvent,
} from './intelligence'

const stream = (chunks: string[]) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })

describe('intelligence query', () => {
  it('needs the ? prefix', () => {
    assert.equal(parseIntelligenceQuery('forcing'), null)
    assert.equal(parseIntelligenceQuery('  ? what   is forcing'), 'what is forcing')
    assert.equal(parseIntelligenceQuery('?'), '')
  })
  it('normalises for the baked key', () => {
    assert.equal(normaliseQuery('  What is  Forcing? '), 'what is forcing')
  })
})

describe('read stream', () => {
  it('parses NDJSON across chunk boundaries', async () => {
    const events: ReadEvent[] = []
    for await (const event of parseReadStream(
      stream([
        '{"t":"the","i":0,"cites":[[1,0.6],[2,0.4]]}\n{"t":" axi',
        'om","i":1}\n{"done":true,"hops":["forcing"]}',
      ]),
    ))
      events.push(event)
    assert.equal(events.length, 3)
    assert.deepEqual(events[1], { t: ' axiom', i: 1 })
    assert.deepEqual(events[2], { done: true, hops: ['forcing'] })
  })
})

describe('cites and hops', () => {
  it('sums and normalises attention mass', () => {
    const cites = aggregateCites([
      {
        t: 'a',
        i: 0,
        cites: [
          [1, 0.5],
          [2, 0.5],
        ],
      },
      { t: 'b', i: 1, cites: [[1, 1.0]] },
    ])
    assert.deepEqual(cites, [
      [1, 0.75],
      [2, 0.25],
    ])
    assert.deepEqual(aggregateCites([{ t: 'a', i: 0 }]), [])
  })
  it('splits trailing search: lines off the body', () => {
    const { body, hops } = splitHops(
      'Forcing adds a generic filter.\nsearch: Cohen forcing\nsearch: generic extension\n',
    )
    assert.equal(body, 'Forcing adds a generic filter.')
    assert.deepEqual(hops, ['Cohen forcing', 'generic extension'])
    assert.deepEqual(splitHops('plain'), { body: 'plain', hops: [] })
  })
})
