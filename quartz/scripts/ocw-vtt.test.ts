import { toString } from 'mdast-util-to-string'
import assert from 'node:assert/strict'
import test from 'node:test'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { formatStamp, groupCues, parseVtt, transcriptMarkdown, type VttCue } from './ocw-vtt'

// Failure list written before the parser: each case names a way a real OCW file can break it.

const cue = (start: number, end: number, text: string): VttCue => ({ start, end, text })

test('parse: BOM, CRLF, header block, NOTE/STYLE/REGION blocks and cue identifiers', () => {
  const text =
    '﻿WEBVTT\r\n\r\nNOTE made by ocw\r\n\r\nSTYLE\r\n::cue { color: red }\r\n\r\n' +
    'REGION\r\nid:r1\r\n\r\n1\r\n00:00:01.000 --> 00:00:02.500 align:middle line:90%\r\nHello\r\nthere\r\n\r\n' +
    'intro-2\r\n00:00:03.000 --> 00:00:04.000\r\nAgain\r\n'
  const { cues, skipped } = parseVtt(text)
  assert.equal(skipped, 0)
  assert.deepEqual(cues, [cue(1, 2.5, 'Hello there'), cue(3, 4, 'Again')])
})

test('parse: mm:ss timestamps, hours over 1, settings after the end time', () => {
  const { cues } = parseVtt(
    'WEBVTT\n\n01:02.250 --> 01:05.000 position:10%\nShort form\n\n01:10:00.000 --> 01:10:01.000\nLong form\n',
  )
  assert.deepEqual(cues, [cue(62.25, 65, 'Short form'), cue(4200, 4201, 'Long form')])
})

test('parse: voice, class, italic and inline-timestamp tags are stripped; entities decoded', () => {
  const { cues } = parseVtt(
    'WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Lawrence Guth>So <i>x</i> &lt; y &amp; <c.colorE5E5E5>z</c><00:00:01.500> ends&nbsp;here\n',
  )
  assert.deepEqual(cues, [cue(1, 2, 'So x < y & z ends here')])
})

test('parse: sound-only cues drop, speech with a bracket stays, empty cues drop', () => {
  const { cues } = parseVtt(
    'WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n[SQUEAKING]\n\n00:00:02.000 --> 00:00:03.000\n[RUSTLING] [CLICKING]\n\n' +
      '00:00:03.000 --> 00:00:04.000\n\n\n00:00:04.000 --> 00:00:05.000\nWe use [INAUDIBLE] here\n',
  )
  assert.deepEqual(cues, [cue(4, 5, 'We use [INAUDIBLE] here')])
})

test('parse: unsorted cues sort by start; malformed times are skipped and counted', () => {
  const { cues, skipped } = parseVtt(
    'WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nsecond\n\n00:00:01.000 --> 00:00:02.000\nfirst\n\n' +
      'garbage --> 00:00:09.000\nnope\n\n00:00:07.000 --> 00:00:06.000\nbackwards\n',
  )
  assert.equal(skipped, 2)
  assert.deepEqual(
    cues.map(c => c.text),
    ['first', 'second'],
  )
})

test('parse: text containing an arrow is not a timing line; missing header gives zero cues', () => {
  const { cues } = parseVtt('WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nso f: a --> b is a map\n')
  assert.deepEqual(cues, [cue(1, 2, 'so f: a --> b is a map')])
  assert.deepEqual(parseVtt('not a vtt file'), { cues: [], skipped: 0 })
  assert.deepEqual(parseVtt(''), { cues: [], skipped: 0 })
})

test('group: speaker label, 2 s gap, 30 s with a sentence end, 60 s cap', () => {
  const cues = [
    cue(0, 1, 'LAWRENCE GUTH: OK.'),
    cue(1, 2, 'So today.'),
    cue(2, 3, 'STUDENT: Question?'),
    cue(3, 4, 'Yes'),
    cue(6.5, 7, 'after a gap'),
    cue(7, 20, 'runs on'),
    cue(20, 38, 'and ends.'),
    cue(38, 39, 'new because 30 s and a period'),
    cue(39, 70, 'no punctuation yet'),
    cue(99.5, 100, 'past 60 s'),
  ]
  const groups = groupCues(cues)
  assert.deepEqual(
    groups.map(g => [g.start, g.text]),
    [
      [0, 'LAWRENCE GUTH: OK. So today.'],
      [2, 'STUDENT: Question? Yes'],
      [6.5, 'after a gap runs on and ends.'],
      [38, 'new because 30 s and a period no punctuation yet'],
      [99.5, 'past 60 s'],
    ],
  )
})

test('markdown: stamps link to youtu.be with whole seconds, markdown and dollars are escaped', () => {
  const cues = [
    cue(754.6, 760, 'So *this* is $x$ and [not a link] # 1. item'),
    cue(4000, 4001, 'later'),
  ]
  const md = transcriptMarkdown(cues, 'AbCdEfGhIjK', 4001)
  const lines = md.split('\n\n')
  assert.equal(lines.length, 2)
  assert.ok(lines[0].startsWith('[0:12:34](https://youtu.be/AbCdEfGhIjK?t=754) '))
  assert.ok(lines[1].startsWith('[1:06:40](https://youtu.be/AbCdEfGhIjK?t=4000) '))
  assert.ok(lines[0].includes('\\*this\\*'), lines[0])
  assert.ok(lines[0].includes('\\$x\\$'), lines[0])
  assert.ok(lines[0].includes('\\[not a link]'), lines[0])
  assert.ok(!/\n/.test(lines[0]), 'one paragraph per line')
})

test('markdown: lectures under an hour use m:ss for every stamp', () => {
  assert.equal(formatStamp(754, false), '12:34')
  assert.equal(formatStamp(754, true), '0:12:34')
  assert.equal(formatStamp(5, false), '0:05')
  const md = transcriptMarkdown([cue(5, 6, 'a'), cue(3000, 3001, 'b')], 'id', 3001)
  assert.ok(md.startsWith('[0:05]('))
  assert.ok(md.includes('\n\n[50:00]('))
})

test('transcript serialization preserves literal markup and backslash-dollar sequences', () => {
  const values = [
    '\\$x$',
    '\\\\$x$',
    '<script>literal</script>',
    '\\<img src=x>',
    '$$literal$$',
    'a | b',
  ]
  for (const value of values) {
    const markdown = transcriptMarkdown([cue(1, 2, value)], 'fixture', 2)
    const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(markdown)
    assert.equal(toString(tree), `0:01 ${value}`)
    assert.equal(tree.children[0]?.type, 'paragraph')
  }
})

test('markdown: empty input gives an empty string', () => {
  assert.equal(transcriptMarkdown([], 'id', 0), '')
})
