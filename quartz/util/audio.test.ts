import assert from 'node:assert/strict'
import test from 'node:test'
import { memoPeaksUrl, resampleAudioPeaks } from './audio'

test('memo waveforms use the small sidecar on either serving host', () => {
  for (const host of ['aarnphm.xyz', 'stream.aarnphm.xyz', 'localhost:8080']) {
    assert.equal(
      memoPeaksUrl(`https://${host}/triathlon/memos/20260905.m4a`),
      `https://${host}/triathlon/memos/20260905.peaks.json`,
    )
  }
  assert.equal(memoPeaksUrl('https://example.com/music.m4a'), undefined)
})

test('resamples validated waveform peaks to the visible bar count', () => {
  assert.deepEqual(resampleAudioPeaks({ peaks: [0.1, 0.5, 0, 1] }, 2), [0.5, 1])
  assert.deepEqual(resampleAudioPeaks({ peaks: [0.1, 0.5] }, 4), [0.1, 0.1, 0.5, 0.5])
  for (const value of [
    null,
    {},
    { peaks: [] },
    { peaks: [NaN] },
    { peaks: [2] },
    { peaks: ['1'] },
  ]) {
    assert.equal(resampleAudioPeaks(value, 2), null)
  }
})
