import assert from 'node:assert'
import test, { describe } from 'node:test'
import { buildYouTubeEmbed } from './youtube'

describe('youtube embeds', () => {
  test('accepts youtube hosts and rejects hostname smuggling', () => {
    assert.strictEqual(
      buildYouTubeEmbed('https://www.youtube.com/watch?v=dQw4w9WgXcQ')?.src,
      'https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ',
    )
    assert.strictEqual(buildYouTubeEmbed('https://evil-youtube.com/watch?v=dQw4w9WgXcQ'), undefined)
    assert.strictEqual(
      buildYouTubeEmbed('https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ'),
      undefined,
    )
  })
})

test('transcripts is an exact opt-in fragment and never part of the video identity', () => {
  const embed = buildYouTubeEmbed('https://www.youtube.com/watch?v=ebx7itIvobo&t=35#transcripts')
  assert.equal(embed?.videoId, 'ebx7itIvobo')
  assert.equal(embed?.transcripts, true)
  assert.equal(embed?.src, 'https://www.youtube-nocookie.com/embed/ebx7itIvobo?start=35')
  assert.equal(buildYouTubeEmbed('https://youtu.be/ebx7itIvobo')?.transcripts, false)
  assert.equal(
    buildYouTubeEmbed('https://youtu.be/ebx7itIvobo#transcripts-other')?.transcripts,
    false,
  )
  assert.equal(
    buildYouTubeEmbed('https://youtube.com/playlist?list=PL123#transcripts')?.transcripts,
    false,
  )
})
