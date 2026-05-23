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
