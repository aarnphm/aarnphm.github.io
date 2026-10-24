import { fromHtml } from 'hast-util-from-html'
import { renderToString } from 'preact-render-to-string'
import type { YouTubeEmbedSpec } from '../util/youtube'
import { TranscriptPanel } from './TranscriptPanel'

export function youtubeTranscriptEmbed(embed: YouTubeEmbedSpec) {
  return fromHtml(
    renderToString(
      <div
        class="youtube-transcript"
        data-youtube-transcript={embed.videoId}
        data-transcript-language="fr"
        data-course-watch={embed.videoId}
        data-course-watch-title="YouTube video"
        data-course-watch-start={embed.startSeconds}
        data-course-watch-params={new URL(embed.src).search.slice(1)}
      >
        <div class="course-player" data-state="idle">
          <iframe
            class="course-player-frame"
            src={embed.src}
            title="YouTube video"
            allow="encrypted-media; picture-in-picture; fullscreen"
            allowFullScreen
          />
        </div>
        <p class="course-player-line">
          <span class="course-player-status" role="status" />
        </p>
        <p class="youtube-transcript-source" role="status">
          preferred language: français · source: local authorized captions
        </p>
        <div data-transcript-panel>
          <TranscriptPanel
            hasTranscript={false}
            unavailable="transcript unavailable until captions load. You can also use captions in the YouTube player."
          />
        </div>
      </div>,
    ),
    { fragment: true },
  ).children[0]
}
