import type { Root } from 'hast'
import { JSX } from 'preact'
import { QuartzComponentProps } from '../types/component'
import { ResourceView, courseFileSlug, lecturerName } from '../util/course'
import { courseStudyEndpoint } from '../util/course-study'
import { renderDescriptionHtml } from '../util/description'
import { htmlToJsx } from '../util/jsx'
import { ResourceHead, ocwLink } from './CourseResource'
import { TranscriptPanel } from './TranscriptPanel'

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

/**
 * Watch view (`kind: video`): head, the lecture summary, the youtube-nocookie player (mounted on
 * load by the script, playback on demand), then the transcript with its toolbar. The body is a
 * viewport-high grid, so the transcript scrolls under a player that stays in view. Scripted by
 * `scripts/course-watch.inline.ts`.
 */
export function CourseWatch({
  rv,
  props,
}: {
  rv: ResourceView
  props: QuartzComponentProps
}): JSX.Element {
  const { fileData, tree } = props
  const from = fileData.slug!
  const { ref, view } = rv
  const youtube = ref.record.youtube ?? ''
  const ocw = ocwLink(tree as Root) ?? view?.record.source
  const description = str(fileData.frontmatter?.description)
  const title = str(fileData.frontmatter?.title)
  const lecturer = lecturerName(view?.record)
  const hasTranscript = ref.record.cues !== undefined && ref.record.cues > 0
  const captions =
    hasTranscript && ref.record.captions && view
      ? `/${courseFileSlug(view.dir, ref.record.captions)}`
      : undefined
  const content = htmlToJsx(fileData.filePath!, tree)
  return (
    <div
      class="course-watch"
      data-course-watch={youtube}
      data-transcript-attribution="mit opencourseware, cc by-nc-sa 4.0"
      data-course-watch-title={title}
      data-course-watch-source={ocw}
      data-course-watch-captions={captions}
    >
      <ResourceHead rv={rv} from={from} ocw={ocw} endpoint={courseStudyEndpoint(props.ctx.argv)} />
      <div class="course-watch-body">
        {lecturer || description ? (
          <div class="course-watch-meta">
            {lecturer ? <p class="course-count">{lecturer}</p> : null}
            {description ? (
              <p
                class="course-watch-desc"
                dangerouslySetInnerHTML={{ __html: renderDescriptionHtml(description, from) }}
              />
            ) : null}
          </div>
        ) : null}
        <div class="course-watch-player">
          <div class="course-player" data-state="idle">
            <noscript>
              <iframe
                class="course-player-frame"
                src={`https://www.youtube-nocookie.com/embed/${encodeURIComponent(youtube)}?rel=0`}
                title={title || 'lecture video'}
                allow="encrypted-media; picture-in-picture; fullscreen"
                allowFullScreen
              />
            </noscript>
          </div>
          <p class="course-player-line course-count">
            <span class="course-player-status" role="status" aria-live="polite" />
          </p>
        </div>
        <TranscriptPanel id="transcript" hasTranscript={hasTranscript}>
          {content}
        </TranscriptPanel>
      </div>
    </div>
  )
}
