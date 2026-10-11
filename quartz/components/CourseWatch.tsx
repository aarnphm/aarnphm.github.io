import type { Root } from 'hast'
import { JSX } from 'preact'
import { QuartzComponentProps } from '../types/component'
import { ResourceView, courseFileSlug, lecturerName } from '../util/course'
import { htmlToJsx } from '../util/jsx'
import { ResourceHead, ocwLink } from './CourseResource'

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
      data-course-watch-title={title}
      data-course-watch-source={ocw}
      data-course-watch-captions={captions}
    >
      <ResourceHead rv={rv} from={from} ocw={ocw} />
      <div class="course-watch-body">
        {lecturer || description ? (
          <div class="course-watch-meta">
            {lecturer ? <p class="course-count">{lecturer}</p> : null}
            {description ? <p class="course-watch-desc">{description}</p> : null}
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
        <section id="transcript" class="course-transcript" aria-labelledby="course-transcript-h">
          <div class="course-transcript-tools" role="toolbar" aria-label="transcript">
            <h2 id="course-transcript-h">transcript</h2>
            {hasTranscript ? (
              <>
                <label class="course-transcript-find">
                  <span class="visually-hidden">search the transcript</span>
                  <input
                    type="search"
                    class="course-transcript-search"
                    placeholder="search"
                    autocomplete="off"
                    spellcheck={false}
                  />
                </label>
                <output class="course-transcript-hits" aria-live="polite" />
                <div class="course-segmented" role="group" aria-label="paragraphs">
                  <button
                    type="button"
                    data-transcript-prev
                    aria-label="previous paragraph"
                    title="previous paragraph"
                  >
                    ‹ prev
                  </button>
                  <button
                    type="button"
                    data-transcript-next
                    aria-label="next paragraph"
                    title="next paragraph"
                  >
                    next ›
                  </button>
                </div>
                <button
                  type="button"
                  data-transcript-follow
                  aria-pressed="true"
                  title="keep the current paragraph in view"
                >
                  follow
                </button>
                <button type="button" data-transcript-copy title="copy the transcript as text">
                  copy
                </button>
              </>
            ) : null}
          </div>
          {hasTranscript ? (
            <div class="course-transcript-pane">
              <div class="course-transcript-list">{content}</div>
            </div>
          ) : (
            <p class="course-count">no transcript: the captions file was not vendored.</p>
          )}
        </section>
      </div>
    </div>
  )
}
