import type { ComponentChildren } from 'preact'

export function TranscriptPanel({
  hasTranscript,
  children,
  id,
  unavailable = 'no transcript: the captions file was not vendored.',
}: {
  hasTranscript: boolean
  children?: ComponentChildren
  id?: string
  unavailable?: string
}) {
  return (
    <section id={id} class="course-transcript" aria-label="transcript">
      <div class="course-transcript-tools" role="toolbar" aria-label="transcript">
        <h2>transcript</h2>
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
          <div class="course-transcript-list">{children}</div>
        </div>
      ) : (
        <p class="course-count">{unavailable}</p>
      )}
    </section>
  )
}
