import type { ComponentChildren } from 'preact'

const Icon = ({ children, name }: { children: ComponentChildren; name?: string }) => (
  <svg
    viewBox="0 0 24 24"
    width="14"
    height="14"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    data-glyph={name}
  >
    {children}
  </svg>
)

export function TranscriptPanel({
  hasTranscript,
  children,
  id,
  headingControls,
  paneId,
  labelledBy,
  unavailable = 'no transcript: the captions file was not vendored.',
}: {
  hasTranscript: boolean
  children?: ComponentChildren
  id?: string
  headingControls?: ComponentChildren
  paneId?: string
  labelledBy?: string
  unavailable?: string
}) {
  return (
    <section id={id} class="course-transcript" aria-label="transcript">
      <div class="course-transcript-tools" role="toolbar" aria-label="transcript">
        {headingControls ? (
          <div class="course-transcript-heading">
            <h2>transcript</h2>
            {headingControls}
          </div>
        ) : (
          <h2>transcript</h2>
        )}
        {hasTranscript ? (
          <>
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
            <div
              class="course-segmented course-transcript-actions"
              role="group"
              aria-label="transcript"
            >
              <label class="course-transcript-find" hidden>
                <span class="visually-hidden">search the transcript</span>
                <input
                  type="search"
                  class="course-transcript-search"
                  id={paneId ? `${paneId}-search` : undefined}
                  placeholder="search"
                  autocomplete="off"
                  spellcheck={false}
                />
              </label>
              <button
                type="button"
                class="course-icon-button"
                data-transcript-find
                aria-expanded="false"
                aria-label="search the transcript"
                title="search the transcript"
              >
                <Icon>
                  <circle cx="11" cy="11" r="7" />
                  <path d="m20 20-4-4" />
                </Icon>
              </button>
              <button
                type="button"
                class="course-icon-button"
                data-transcript-follow
                aria-pressed="true"
                aria-label="follow the video"
                title="keep the current paragraph in view"
              >
                <Icon>
                  <circle cx="12" cy="12" r="7" />
                  <circle cx="12" cy="12" r="2" />
                  <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
                </Icon>
              </button>
              <button
                type="button"
                class="course-icon-button"
                data-transcript-copy
                aria-label="copy the transcript"
                title="copy the transcript as text"
              >
                <Icon name="copy">
                  <rect x="8" y="8" width="13" height="13" rx="1" />
                  <path d="M16 8V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h4" />
                </Icon>
                <Icon name="done">
                  <path d="m5 12 5 5 9-10" />
                </Icon>
              </button>
            </div>
          </>
        ) : null}
      </div>
      {hasTranscript ? (
        <div
          class="course-transcript-pane"
          id={paneId}
          role={labelledBy ? 'tabpanel' : undefined}
          aria-labelledby={labelledBy}
          tabIndex={labelledBy ? 0 : undefined}
        >
          <div class="course-transcript-list">{children}</div>
        </div>
      ) : (
        <p class="course-count">{unavailable}</p>
      )}
    </section>
  )
}
