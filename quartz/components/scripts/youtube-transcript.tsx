import { render } from 'preact'
import { getFullSlug } from '../../util/path'
import { groupCues, parseVtt } from '../../util/vtt'
import { TranscriptPanel } from '../TranscriptPanel'
import { mountTranscriptPlayer } from './transcript-player'
import { formatStamp } from './transcript-text'

interface CaptionTrack {
  language: string
  source: string
  url: string
}

function tracks(value: unknown): CaptionTrack[] {
  if (!Array.isArray(value)) return []
  return value.filter(
    (track): track is CaptionTrack =>
      typeof track === 'object' &&
      track !== null &&
      typeof track.language === 'string' &&
      /^[a-z]{2,3}(?:-[a-zA-Z0-9]+)*$/.test(track.language) &&
      typeof track.source === 'string' &&
      track.source.trim().length > 0 &&
      typeof track.url === 'string' &&
      track.url.startsWith('/') &&
      !track.url.startsWith('//') &&
      new URL(track.url, location.origin).origin === location.origin,
  )
}

async function readTrack(track: CaptionTrack, signal: AbortSignal) {
  try {
    const response = await fetch(track.url, { signal })
    if (!response.ok) return
    const parsed = parseVtt(await response.text())
    if (parsed.cues.length === 0 || parsed.skipped > 0) return
    return { ...track, paragraphs: groupCues(parsed.cues) }
  } catch {
    return
  }
}

async function readManifest(video: string, signal: AbortSignal): Promise<CaptionTrack[]> {
  try {
    const response = await fetch(`/transcripts/${encodeURIComponent(video)}.json`, { signal })
    if (!response.ok) return []
    const manifest: unknown = await response.json()
    return tracks(manifest)
  } catch {
    return []
  }
}

export async function loadYouTubeTranscript(root: HTMLElement, signal: AbortSignal) {
  const panel = root.querySelector<HTMLElement>('[data-transcript-panel]')
  const video = root.dataset.youtubeTranscript
  if (!panel || !video || signal.aborted) return
  let player: AbortController | undefined
  signal.addEventListener(
    'abort',
    () => {
      player?.abort()
      render(null, panel)
    },
    { once: true },
  )
  const slug = root.closest<HTMLElement>('[data-slug]')?.dataset.slug ?? getFullSlug(window)
  const directory = slug.slice(0, slug.lastIndexOf('/') + 1)
  const base = `/${directory}${encodeURIComponent(video)}`
  const [siblings, manifest] = await Promise.all([
    Promise.all(
      ['fr', 'en'].map(language =>
        readTrack({ language, source: 'local captions', url: `${base}.${language}.vtt` }, signal),
      ),
    ),
    readManifest(video, signal),
  ])
  if (signal.aborted) return
  const available = siblings.filter(track => track !== undefined)
  if (!available.some(track => track.language === 'fr')) {
    const legacy = await readTrack(
      { language: 'fr', source: 'local captions', url: `${base}.vtt` },
      signal,
    )
    if (legacy) available.unshift(legacy)
  }
  const additional = await Promise.all(
    manifest
      .filter(track => !available.some(local => local.language === track.language))
      .map(track => readTrack(track, signal)),
  )
  for (const track of additional) {
    if (track && !available.some(local => local.language === track.language)) available.push(track)
  }
  if (signal.aborted) return
  if (available.length === 0) {
    render(
      <TranscriptPanel
        hasTranscript={false}
        unavailable="transcript unavailable here. use captions in the video player."
      />,
      panel,
    )
    mountTranscriptPlayer(root, signal)
    return
  }
  const panelId = `transcript-${root.dataset.transcriptInstance ?? video}`
  let selected = -1
  const select = (index: number, focus = false) => {
    const track = available[index]
    if (!track || signal.aborted || index === selected) return
    const reusePlayer = selected !== -1
    const follow =
      root.querySelector('[data-transcript-follow]')?.getAttribute('aria-pressed') !== 'false'
    player?.abort()
    selected = index
    root.dataset.transcriptAttribution = `language: ${track.language} · source: ${track.source}`
    root.dataset.courseWatchSource = `https://www.youtube.com/watch?v=${video}`
    root.dataset.transcriptLanguage = track.language
    root.dataset.courseWatchCaptions = track.url
    // The controller mutates cue spans and controls; give each language fresh panel DOM.
    render(
      <TranscriptPanel
        key={track.language}
        hasTranscript
        paneId={panelId}
        labelledBy={available.length > 1 ? `${panelId}-${track.language}` : undefined}
        headingControls={
          available.length > 1 ? (
            <div
              class="course-transcript-languages"
              role="tablist"
              aria-label="transcript language"
            >
              {available.map((option, optionIndex) => (
                <button
                  type="button"
                  role="tab"
                  id={`${panelId}-${option.language}`}
                  aria-controls={panelId}
                  aria-selected={optionIndex === index}
                  tabIndex={optionIndex === index ? 0 : -1}
                  lang={option.language}
                  onClick={() => select(optionIndex, true)}
                  onKeyDown={event => {
                    const next =
                      event.key === 'Home'
                        ? 0
                        : event.key === 'End'
                          ? available.length - 1
                          : event.key === 'ArrowRight'
                            ? (optionIndex + 1) % available.length
                            : event.key === 'ArrowLeft'
                              ? (optionIndex + available.length - 1) % available.length
                              : undefined
                    if (next === undefined) return
                    event.preventDefault()
                    event.stopPropagation()
                    select(next, true)
                  }}
                >
                  {option.language === 'fr'
                    ? 'français'
                    : option.language === 'en'
                      ? 'english'
                      : option.language}
                </button>
              ))}
            </div>
          ) : undefined
        }
      >
        {track.paragraphs.map(paragraph => (
          <p lang={track.language}>
            <a href={`https://www.youtube.com/watch?v=${video}&t=${Math.floor(paragraph.start)}`}>
              {formatStamp(paragraph.start, (track.paragraphs.at(-1)?.start ?? 0) >= 3600)}
            </a>{' '}
            {paragraph.text}
          </p>
        ))}
      </TranscriptPanel>,
      panel,
    )
    const followButton = root.querySelector('[data-transcript-follow]')
    followButton?.setAttribute('aria-pressed', String(follow))
    player = new AbortController()
    mountTranscriptPlayer(root, player.signal, { reusePlayer })
    if (focus) document.getElementById(`${panelId}-${track.language}`)?.focus()
  }
  const french = available.findIndex(track => /^fr(?:-|$)/.test(track.language))
  select(french >= 0 ? french : 0)
}
