import { render } from 'preact'
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

export async function loadYouTubeTranscript(root: HTMLElement, signal: AbortSignal) {
  const panel = root.querySelector<HTMLElement>('[data-transcript-panel]')
  const provenance = root.querySelector<HTMLElement>('.youtube-transcript-source')
  const video = root.dataset.youtubeTranscript
  if (!panel || !provenance || !video) return
  const unavailable = (message: string) => {
    render(<TranscriptPanel hasTranscript={false} unavailable={message} />, panel)
  }
  // Register before awaiting: navigation must also clean up an unfinished request.
  window.addCleanup(() => render(null, panel))
  provenance.textContent = 'loading captions · preferred language: français'
  try {
    const response = await fetch(`/transcripts/${encodeURIComponent(video)}.json`, { signal })
    if (!response.ok) throw new Error('manifest unavailable')
    const manifest: unknown = await response.json()
    const available = tracks(manifest)
    const track = available.find(track => /^fr(?:-|$)/.test(track.language)) ?? available[0]
    if (!track) throw new Error('no valid tracks')
    const captions = await fetch(track.url, { signal })
    if (!captions.ok) throw new Error('captions unavailable')
    const parsed = parseVtt(await captions.text())
    if (parsed.cues.length === 0 || parsed.skipped > 0) throw new Error('captions invalid or empty')
    if (signal.aborted) return
    const paragraphs = groupCues(parsed.cues)
    root.dataset.transcriptAttribution = `language: ${track.language} · source: ${track.source}`
    root.dataset.courseWatchSource = `https://www.youtube.com/watch?v=${video}`
    root.dataset.transcriptLanguage = track.language
    root.dataset.courseWatchCaptions = track.url
    provenance.textContent = `language: ${track.language} · source: ${track.source}`
    render(
      <TranscriptPanel hasTranscript>
        {paragraphs.map(paragraph => (
          <p lang={track.language}>
            <a href={`https://www.youtube.com/watch?v=${video}&t=${Math.floor(paragraph.start)}`}>
              {formatStamp(paragraph.start, (paragraphs.at(-1)?.start ?? 0) >= 3600)}
            </a>{' '}
            {paragraph.text}
          </p>
        ))}
      </TranscriptPanel>,
      panel,
    )
  } catch {
    if (signal.aborted) return
    provenance.textContent = 'preferred language: français · source: no readable local captions'
    unavailable(
      'transcript unavailable: no readable authorized captions were supplied. Use captions in the YouTube player or supply a local WebVTT track.',
    )
  }
  if (!signal.aborted) mountTranscriptPlayer(root, signal)
}
