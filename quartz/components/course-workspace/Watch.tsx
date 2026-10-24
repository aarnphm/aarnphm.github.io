import type { ComponentChildren } from 'preact'
import { TranscriptPanel } from '../TranscriptPanel'
import { WorkspaceSourceLinks } from './SourceLinks'

interface WatchProps {
  youtube: string
  title: string
  captions?: string
  source?: string
  hasTranscript: boolean
  children?: ComponentChildren
}

/**
 * Video tab body, server-rendered with the transcript. `data-workspace-watch` keeps the watch-page
 * script (`[data-course-watch]`) off it: the workspace mounts the player when the tab first opens.
 */
export function WorkspaceWatch({
  youtube,
  title,
  captions,
  source,
  hasTranscript,
  children,
}: WatchProps) {
  return (
    <div
      class="course-workspace-watch"
      data-workspace-watch={youtube}
      data-transcript-attribution="mit opencourseware, cc by-nc-sa 4.0"
      data-transcript-instance="workspace"
      data-course-watch-title={title}
      data-course-watch-source={source}
      data-course-watch-captions={captions}
    >
      <WorkspaceSourceLinks video={youtube} />
      <div class="course-workspace-watch-body">
        <div class="course-player" data-state="idle" />
        <p class="course-player-line course-count">
          <span class="course-player-status" role="status" aria-live="polite" />
        </p>
        <TranscriptPanel hasTranscript={hasTranscript}>{children}</TranscriptPanel>
      </div>
    </div>
  )
}
