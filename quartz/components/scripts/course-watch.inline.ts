import { currentNavSignal } from './nav-lifecycle'
import { mountTranscriptPlayer } from './transcript-player'

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(
    '[data-course-watch]:not([data-youtube-transcript])',
  )) {
    mountTranscriptPlayer(root, currentNavSignal())
  }
})
