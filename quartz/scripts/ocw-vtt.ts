import { toMarkdown } from 'mdast-util-to-markdown'
import { groupCues, type VttCue } from '../util/vtt'

export { groupCues, parseVtt } from '../util/vtt'
export type { VttCue, VttParagraph } from '../util/vtt'

export function formatStamp(totalSeconds: number, hours: boolean): string {
  const whole = Math.floor(totalSeconds)
  const h = Math.floor(whole / 3600)
  const m = Math.floor((whole % 3600) / 60)
  const s = whole % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(s).padStart(2, '0')
  return hours ? `${h}:${mm}:${ss}` : `${m}:${ss}`
}

function escapeParagraph(text: string): string {
  // toMarkdown escapes every construct remark would read; `$` is remark-math's, so it is escaped here.
  return toMarkdown({ type: 'paragraph', children: [{ type: 'text', value: text }] })
    .trim()
    .replace(/\n/g, ' ')
    .replace(/\$/g, '\\$')
}

/** Markdown body for a watch page: one line per paragraph, opened by a stamp that links into YouTube. */
export function transcriptMarkdown(cues: VttCue[], youtubeId: string, duration: number): string {
  const hours = duration >= 3600
  return groupCues(cues)
    .map(paragraph => {
      const t = Math.floor(paragraph.start)
      const stamp = formatStamp(paragraph.start, hours)
      return `[${stamp}](https://youtu.be/${youtubeId}?t=${t}) ${escapeParagraph(paragraph.text)}`
    })
    .join('\n\n')
}
