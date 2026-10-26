import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'

export interface Cue {
  start: number
  end: number
  span: HTMLElement
  paragraph: number
}

const CUE_SWEEP_MIN = 300

/** Index of the last cue starting at or before `seconds`. */
function cueAt(cues: Cue[], seconds: number): number {
  let lo = 0
  let hi = cues.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (cues[mid].start <= seconds) {
      found = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return found
}

export function createCueHighlighter(cues: Cue[], color: string, reducedMotion: boolean) {
  const marks = new Map<number, RoughAnnotation>()
  let current = -1
  let playing = false
  let lastTime = 0

  const markPaths = (cue: Cue): HTMLElement[] => {
    const svg = cue.span.nextElementSibling
    return svg instanceof SVGElement && svg.classList.contains('rough-annotation')
      ? Array.from(svg.querySelectorAll<HTMLElement>('path'))
      : []
  }

  const syncMark = (cue: Cue, seconds: number) => {
    const offset = Math.max(0, (seconds - cue.start) * 1000)
    for (const path of markPaths(cue)) {
      const base = Number(path.dataset.delay ?? 0)
      path.style.animationDelay = `${base - offset}ms`
      path.style.animationPlayState = playing ? 'running' : 'paused'
    }
  }

  const inkCue = (cue: Cue, sweep: boolean, seconds: number): RoughAnnotation => {
    const mark = annotate(cue.span, {
      type: 'underline',
      color,
      strokeWidth: 1.6,
      padding: 1,
      iterations: 1,
      multiline: true,
      animate: sweep && !reducedMotion,
      animationDuration: Math.max(CUE_SWEEP_MIN, (cue.end - cue.start) * 1000),
    })
    mark.show()
    if (sweep && !reducedMotion) {
      for (const path of markPaths(cue)) {
        path.style.animationTimingFunction = 'linear'
        path.dataset.delay = String(parseFloat(path.style.animationDelay) || 0)
      }
      syncMark(cue, seconds)
    }
    return mark
  }

  const setCue = (seconds: number) => {
    const index = cueAt(cues, seconds)
    const cue = cues[index]
    const backward = seconds < lastTime - 0.5
    lastTime = seconds
    if (index === current) {
      if (cue && backward) {
        marks.get(index)?.remove()
        marks.set(index, inkCue(cue, true, seconds))
      } else if (cue && playing) {
        syncMark(cue, seconds)
      }
      return
    }
    for (const [key, mark] of marks) {
      if (!cue || key > index || cues[key].paragraph !== cue.paragraph) {
        mark.remove()
        marks.delete(key)
      }
    }
    current = index
    if (!cue) return
    for (let k = index - 1; k >= 0 && cues[k].paragraph === cue.paragraph; k--) {
      if (!marks.has(k)) marks.set(k, inkCue(cues[k], false, seconds))
    }
    marks.set(index, inkCue(cue, true, seconds))
  }

  const setPlaying = (state: boolean) => {
    if (state === playing) return
    playing = state
    const cue = cues[current]
    if (cue) syncMark(cue, lastTime)
  }

  const dispose = () => {
    for (const mark of marks.values()) mark.remove()
    marks.clear()
  }

  return { setCue, setPlaying, dispose }
}
