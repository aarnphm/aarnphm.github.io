/** Shorter recording gaps are traffic stops or dropped records; they keep their width. */
export const RECORDING_PAUSE_MIN_S = 60

/** Stopped interval of an activity recording, in elapsed seconds. */
export interface ElapsedPause {
  startS: number
  endS: number
}

type PausedTrace = { pauses: readonly ElapsedPause[] } | null | undefined

/** Recorded pauses of an activity, from whichever elapsed trace its sport carries. */
export const activityRecordedPauses = (activity: {
  cyclingPowerTrace?: PausedTrace
  runHeartCost?: PausedTrace
}): readonly ElapsedPause[] =>
  activity.cyclingPowerTrace?.pauses ?? activity.runHeartCost?.pauses ?? []

/** Maps elapsed time to a horizontal fraction that gives removed pauses no width. */
export interface ElapsedAxis {
  startS: number
  endS: number
  pauses: readonly ElapsedPause[]
  /** Elapsed time to a fraction in [0, 1]; a time inside a pause lands on its join. */
  fraction: (elapsedS: number) => number
  /** Fraction in [0, 1] to elapsed time; a join resolves to the end of its pause. */
  elapsedAt: (fraction: number) => number
  /** True when the span between two elapsed times overlaps a removed pause. */
  crossesPause: (leftS: number, rightS: number) => boolean
}

export const elapsedAxis = (
  startS: number,
  endS: number,
  pauses: readonly ElapsedPause[] = [],
): ElapsedAxis => {
  const inside = pauses
    .map(pause => ({ startS: Math.max(startS, pause.startS), endS: Math.min(endS, pause.endS) }))
    .filter(pause => pause.endS > pause.startS)
    .sort((left, right) => left.startS - right.startS)
  const removedS = inside.reduce((total, pause) => total + pause.endS - pause.startS, 0)
  const shownS = endS - startS - removedS
  const shownBefore = (elapsedS: number): number => {
    let removed = 0
    for (const pause of inside) {
      if (elapsedS <= pause.startS) break
      removed += Math.min(elapsedS, pause.endS) - pause.startS
    }
    return elapsedS - startS - removed
  }
  return {
    startS,
    endS,
    pauses: inside,
    fraction: elapsedS =>
      shownS > 0
        ? Math.max(0, Math.min(1, shownBefore(Math.max(startS, Math.min(endS, elapsedS))) / shownS))
        : 0,
    elapsedAt: fraction => {
      let elapsedS = startS + Math.max(0, Math.min(1, fraction)) * shownS
      for (const pause of inside)
        if (elapsedS >= pause.startS) elapsedS += pause.endS - pause.startS
        else break
      return Math.min(endS, elapsedS)
    },
    crossesPause: (leftS, rightS) =>
      inside.some(pause => leftS < pause.endS && rightS > pause.startS),
  }
}

/** `data-domain-pauses` attribute text: `start:end` pairs in elapsed seconds. */
export const formatElapsedPauses = (pauses: readonly ElapsedPause[]): string =>
  pauses.map(pause => `${pause.startS}:${pause.endS}`).join(' ')

export const parseElapsedPauses = (value: string | undefined): ElapsedPause[] =>
  (value ?? '').split(' ').flatMap(pair => {
    const [startS, endS] = pair.split(':').map(Number)
    return Number.isFinite(startS) && Number.isFinite(endS) && endS > startS
      ? [{ startS, endS }]
      : []
  })
