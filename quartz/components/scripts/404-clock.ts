type Mode = 'idle' | 'hot' | 'enter'

// Hand angles in degrees, clockwise from twelve. The landscape paints them into the rift.
export type Clock = {
  hour: number
  minute: number
  second: number
  step(dt: number): void
  mode(mode: Mode): void
}

// Hilfiker's railway clock. The master clock sends one impulse a minute: the minute hand jumps on
// it, the hour hand creeps half a degree, and the red second hand, which sweeps round in 58.5 s,
// waits at twelve until the impulse releases it. Hovering the gate warps time forward, so impulses
// arrive faster than the second hand can leave twelve; letting go winds the hands back to now.
const WARP: Record<Mode, number> = { idle: 1, hot: 900, enter: 9000 }
const SWEEP = 58.5

export function makeClock(reduce: boolean): Clock {
  let mode: Mode = 'idle'
  let warp = 1
  let offset = 0
  let minute = NaN
  let impulse = 0
  let way = 1
  let sweep = 0

  const clock: Clock = {
    hour: 0,
    minute: 0,
    second: 0,
    step(dt) {
      if (reduce || mode === 'idle') {
        warp = 1
        offset = reduce ? 0 : offset * Math.exp(-dt * 2.4)
        if (Math.abs(offset) < 0.5) offset = 0
      } else {
        warp += (WARP[mode] - warp) * Math.min(1, dt * 2)
        offset += (warp - 1) * dt
      }
      const now = performance.now() / 1000
      const local = Date.now() / 1000 + offset - new Date().getTimezoneOffset() * 60
      const n = Math.floor(local / 60)
      if (n !== minute) {
        way = n >= minute || Number.isNaN(minute) ? 1 : -1
        impulse = Number.isNaN(minute) ? now - 1 : now
        minute = n
      }
      const since = now - impulse
      // In step with the wall clock the hand reads true seconds; while time is warped it only
      // gets as far as it can between impulses. Either way it eases rather than snaps.
      const seconds = offset === 0 ? local - n * 60 : Math.min(since, local - n * 60)
      const target = reduce ? 0 : (Math.min(1, seconds / SWEEP) * 360) % 360
      const gap = ((target - sweep + 540) % 360) - 180
      sweep = Math.abs(gap) < 3 || dt === 0 ? target : sweep + gap * Math.min(1, dt * 10)
      // The minute hand overshoots its jump and rings down, as the real ones do.
      const ring = reduce ? 0 : Math.exp(-since / 0.07) * Math.cos(since * 48)
      clock.minute = 6 * (n % 60) - 6 * way * ring
      clock.hour = 0.5 * (n % 720)
      clock.second = sweep
    },
    mode(next) {
      mode = next
    },
  }
  return clock
}
