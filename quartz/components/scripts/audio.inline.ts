/**
 * custom audio player with Substack-style controls
 * features:
 * - play/pause toggle
 * - progress bar scrubbing (click + drag)
 * - time display (current and remaining)
 * - playback speed control (0.5x, 1x, 1.5x, 2x)
 * - keyboard shortcuts (space, left/right arrows, up/down for speed)
 * - timestamp links (#t=1m30s)
 * - progress persistence (localStorage)
 */

/**
 * format seconds as MM:SS or H:MM:SS
 */
function formatTime(seconds: number): string {
  if (isNaN(seconds) || !isFinite(seconds)) return "0:00"

  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)

  if (h > 0) {
    return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
  }
  return `${m}:${s.toString().padStart(2, "0")}`
}

/**
 * load saved progress from localStorage
 */
function loadProgress(src: string): number | null {
  try {
    const saved = localStorage.getItem(`audio-progress:${src}`)
    return saved ? parseFloat(saved) : null
  } catch {
    return null
  }
}

/**
 * save progress to localStorage (throttled)
 */
function saveProgress(src: string, time: number, duration: number): void {
  try {
    // if within last 10 seconds, clear saved progress (treat as finished)
    if (duration - time < 10) {
      localStorage.removeItem(`audio-progress:${src}`)
    } else {
      localStorage.setItem(`audio-progress:${src}`, time.toString())
    }
  } catch {
    // swallow storage errors
  }
}

class AudioPlayer {
  private container: HTMLElement
  private audio: HTMLAudioElement
  private playButton: HTMLButtonElement
  private progressContainer: HTMLElement
  private progressTrack: HTMLElement
  private progressBar: HTMLElement
  private playhead: HTMLElement
  private timeCurrent: HTMLElement
  private timeRemaining: HTMLElement
  private speedButton: HTMLButtonElement
  private isDragging: boolean = false
  private lastProgressSave: number = 0

  constructor(container: HTMLElement) {
    this.container = container
    this.audio = container.querySelector("audio")!
    this.playButton = container.querySelector(".audio-play-button")!
    this.progressContainer = container.querySelector(".audio-progress-container")!
    this.progressTrack = container.querySelector(".audio-progress-track")!
    this.progressBar = container.querySelector(".audio-progress-bar")!
    this.playhead = container.querySelector(".audio-playhead")!
    this.timeCurrent = container.querySelector(".audio-time-current")!
    this.timeRemaining = container.querySelector(".audio-time-remaining")!
    this.speedButton = container.querySelector(".audio-speed")!
  }

  public init(): void {
    // load saved speed preference
    const savedSpeed = parseFloat(this.speedButton.dataset.speed || "1")
    this.audio.playbackRate = savedSpeed
    this.updateSpeedDisplay()

    // handle timestamp from URL or data attribute
    const startTime = this.container.dataset.startTime
    if (startTime) {
      const time = parseFloat(startTime)
      if (!isNaN(time)) {
        this.audio.currentTime = time
      }
    } else {
      const savedTime = loadProgress(this.audio.src)
      if (savedTime !== null && savedTime > 0) {
        this.audio.currentTime = savedTime
      }
    }

    // attach event listeners
    this.playButton.addEventListener("click", this.togglePlay)
    this.audio.addEventListener("timeupdate", this.onTimeUpdate)
    this.audio.addEventListener("loadedmetadata", this.onMetadataLoaded)
    this.audio.addEventListener("ended", this.onEnded)
    this.progressContainer.addEventListener("mousedown", this.onProgressMouseDown)
    this.speedButton.addEventListener("click", this.cycleSpeed)
    document.addEventListener("keydown", this.onKeyDown)

    // register cleanup
    window.addCleanup(() => {
      this.playButton.removeEventListener("click", this.togglePlay)
      this.audio.removeEventListener("timeupdate", this.onTimeUpdate)
      this.audio.removeEventListener("loadedmetadata", this.onMetadataLoaded)
      this.audio.removeEventListener("ended", this.onEnded)
      this.progressContainer.removeEventListener("mousedown", this.onProgressMouseDown)
      this.speedButton.removeEventListener("click", this.cycleSpeed)
      document.removeEventListener("keydown", this.onKeyDown)
      // pause on cleanup
      if (!this.audio.paused) {
        this.audio.pause()
      }
    })

    // initial time display update
    this.updateTimeDisplay()
  }

  private togglePlay = (): void => {
    if (this.audio.paused) {
      // pause all other audio players
      document.querySelectorAll<HTMLAudioElement>(".audio-embed audio").forEach((otherAudio) => {
        if (otherAudio !== this.audio && !otherAudio.paused) {
          otherAudio.pause()
        }
      })

      this.audio.play()
      this.container.classList.add("is-playing")
      this.playButton.setAttribute("aria-label", "Pause")
      this.playButton.setAttribute("aria-pressed", "true")
    } else {
      this.audio.pause()
      this.container.classList.remove("is-playing")
      this.playButton.setAttribute("aria-label", "Play")
      this.playButton.setAttribute("aria-pressed", "false")
    }
  }

  private onTimeUpdate = (): void => {
    this.updateTimeDisplay()
    this.updateProgress()

    // throttled progress save
    const now = Date.now()
    // millisecond here.
    if (now - this.lastProgressSave > 1000) {
      saveProgress(this.audio.src, this.audio.currentTime, this.audio.duration)
      this.lastProgressSave = now
    }
  }

  private onMetadataLoaded = (): void => {
    this.updateTimeDisplay()
  }

  private onEnded = (): void => {
    this.container.classList.remove("is-playing")
    this.playButton.setAttribute("aria-label", "Play")
    this.playButton.setAttribute("aria-pressed", "false")
    // clear saved progress when finished
    localStorage.removeItem(getStorageKey(this.audio.src))
  }

  private updateTimeDisplay(): void {
    const current = this.audio.currentTime || 0
    const duration = this.audio.duration || 0

    this.timeCurrent.textContent = formatTime(current)
    this.timeRemaining.textContent = duration > 0 ? `-${formatTime(duration - current)}` : "0:00"
  }

  private updateProgress(): void {
    const percent = this.audio.duration ? (this.audio.currentTime / this.audio.duration) * 100 : 0
    this.progressBar.style.width = `${percent}%`
    this.playhead.style.left = `${percent}%`
  }

  private onProgressMouseDown = (e: MouseEvent): void => {
    this.isDragging = true
    this.seekToPosition(e)

    const onMouseMove = (e: MouseEvent) => {
      if (this.isDragging) {
        this.seekToPosition(e)
      }
    }

    const onMouseUp = () => {
      this.isDragging = false
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }

    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }

  private seekToPosition(e: MouseEvent): void {
    const rect = this.progressTrack.getBoundingClientRect()
    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    this.audio.currentTime = percent * this.audio.duration
  }

  private cycleSpeed = (): void => {
    const SPEED_OPTIONS = [0.5, 1, 1.5, 2]

    const currentSpeed = this.audio.playbackRate
    const currentIndex = SPEED_OPTIONS.indexOf(currentSpeed)
    const nextIndex = (currentIndex + 1) % SPEED_OPTIONS.length
    const nextSpeed = SPEED_OPTIONS[nextIndex]

    this.audio.playbackRate = nextSpeed
    this.speedButton.dataset.speed = nextSpeed.toString()
    this.updateSpeedDisplay()
  }

  private updateSpeedDisplay(): void {
    const speed = this.audio.playbackRate
    this.speedButton.textContent = `${speed}×`
    this.speedButton.setAttribute("aria-label", `Playback speed: ${speed}x`)
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    const SPEED_OPTIONS = [0.5, 1, 1.5, 2]

    // only handle if this player's container or controls are focused, or if no other input is focused
    const target = e.target as HTMLElement
    const isInputFocused =
      target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable

    // check if this player is in focus context
    const isThisPlayerFocused =
      this.container.contains(target) || this.container.contains(document.activeElement)

    if (isInputFocused && !isThisPlayerFocused) return

    // space: play/pause (only if this player is focused or playing)
    if (e.code === "Space" && (isThisPlayerFocused || !this.audio.paused)) {
      e.preventDefault()
      this.togglePlay()
      return
    }

    // only handle other shortcuts if this player is focused or playing
    if (!isThisPlayerFocused && this.audio.paused) return

    switch (e.code) {
      case "ArrowLeft":
        e.preventDefault()
        this.audio.currentTime = Math.max(0, this.audio.currentTime - 5)
        break
      case "ArrowRight":
        e.preventDefault()
        this.audio.currentTime = Math.min(this.audio.duration, this.audio.currentTime + 5)
        break
      case "ArrowUp":
        e.preventDefault()
        this.cycleSpeed()
        break
      case "ArrowDown":
        e.preventDefault()
        // cycle backwards through speeds
        const currentSpeed = this.audio.playbackRate
        const currentIndex = SPEED_OPTIONS.indexOf(currentSpeed)
        const prevIndex = (currentIndex - 1 + SPEED_OPTIONS.length) % SPEED_OPTIONS.length
        this.audio.playbackRate = SPEED_OPTIONS[prevIndex]
        this.speedButton.dataset.speed = SPEED_OPTIONS[prevIndex].toString()
        this.updateSpeedDisplay()
        break
    }
  }
}

document.addEventListener("nav", () => {
  const audioEmbeds = document.querySelectorAll(".audio-embed") as NodeListOf<HTMLElement>

  for (const embed of audioEmbeds) {
    if (embed.dataset.initialized) continue

    const _ = new AudioPlayer(embed)
    _.init()
    embed.dataset.initialized = "true"
  }
})
