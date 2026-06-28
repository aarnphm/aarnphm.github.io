import { AUDIO_ICON_PATHS, audioBarHeight, audioIconSvg } from '../../util/audio'

const PITCH = 4
const SVG_PLAY = audioIconSvg(AUDIO_ICON_PATHS.play)
const SVG_PAUSE = audioIconSvg(AUDIO_ICON_PATHS.pause)

function fmt(t: number): string {
  if (!isFinite(t) || t < 0) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const MAX_DECODE_BYTES = 20_000_000
const peakCache = new Map<string, number[]>()
let decodeCtx: AudioContext | null = null

async function computePeaks(
  url: string,
  count: number,
  signal: AbortSignal,
): Promise<number[] | null> {
  const key = `${url}@${count}`
  const cached = peakCache.get(key)
  if (cached) return cached
  try {
    const res = await fetch(url, { signal })
    if (!res.ok) return null
    if (Number(res.headers.get('content-length') ?? '0') > MAX_DECODE_BYTES) return null
    const bytes = await res.arrayBuffer()
    decodeCtx ??= new AudioContext()
    const buf = await decodeCtx.decodeAudioData(bytes)
    const data = buf.getChannelData(0)
    const block = Math.max(1, Math.floor(data.length / count))
    const rms: number[] = []
    let max = 0
    for (let i = 0; i < count; i++) {
      const start = i * block
      let sum = 0
      let n = 0
      for (let j = 0; j < block && start + j < data.length; j++) {
        const v = data[start + j]
        sum += v * v
        n++
      }
      const r = n ? Math.sqrt(sum / n) : 0
      rms.push(r)
      if (r > max) max = r
    }
    const norm = rms.map(r => (max ? Math.pow(r / max, 0.8) : 0))
    peakCache.set(key, norm)
    return norm
  } catch {
    return null
  }
}

function hydrate(wrap: HTMLElement): void {
  if (wrap.dataset.customized === 'true') return
  const audio = wrap.querySelector<HTMLAudioElement>('audio')
  const btn = wrap.querySelector<HTMLButtonElement>('.ap-play')
  const repeat = wrap.querySelector<HTMLButtonElement>('.ap-repeat')
  const bars = wrap.querySelector<HTMLElement>('.ap-bars')
  const time = wrap.querySelector<HTMLElement>('.ap-time')
  if (!audio || !btn || !bars || !time) return
  wrap.dataset.customized = 'true'

  audio.removeAttribute('controls')
  audio.removeAttribute('loading')
  audio.preload = 'metadata'

  const count = Math.max(24, Math.floor((bars.clientWidth || 600) / PITCH))
  bars.replaceChildren()
  const barEls: HTMLElement[] = []
  for (let i = 0; i < count; i++) {
    const b = document.createElement('span')
    b.style.height = `${audioBarHeight(i)}%`
    bars.appendChild(b)
    barEls.push(b)
  }

  const ac = new AbortController()
  let decoding = false
  const io = new IntersectionObserver(
    entries => {
      if (decoding || !entries.some(e => e.isIntersecting)) return
      decoding = true
      void computePeaks(audio.src, barEls.length, ac.signal).then(peaks => {
        decoding = false
        if (!peaks) return
        io.disconnect()
        for (let i = 0; i < barEls.length; i++) {
          barEls[i].style.height = `${Math.round(8 + (peaks[i] ?? 0) * 88)}%`
        }
      })
    },
    { rootMargin: '200px 0px' },
  )
  io.observe(wrap)

  const render = () => {
    const pct = audio.duration ? audio.currentTime / audio.duration : 0
    const played = Math.round(pct * barEls.length)
    for (let i = 0; i < barEls.length; i++) {
      barEls[i].classList.toggle('played', i < played)
    }
    time.textContent = `${fmt(audio.currentTime)} / ${fmt(audio.duration)}`
  }
  const onPlay = () => {
    btn.innerHTML = SVG_PAUSE
    btn.setAttribute('aria-label', 'Pause')
  }
  const onPause = () => {
    btn.innerHTML = SVG_PLAY
    btn.setAttribute('aria-label', 'Play')
  }

  let busy = false
  const toggle = () => {
    if (busy) return
    if (audio.paused) {
      busy = true
      const p = audio.play()
      if (p && typeof p.then === 'function') {
        p.then(() => (busy = false)).catch(() => (busy = false))
      } else {
        busy = false
      }
    } else {
      audio.pause()
    }
  }
  const seek = (e: MouseEvent) => {
    const rect = bars.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    if (audio.duration) audio.currentTime = ratio * audio.duration
    if (audio.paused) void audio.play().catch(() => {})
  }
  const toggleRepeat = () => {
    audio.loop = !audio.loop
    repeat?.classList.toggle('active', audio.loop)
    repeat?.setAttribute('aria-pressed', String(audio.loop))
  }

  btn.addEventListener('click', toggle)
  repeat?.addEventListener('click', toggleRepeat)
  bars.addEventListener('click', seek)
  audio.addEventListener('timeupdate', render)
  audio.addEventListener('loadedmetadata', render)
  audio.addEventListener('durationchange', render)
  audio.addEventListener('play', onPlay)
  audio.addEventListener('pause', onPause)
  audio.addEventListener('ended', onPause)
  render()
  audio.load()

  window.addCleanup(() => {
    btn.removeEventListener('click', toggle)
    repeat?.removeEventListener('click', toggleRepeat)
    bars.removeEventListener('click', seek)
    audio.removeEventListener('timeupdate', render)
    audio.removeEventListener('loadedmetadata', render)
    audio.removeEventListener('durationchange', render)
    audio.removeEventListener('play', onPlay)
    audio.removeEventListener('pause', onPause)
    audio.removeEventListener('ended', onPause)
    io.disconnect()
    ac.abort()
    if (!audio.paused) audio.pause()
  })
}

document.addEventListener('nav', () => {
  document.querySelectorAll<HTMLElement>('figure.audio-player[data-audio-embed]').forEach(hydrate)
})
