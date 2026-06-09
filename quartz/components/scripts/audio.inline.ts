const PITCH = 4
const SVG_PLAY = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>`
const SVG_PAUSE = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.4v14H7zm6.6 0H17v14h-3.4z"/></svg>`

function fmt(t: number): string {
  if (!isFinite(t) || t < 0) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function barHeight(i: number): number {
  return 30 + Math.round(60 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.55)))
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

function buildPlayer(audio: HTMLAudioElement): void {
  if (audio.dataset.customized === 'true') return
  audio.dataset.customized = 'true'
  audio.removeAttribute('controls')
  audio.removeAttribute('loading')
  audio.preload = 'metadata'

  const wrap = document.createElement('div')
  wrap.className = 'audio-player'

  const row = document.createElement('div')
  row.className = 'ap-row'

  const btn = document.createElement('button')
  btn.className = 'ap-play'
  btn.type = 'button'
  btn.setAttribute('aria-label', 'Play')
  btn.innerHTML = SVG_PLAY

  const bars = document.createElement('div')
  bars.className = 'ap-bars'

  const time = document.createElement('div')
  time.className = 'ap-time'
  time.textContent = '0:00 / 0:00'

  row.append(btn, bars)
  wrap.append(row, time)
  audio.parentNode?.insertBefore(wrap, audio)

  const count = Math.max(24, Math.floor((bars.clientWidth || 600) / PITCH))
  const barEls: HTMLElement[] = []
  for (let i = 0; i < count; i++) {
    const b = document.createElement('span')
    b.style.height = `${barHeight(i)}%`
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
  }

  btn.addEventListener('click', toggle)
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
  document.querySelectorAll<HTMLAudioElement>('audio').forEach(buildPlayer)
})
