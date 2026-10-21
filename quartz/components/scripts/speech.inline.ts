import {
  DEFAULT_SPEECH_MODEL,
  PHONON_MODELS,
  SPEECH_MODEL_IDS,
  SPEECH_MODEL_INFO,
  isSpeechModelId,
  speechModelLabel,
  type SpeechModelId,
} from '../../util/speech-protocol'
import { registerEscapeHandler } from './escape-handler'
import { speechRuntime } from './speech-runtime'
import { createSpeechPicker, createSpeechSpeed, type SpeechSpeed } from './speech-speed'

const SPEECH_SPEED_KEY = 'garden:speech-speed'
const SPEECH_MODEL_KEY = 'garden:speech-model'

function savedSpeed(): SpeechSpeed {
  try {
    return localStorage.getItem(SPEECH_SPEED_KEY) === '0.8' ? '0.8' : '1'
  } catch {
    return '1'
  }
}

function savedModel(): SpeechModelId {
  try {
    const value = localStorage.getItem(SPEECH_MODEL_KEY)
    return isSpeechModelId(value) ? value : DEFAULT_SPEECH_MODEL
  } catch {
    return DEFAULT_SPEECH_MODEL
  }
}

document.addEventListener('nav', () => {
  const phrases = document.querySelectorAll<HTMLElement>('.speech-phrase[data-speech-text]')
  if (phrases.length === 0) return
  const firstPhrase = phrases[0]
  const root = firstPhrase.closest<HTMLElement>('.flashcards-root, article')
  if (!root || document.getElementById('speech-settings')) return
  const listeners = new AbortController()
  const modelBaseUrl = firstPhrase.dataset.speechModelBaseUrl
  let model = savedModel()
  let runtime = speechRuntime(modelBaseUrl, model)

  const controls = document.createElement('aside')
  controls.id = 'speech-settings'
  controls.className = 'speech-controls'
  controls.lang = 'fr'
  controls.popover = 'auto'
  controls.setAttribute('aria-label', 'Prononciation')

  const header = document.createElement('div')
  header.className = 'speech-controls-head'
  const title = document.createElement('span')
  title.className = 'speech-controls-title'
  title.textContent = 'Prononciation'
  const shortcut = document.createElement('kbd')
  shortcut.className = 'speech-controls-shortcut'
  shortcut.textContent = 'gv'
  const closeButton = document.createElement('button')
  closeButton.type = 'button'
  closeButton.className = 'speech-close'
  closeButton.autofocus = true
  closeButton.setAttribute('aria-label', 'Fermer les réglages de prononciation')
  closeButton.dataset.siteCursorClose = ''
  closeButton.innerHTML =
    '<svg data-site-cursor-icon viewBox="0 0 12 12" aria-hidden="true"><path d="M2.25 2.25 9.75 9.75M9.75 2.25 2.25 9.75"/></svg>'
  header.append(title, shortcut, closeButton)

  const row = document.createElement('div')
  row.className = 'speech-controls-row'
  let speed = savedSpeed()
  const speedPicker = createSpeechSpeed(
    speed,
    value => {
      speed = value
      player.playbackRate = Number(speed)
      try {
        localStorage.setItem(SPEECH_SPEED_KEY, speed)
      } catch {
        // Playback still works when the browser disallows persistent preferences.
      }
    },
    listeners.signal,
  )
  const stopButton = document.createElement('button')
  stopButton.type = 'button'
  stopButton.className = 'speech-stop'
  stopButton.textContent = 'Arrêter'
  stopButton.hidden = true
  row.append(speedPicker.element, stopButton)

  const details = document.createElement('details')
  details.className = 'speech-model-details'
  const summary = document.createElement('summary')
  const explanation = document.createElement('p')
  const describeModel = () => {
    const phonon = model === 'supertonic-2' ? undefined : PHONON_MODELS[model]
    const info = phonon ?? SPEECH_MODEL_INFO
    const downloadMb = Math.ceil(info.downloadBytes / 1_000_000)
    summary.textContent = `Voix française synthétique · ${downloadMb} Mo au premier clic`
    const modelLink = document.createElement('a')
    modelLink.href = info.sourceUrl
    modelLink.textContent = phonon
      ? `${phonon.label}, voix par défaut, licence ${phonon.license}`
      : `${SPEECH_MODEL_INFO.label}, voix ${SPEECH_MODEL_INFO.voice}`
    explanation.replaceChildren(
      modelLink,
      document.createTextNode(
        `. Au premier clic, environ ${downloadMb} Mo sont téléchargés, puis gardés en cache si le navigateur le permet. Votre texte reste sur cet appareil. Cette voix n'est pas une référence d'accent québécois.`,
      ),
    )
  }
  describeModel()
  details.append(summary, explanation)
  const status = document.createElement('p')
  status.className = 'speech-status'
  status.setAttribute('role', 'status')
  status.setAttribute('aria-live', 'polite')
  status.hidden = true
  const modelPicker = createSpeechPicker({
    id: 'speech-model-options',
    label: 'Modèle',
    menuLabel: 'Modèle vocal',
    choices: SPEECH_MODEL_IDS.map(id => [id, speechModelLabel(id)]),
    initial: model,
    select: value => {
      if (value === model) return
      stop()
      model = value
      try {
        localStorage.setItem(SPEECH_MODEL_KEY, model)
      } catch {
        // The choice still applies to this page.
      }
      if (runtime.onProgress === onProgress) runtime.onProgress = undefined
      runtime = speechRuntime(modelBaseUrl, model)
      runtime.onProgress = onProgress
      describeModel()
      void runtime.client.warmCached()
    },
    signal: listeners.signal,
  })
  const modelRow = document.createElement('div')
  modelRow.className = 'speech-controls-row'
  modelRow.append(modelPicker.element)
  controls.append(header, row, modelRow, details, status)
  const card = root.querySelector('.flashcards-card')
  document.body.append(controls)

  let previousFocus: HTMLElement | undefined
  controls.addEventListener(
    'beforetoggle',
    event => {
      if (event.newState === 'open') {
        previousFocus =
          document.activeElement instanceof HTMLElement ? document.activeElement : undefined
      } else {
        speedPicker.close()
        modelPicker.close()
        if (controls.contains(document.activeElement) && previousFocus?.isConnected)
          previousFocus.focus({ preventScroll: true })
      }
    },
    { signal: listeners.signal },
  )
  closeButton.addEventListener('click', () => controls.hidePopover(), { signal: listeners.signal })
  const unregisterEscape = registerEscapeHandler(
    controls,
    () => controls.hidePopover(),
    () => controls.matches(':popover-open'),
  )
  const player = new Audio()
  player.className = 'speech-audio'
  player.hidden = true
  player.preservesPitch = true
  controls.append(player)
  let active: HTMLButtonElement | undefined
  let request = 0
  let disposed = false
  let requestedAt = 0
  let intentTimer: ReturnType<typeof setTimeout> | undefined

  for (const phrase of phrases) {
    const button = phrase.querySelector<HTMLButtonElement>('.speech-play')
    if (button) {
      button.disabled = false
      button.title = 'Écouter la prononciation'
    }
  }

  // Tells the page what the voice is doing; the pet (pet.inline.ts) says the phrase in its bubble.
  const announce = (state: 'loading' | 'playing' | 'idle', phrase?: HTMLElement) => {
    const detail = {
      state,
      text: phrase?.dataset.speechText?.trim() ?? '',
      lang: phrase?.closest<HTMLElement>('[lang]')?.lang || controls.lang,
      phrase,
    }
    document.dispatchEvent(new CustomEvent('speech', { detail }))
  }
  const setStatus = (text: string, error = false) => {
    status.textContent = text
    status.hidden = !text
    status.dataset.error = String(error)
  }
  const onProgress = (message: string) => {
    if (!disposed && active?.dataset.speechState === 'loading') setStatus(message)
  }
  runtime.onProgress = onProgress
  void runtime.client.warmCached()
  const resetActive = () => {
    if (!active) return
    active.removeAttribute('aria-busy')
    active.removeAttribute('data-speech-state')
    const text = active.closest<HTMLElement>('.speech-phrase')?.dataset.speechText ?? ''
    active.setAttribute('aria-label', `Écouter : ${text}`)
    active.title = 'Écouter la prononciation'
    active = undefined
  }
  const stop = () => {
    clearTimeout(intentTimer)
    request++
    player.pause()
    player.removeAttribute('src')
    player.load()
    resetActive()
    stopButton.hidden = true
    setStatus('')
    announce('idle')
  }
  const speak = async (button: HTMLButtonElement, phrase: HTMLElement) => {
    if (button === active) {
      stop()
      return
    }
    const text = phrase.dataset.speechText?.trim()
    if (!text || phrase.closest('[hidden]')) return
    stop()
    requestedAt = performance.now()
    const currentRequest = request
    active = button
    button.removeAttribute('data-speech-latency-ms')
    button.dataset.speechState = 'loading'
    button.setAttribute('aria-busy', 'true')
    button.setAttribute('aria-label', `Arrêter : ${text}`)
    stopButton.hidden = false
    setStatus('Préparation de la voix…')
    announce('loading', phrase)
    try {
      const url = await runtime.prepare(text)
      if (disposed || currentRequest !== request) return
      player.src = url
      player.playbackRate = Number(speed)
      await player.play()
      if (disposed || currentRequest !== request) return
      button.removeAttribute('aria-busy')
      button.dataset.speechState = 'playing'
      button.title = 'Arrêter la lecture'
      setStatus(`Lecture : ${text}`)
      announce('playing', phrase)
    } catch (error) {
      if (disposed || currentRequest !== request) return
      resetActive()
      stopButton.hidden = true
      announce('idle')
      const message = error instanceof Error ? error.message : String(error)
      setStatus(`Lecture indisponible. ${message}`, true)
    }
  }

  root.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const button = event.target.closest<HTMLButtonElement>('button.speech-play')
      const phrase = button?.closest<HTMLElement>('.speech-phrase[data-speech-text]')
      if (!button || !phrase) return
      event.stopPropagation()
      void speak(button, phrase)
    },
    { signal: listeners.signal },
  )
  stopButton.addEventListener('click', stop, { signal: listeners.signal })
  const prepareIntent = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const phrase = event.target.closest<HTMLElement>('.speech-phrase[data-speech-text]')
    if (!phrase || phrase.closest('[hidden]') || active?.dataset.speechState === 'loading') return
    const text = phrase.dataset.speechText?.trim()
    if (!text) return
    clearTimeout(intentTimer)
    intentTimer = setTimeout(() => {
      if (!disposed && !phrase.closest('[hidden]')) runtime.prefetch(text)
    }, 80)
  }
  root.addEventListener('pointerover', prepareIntent, { signal: listeners.signal })
  root.addEventListener('focusin', prepareIntent, { signal: listeners.signal })
  player.addEventListener(
    'playing',
    () => {
      if (!active) return
      active.dataset.speechLatencyMs = String(Math.round(performance.now() - requestedAt))
      const text = active.closest<HTMLElement>('.speech-phrase')?.dataset.speechText?.trim()
      const timing = text ? runtime.timings.get(text) : undefined
      active.dataset.speechModel = model
      if (timing) {
        active.dataset.speechSynthMs = String(Math.round(timing.totalMs))
        active.dataset.speechFirstAudioMs = String(Math.round(timing.firstAudioMs))
        active.dataset.speechAudioMs = String(Math.round(timing.audioMs))
        active.dataset.speechDevice = timing.device
      }
    },
    { signal: listeners.signal },
  )
  player.addEventListener(
    'ended',
    () => {
      resetActive()
      stopButton.hidden = true
      setStatus('')
      announce('idle')
    },
    { signal: listeners.signal },
  )
  player.addEventListener(
    'error',
    () => {
      if (!player.hasAttribute('src')) return
      resetActive()
      stopButton.hidden = true
      announce('idle')
      setStatus('Lecture indisponible. Réessayez avec une autre phrase.', true)
    },
    { signal: listeners.signal },
  )

  const cardObserver = card
    ? new MutationObserver(() => {
        if (active && (active.closest('[hidden]') || !active.closest('.flashcard.is-active')))
          stop()
      })
    : undefined
  cardObserver?.observe(card ?? root, {
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'hidden'],
  })

  window.addCleanup(() => {
    disposed = true
    stop()
    listeners.abort()
    unregisterEscape()
    clearTimeout(intentTimer)
    cardObserver?.disconnect()
    if (runtime.onProgress === onProgress) runtime.onProgress = undefined
    controls.remove()
  })
})
