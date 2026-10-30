;(() => {
  if (globalThis.audioOutputConnected) return
  globalThis.audioOutputConnected = true
  let port
  let desired = { enabled: false }
  let running = false
  let pending = false
  let notice
  let reconnect
  const audioGraphs = new Map()
  let dismissedUID

  function report(state, detail) {
    port?.postMessage({ type: 'status', state, detail, outputUID: desired.rule?.outputUID ?? '' })
  }

  function removeNotice() {
    notice?.remove()
    notice = undefined
  }

  function showPermission() {
    if (notice || dismissedUID === desired.rule?.outputUID) return
    notice = document.createElement('div')
    Object.assign(notice.style, {
      position: 'fixed',
      right: '16px',
      bottom: '96px',
      zIndex: '2147483647',
    })
    const shadow = notice.attachShadow({ mode: 'closed' })
    const style = document.createElement('style')
    style.textContent = `:host{color-scheme:light dark}section{font:13px system-ui;background:Canvas;color:CanvasText;border:1px solid GrayText;padding:12px;max-width:290px}p{margin:0 0 10px;line-height:1.4}button{font:inherit;padding:6px 10px;margin-right:6px;cursor:pointer}`
    const section = document.createElement('section')
    const text = document.createElement('p')
    const canSelect = typeof navigator.mediaDevices.selectAudioOutput === 'function'
    text.textContent = canSelect
      ? `Allow Audio Output to use ${desired.rule.outputName}.`
      : 'Allow microphone access to reveal output devices. The microphone stream closes immediately; no audio is saved.'
    const button = document.createElement('button')
    button.textContent = canSelect ? 'Choose output' : 'Allow device access'
    button.addEventListener('click', async () => {
      try {
        if (canSelect) {
          const chosen = await navigator.mediaDevices.selectAudioOutput()
          if (!sameDevice(chosen.label, desired.rule.outputName))
            throw new Error(`Choose ${desired.rule.outputName} to match the desktop setting.`)
          await chrome.storage.local.set({ [bindingKey()]: chosen.deviceId })
        } else {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
          stream.getTracks().forEach(track => track.stop())
        }
        removeNotice()
        schedule()
      } catch (error) {
        text.textContent = error.message
        report('permission', 'Output permission needed')
      }
    })
    const close = document.createElement('button')
    close.textContent = 'Dismiss'
    close.addEventListener('click', () => {
      dismissedUID = desired.rule?.outputUID
      removeNotice()
    })
    section.append(text, button, close)
    shadow.append(style, section)
    document.documentElement.append(notice)
  }

  function bindingKey() {
    return `output:${location.hostname}:${desired.rule.outputUID}`
  }

  function sameDevice(label, name) {
    const actual = label.trim().toLocaleLowerCase()
    const expected = name.trim().toLocaleLowerCase()
    return actual === expected || actual.startsWith(`${expected} (`)
  }

  function desiredGain() {
    const gain = desired.rule?.volume
    return desired.enabled && Number.isFinite(gain) ? Math.min(3, Math.max(0, gain)) : 1
  }

  async function setOutput(id, gain = 1) {
    const elements = [...document.querySelectorAll('audio, video')]
    for (const [element, graph] of audioGraphs) {
      if (!element.isConnected) {
        graph.gain.gain.value = 1
        await graph.context.setSinkId('')
      }
    }
    for (const element of elements) {
      let graph = audioGraphs.get(element)
      if (!graph && gain !== 1 && element.currentSrc) {
        // Remote URLs can redirect to media without CORS. Only attach to local
        // blob media or an explicitly CORS-enabled element.
        const sourceURL = new URL(element.currentSrc, location.href)
        const localBlob = sourceURL.protocol === 'blob:' && sourceURL.origin === location.origin
        if (!localBlob && element.crossOrigin === null)
          throw new Error('Volume needs CORS-enabled or local blob media. Normal playback retained.')
        if (element.mediaKeys)
          throw new Error('Volume is unavailable for protected media. Normal playback retained.')
        const context = new AudioContext()
        if (typeof context.setSinkId !== 'function') {
          await context.close()
          throw new Error('This browser does not support volume with output routing')
        }
        // Resume before taking ownership of the element, so autoplay restrictions
        // cannot silence its original playback.
        void context.resume().catch(() => {})
        await new Promise(resolve => setTimeout(resolve, 50))
        if (context.state !== 'running') {
          await context.close()
          throw new Error('Click the player to enable volume, then retry')
        }
        try {
          await context.setSinkId(id)
          const source = context.createMediaElementSource(element)
          const amplifier = context.createGain()
          amplifier.gain.value = gain
          source.connect(amplifier).connect(context.destination)
          graph = { context, gain: amplifier, source }
          audioGraphs.set(element, graph)
        } catch (error) {
          await context.close()
          throw error
        }
      }
      if (graph) {
        if (graph.context.sinkId !== id) await graph.context.setSinkId(id)
        graph.gain.gain.setTargetAtTime(gain, graph.context.currentTime, 0.015)
      } else {
        if (typeof element.setSinkId !== 'function')
          throw new Error('This browser does not support per-site audio output')
        if (element.sinkId !== id) await element.setSinkId(id)
        if (element.sinkId !== id) throw new Error('The browser did not apply the output selection')
      }
    }
    return elements.filter(element => element.currentSrc).length
  }

  async function apply() {
    const rule = desired.rule
    if (!desired.enabled || !rule) {
      await setOutput('')
      removeNotice()
      report('normal', desired.error || 'Normal output')
      return
    }
    const gain = desiredGain()
    if (!rule.outputUID) {
      const count = await setOutput('', gain)
      removeNotice()
      report(gain === 1 ? 'normal' : count ? 'routing' : 'waiting',
        count ? `Normal output · ${Math.round(gain * 100)}%` : 'Waiting for a media element')
      return
    }
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
      device => device.kind === 'audiooutput',
    )
    const stored = await chrome.storage.local.get(bindingKey())
    let device = devices.find(
      item => item.deviceId === stored[bindingKey()] && item.deviceId !== 'default',
    )
    if (!device) {
      const matches = devices.filter(
        item =>
          item.deviceId !== 'default' &&
          item.deviceId !== 'communications' &&
          sameDevice(item.label, rule.outputName),
      )
      if (matches.length === 1) device = matches[0]
      if (matches.length > 1) {
        await setOutput('')
        report('error', 'Multiple matching outputs; choose one in the page')
        showPermission()
        return
      }
    }
    if (!device) {
      await setOutput('')
      const hidden = !devices.some(
        item => item.deviceId !== 'default' && item.deviceId !== 'communications' && item.label,
      )
      report(
        hidden ? 'permission' : 'missing',
        hidden
          ? `Allow device access in ${location.hostname}`
          : `${rule.outputName} unavailable · normal output`,
      )
      if (hidden) showPermission()
      else removeNotice()
      return
    }
    const count = await setOutput(device.deviceId, gain)
    removeNotice()
    report(
      count ? 'routing' : 'waiting',
      count ? `Routing to ${rule.outputName} · ${Math.round(gain * 100)}%` : 'Waiting for a media element',
    )
  }

  async function schedule() {
    pending = true
    if (running) return
    running = true
    while (pending) {
      pending = false
      try {
        await apply()
      } catch (error) {
        try {
          await setOutput('')
        } catch {}
        report(error.name === 'NotAllowedError' ? 'permission' : 'error', error.message)
        if (error.name === 'NotAllowedError') showPermission()
      }
    }
    running = false
  }

  function connect() {
    clearTimeout(reconnect)
    port = chrome.runtime.connect({ name: 'audiooutput-site' })
    port.onMessage.addListener(message => {
      if (message.type !== 'configuration') return
      desired = message
      schedule()
    })
    port.onDisconnect.addListener(() => {
      void chrome.runtime.lastError
      port = undefined
      desired = { enabled: false }
      schedule()
      reconnect = setTimeout(connect, 3000)
    })
  }

  new MutationObserver(changes => {
    if (
      changes.some(change =>
        [...change.addedNodes].some(
          node =>
            node.nodeType === Node.ELEMENT_NODE &&
            (node.matches('audio, video') || node.querySelector('audio, video')),
        ),
      )
    )
      schedule()
  }).observe(document.documentElement, { childList: true, subtree: true })
  document.addEventListener('play', schedule, true)
  document.addEventListener('loadedmetadata', schedule, true)
  document.addEventListener('click', () => {
    for (const graph of audioGraphs.values()) void graph.context.resume().catch(() => {})
    schedule()
  }, true)
  navigator.mediaDevices.addEventListener('devicechange', schedule)
  connect()
})()
