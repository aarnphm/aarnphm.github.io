document.addEventListener('nav', () => {
  const onMessage = (event: MessageEvent) => {
    const data = event.data
    if (!Array.isArray(data) || data[1] !== 'BROADCAST_IFRAME_HEIGHT') return
    const height = Number(data[2])
    if (!Number.isFinite(height) || height <= 0) return
    for (const frame of Array.from(
      document.querySelectorAll<HTMLIFrameElement>('iframe.external-embed'),
    )) {
      if (frame.contentWindow && frame.contentWindow === event.source) {
        frame.style.height = `${Math.round(height)}px`
        frame.style.aspectRatio = 'auto'
        return
      }
    }
  }
  window.addEventListener('message', onMessage)
  window.addCleanup(() => window.removeEventListener('message', onMessage))
})
