import {
  cardFaces,
  cardResolution,
  FORMATS,
  palettes,
  prepareCard,
  prepareCardScene,
  renderCard,
  type Fonts,
  type SiteColors,
} from '../press-kit-card'
import { renderLinks } from './link-sharing'
import { $, arts, session, show, toast } from './session-state'
import { model, shareUrl, titleText } from './share-subject'
import { layoutMetricControls } from './training-controls'

export function resolveFonts(): Fonts {
  const root = getComputedStyle(document.documentElement)
  const variable = (name: string, fallback: string) =>
    root.getPropertyValue(name).trim() || fallback
  const heading = document.querySelector('h1.article-title')
  const weight = heading ? Number.parseInt(getComputedStyle(heading).fontWeight, 10) : 700
  const header = variable('--headerFont', 'system-ui, sans-serif')
  const headerFamily = header
    .split(',')[0]
    .trim()
    .replace(/^["']|["']$/g, '')
  const availableWeights = Array.from(document.fonts)
    .filter(
      face => face.family.replace(/^["']|["']$/g, '') === headerFamily && face.style === 'normal',
    )
    .map(face => Number(face.weight))
    .filter(Number.isFinite)
  // The page disables synthetic font weights; detached text canvases need the
  // recorded face's weight to render the same heading instead of synthesizing bold.
  const headerWeight = availableWeights.reduce(
    (closest, candidate) =>
      Math.abs(candidate - weight) < Math.abs(closest - weight) ? candidate : closest,
    availableWeights[0] ?? weight,
  )
  return {
    header,
    headerWeight: Number.isFinite(headerWeight) ? headerWeight : 700,
    body: variable('--bodyFont', 'system-ui, sans-serif'),
    serif: '"Parclo Serif", Cambria, Georgia, serif',
    mono: variable('--codeFont', 'ui-monospace, monospace'),
  }
}

/** Draws at once with whatever is loaded, then again once every card face has arrived. */
export function loadFonts(target: Fonts) {
  const loads = cardFaces(target).map(face => document.fonts.load(face, 'Aa“…').catch(() => []))
  const timeout = new Promise(resolve => setTimeout(resolve, 3000))
  void Promise.race([Promise.all(loads), timeout]).then(() => {
    if (session.fonts === target) {
      layoutMetricControls()
      schedule()
    }
  })
}

// The base social card; same-origin, so drawing it leaves the canvas exportable. The night card
// (quartz/scripts/brand-art.ts) is painted on the encre ground and only ever drawn into cards.
const ART_SRC = '/static/og-image.webp'
const ART_NIGHT_SRC = '/static/og-image-dark.webp'

export function loadArt() {
  for (const [key, src] of [
    ['day', ART_SRC],
    ['night', ART_NIGHT_SRC],
  ] as const) {
    if (arts[key]) continue
    const img = new Image()
    img.decoding = 'async'
    img.src = src
    img.onload = () => {
      arts[key] = img
      if (show.art) schedule()
    }
  }
}

function caption() {
  const spec = FORMATS[session.format]
  const resolution = cardResolution(session.format)
  const el = $('.press-kit-caption')
  if (!el) return
  el.textContent =
    session.tab === 'links'
      ? 'aperçu du lien partagé'
      : `${resolution.width} × ${resolution.height} · png · ${spec.label}`
}

export function draw() {
  session.frame = 0
  const version = ++session.drawVersion
  session.rendering = null
  window.clearTimeout(session.blobTimer)
  session.blob = null
  session.renderFailed = false
  if (!session.dialog?.open || !session.subject || !session.fonts || !session.page) return
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  const unfurl = $('.press-kit-unfurl')
  if (!canvas || !unfurl) return
  session.dialog.dataset.format = session.format
  caption()
  session.editor?.context(
    `${session.subject.url}:${session.tab}`,
    `${session.format}:${session.overlay}`,
    session.tab !== 'links',
  )
  if (session.tab === 'links') {
    canvas.hidden = true
    unfurl.hidden = false
    renderLinks()
    return
  }
  canvas.hidden = false
  unfurl.hidden = true
  const colors = session.page.colors as unknown as { light: SiteColors; dark: SiteColors }
  const palette = palettes(colors.light, colors.dark)[session.theme]
  if (session.editor?.textColor && session.editor.textColor !== 'auto') {
    const ink = session.editor.textColor === 'white' ? 'white' : 'black'
    palette.ink = ink
    palette.muted = ink
    palette.hlInk = ink
  }
  const targetFormat = session.format
  const targetFonts = session.fonts
  const targetDialog = session.dialog
  canvas.setAttribute(
    'aria-label',
    `aperçu de la carte, ${FORMATS[session.format].label}, ${session.theme}`,
  )
  canvas.setAttribute('aria-busy', 'true')
  const targetEditor = session.editor
  session.rendering = prepareCard(model(), palette, targetFormat, targetFonts, targetEditor?.active)
    .then(async prepared => {
      if (
        version !== session.drawVersion ||
        session.dialog !== targetDialog ||
        !session.dialog.open
      )
        return
      if (targetEditor?.active) {
        const scene = await prepareCardScene(
          prepared,
          targetFormat,
          palette,
          targetFonts,
          session.activeSignal ?? undefined,
        )
        if (
          version !== session.drawVersion ||
          session.dialog !== targetDialog ||
          !session.dialog.open
        )
          return
        targetEditor.setScene(scene)
      } else renderCard(canvas, prepared, targetFormat, palette, targetFonts)
      session.blobTimer = window.setTimeout(() => {
        session.blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
      }, 160)
    })
    .catch(() => {
      if (version !== session.drawVersion) return
      session.renderFailed = true
      toast('impossible de rendre la carte, veuillez réessayer')
    })
    .finally(() => {
      if (version !== session.drawVersion) return
      canvas.setAttribute('aria-busy', 'false')
      session.rendering = null
    })
}

export function schedule() {
  if (!session.frame) session.frame = requestAnimationFrame(draw)
}

async function currentBlob(): Promise<Blob | null> {
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  const targetDialog = session.dialog
  const targetEditor = session.editor
  do {
    if (session.frame) {
      cancelAnimationFrame(session.frame)
      draw()
    }
    if (session.rendering) await session.rendering
    if ((await targetEditor?.flush()) === false) return null
    if (
      !canvas?.isConnected ||
      session.dialog !== targetDialog ||
      !session.dialog?.open ||
      session.editor !== targetEditor
    )
      return null
  } while (session.frame || session.rendering)
  if (session.renderFailed) return null
  if (!session.blob) {
    window.clearTimeout(session.blobTimer)
    session.blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  }
  return session.blob
}

function fileName(): string {
  if (session.editor?.title) {
    const title = Array.from(session.editor.title.normalize('NFKC'))
      .slice(0, 50)
      .join('')
      .replace(/[<>:"/\\|?*\p{Cc}]/gu, '-')
      .replace(/\s+/g, ' ')
      .replace(/^[. ]+|[. ]+$/g, '')
    if (title) return `${title}-${session.format}.png`
  }
  if (session.subject?.calendar)
    return `races-${session.subject.calendar.period}-list-${session.format}.png`
  const tail =
    session.subject?.url.split('#')[0].split('/').filter(Boolean).at(-1) ?? session.page!.handle
  const name = tail.includes('.') && session.subject?.kind === 'page' ? session.page!.handle : tail
  return `${session.page!.handle}-${name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${session.tab}-${session.format}.png`
}

export async function download() {
  const data = await currentBlob()
  if (!data) return toast("impossible de générer l'image")
  const url = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName()
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

export async function copyImage() {
  if (typeof ClipboardItem === 'undefined' || !navigator.clipboard?.write) {
    return toast('copie d’image non prise en charge ici')
  }
  // Safari keeps the user gesture only when the item is built synchronously around a promise.
  const data = currentBlob().then(value => {
    if (!value) throw new Error('empty image')
    return value
  })
  try {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': data })])
    toast('image copiée')
  } catch {
    toast("impossible de copier l'image")
  }
}

export function canShareFiles(): boolean {
  try {
    const probe = new File([new Uint8Array(1)], 'probe.png', { type: 'image/png' })
    return !!navigator.canShare?.({ files: [probe] })
  } catch {
    return false
  }
}

export async function shareImage() {
  const data = await currentBlob()
  if (!data) return
  const file = new File([data], fileName(), { type: 'image/png' })
  try {
    await navigator.share({
      files: [file],
      title: session.editor?.title ?? titleText(),
      url: shareUrl(),
    })
  } catch (error) {
    if ((error as DOMException)?.name !== 'AbortError') toast('partage impossible')
  }
}
