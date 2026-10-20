import { getFullSlug } from '../../util/path'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const configuredDialogs = new WeakMap<HTMLDialogElement, AbortSignal>()

function shouldSkipImage(img: HTMLImageElement) {
  return (
    img.dataset.noPopover === '' ||
    img.dataset.noPopover === 'true' ||
    img.dataset.ignorePopup === '' ||
    img.dataset.ignorePopup === 'true' ||
    img.closest('[data-remark-tikz]') !== null ||
    img.closest('#image-popup-modal') !== null
  )
}

function markPopupImages(root: ParentNode = document) {
  for (const img of root.querySelectorAll('img')) {
    if (shouldSkipImage(img)) continue
    img.style.cursor = 'var(--cursor-pointer, pointer)'
  }
}

document.addEventListener('nav', () => {
  const slug = getFullSlug(window)
  if (slug === 'lyd' || slug === 'vibes') return

  const dialog = document.getElementById('image-popup-modal')
  const modalImg = dialog?.querySelector<HTMLImageElement>('.image-popup-img')
  const closeBtn = dialog?.querySelector<HTMLButtonElement>('.image-popup-close')

  if (!(dialog instanceof HTMLDialogElement) || !modalImg || !closeBtn) return
  const imageDialog = dialog
  const imageElement = modalImg
  const closeButton = closeBtn

  function closeModal() {
    if (imageDialog.open) imageDialog.close()
  }

  function openModal(imgSrc: string, imgAlt: string) {
    imageElement.src = imgSrc
    imageElement.alt = imgAlt
    if (!imageDialog.open) imageDialog.showModal()
    closeButton.focus({ preventScroll: true })
  }

  closeModal()
  markPopupImages(document)
  registerEscapeHandler(imageDialog, closeModal, () => imageDialog.open)

  const dialogSignal = rootNavSignal(imageDialog)
  if (configuredDialogs.get(imageDialog) === dialogSignal) return
  configuredDialogs.set(imageDialog, dialogSignal)
  document.addEventListener(
    'click',
    event => {
      const element =
        event.target instanceof Element
          ? event.target
          : event.target instanceof Node
            ? event.target.parentElement
            : null
      const img = element?.closest('img')
      if (!(img instanceof HTMLImageElement) || shouldSkipImage(img)) return
      openModal(img.src, img.alt)
    },
    { signal: dialogSignal },
  )
  document.addEventListener(
    'contentdecrypted',
    (event: CustomEventMap['contentdecrypted']) => {
      markPopupImages(event.detail.content)
    },
    { signal: dialogSignal },
  )
  closeButton.addEventListener('click', closeModal, { signal: dialogSignal })
  dialogSignal.addEventListener(
    'abort',
    () => {
      closeModal()
      if (configuredDialogs.get(imageDialog) === dialogSignal) {
        configuredDialogs.delete(imageDialog)
      }
    },
    { once: true },
  )
})
