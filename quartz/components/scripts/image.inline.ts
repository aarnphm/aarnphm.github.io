import { getFullSlug } from '../../util/path'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const configuredDialogs = new WeakMap<HTMLDialogElement, AbortSignal>()

document.addEventListener('nav', () => {
  if (getFullSlug(window) === 'lyd') return

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
  }

  function shouldSkipImage(img: HTMLImageElement) {
    return (
      img.dataset.noPopover === '' ||
      img.dataset.noPopover === 'true' ||
      img.dataset.ignorePopup === '' ||
      img.dataset.ignorePopup === 'true' ||
      img.closest('[data-pet-widget]') !== null ||
      img.closest('[data-remark-tikz]') !== null ||
      img.closest('#image-popup-modal') !== null
    )
  }

  for (const img of document.querySelectorAll('img')) {
    if (shouldSkipImage(img)) continue
    img.style.cursor = 'pointer'
  }

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
  closeButton.addEventListener('click', closeModal, { signal: dialogSignal })
  registerEscapeHandler(imageDialog, closeModal, () => imageDialog.open)
  dialogSignal.addEventListener(
    'abort',
    () => {
      if (configuredDialogs.get(imageDialog) === dialogSignal) {
        configuredDialogs.delete(imageDialog)
      }
    },
    { once: true },
  )
})
