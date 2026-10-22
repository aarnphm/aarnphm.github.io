import {
  cleanupMaps,
  cleanupPdfs,
  hydrateInternalHosts,
  hydrateMapboxMaps,
  hydrateSubstackEmbeds,
  mountPdfEmbeds,
} from './embeds'
import { session } from './session-state'

function lockPageScroll() {
  if (session.scrollLockState) return
  session.scrollLockState = { x: window.scrollX, y: window.scrollY }
  document.documentElement.classList.add('arena-modal-open')
  document.body.classList.add('arena-modal-open')
  document.body.style.position = 'fixed'
  document.body.style.top = `-${session.scrollLockState.y}px`
  document.body.style.left = '0'
  document.body.style.right = '0'
  document.body.style.width = '100%'
  document.body.style.overflow = 'hidden'
}

function unlockPageScroll() {
  const state = session.scrollLockState
  document.documentElement.classList.remove('arena-modal-open')
  document.body.classList.remove('arena-modal-open')
  document.body.style.position = ''
  document.body.style.top = ''
  document.body.style.left = ''
  document.body.style.right = ''
  document.body.style.width = ''
  document.body.style.overflow = ''
  session.scrollLockState = null
  if (state) {
    window.scrollTo(state.x, state.y)
  }
}

export async function showModal(blockId: string) {
  const modal = document.getElementById('arena-modal')
  const modalBody = modal?.querySelector('.arena-modal-body') as HTMLElement | null
  if (!modal || !modalBody) return

  const dataEl = document.getElementById(`arena-modal-data-${blockId}`)
  if (!dataEl && !session.channelViews) {
    console.warn(`Modal data not found for block ${blockId}`)
    return
  }

  session.currentBlockIndex = session.channelViews
    ? session.channelViews.blockIds.indexOf(blockId)
    : 0
  if (session.currentBlockIndex < 0) return
  session.modalRequest?.abort()
  const request = new AbortController()
  session.modalRequest = request

  cleanupMaps(modalBody)
  cleanupPdfs(modalBody)
  session.externalEmbeds?.cleanup()
  const status = document.createElement('p')
  status.className = 'arena-embed-loading'
  status.setAttribute('role', 'status')
  status.textContent = 'loading block…'
  modalBody.replaceChildren(status)
  modalBody.setAttribute('aria-busy', 'true')
  updateNavButtons()
  modal.classList.add('active')
  lockPageScroll()

  let clonedContent: HTMLElement
  try {
    const content = session.channelViews
      ? await session.channelViews.loadModal(blockId, request.signal)
      : dataEl?.cloneNode(true)
    if (request.signal.aborted || !modalBody.isConnected) return
    if (!(content instanceof HTMLElement)) throw new Error('Arena block content is unavailable')
    clonedContent = content
    clonedContent.style.display = 'block'
    modalBody.replaceChildren(clonedContent)
  } catch (error) {
    if (request.signal.aborted) return
    status.textContent = 'could not load this block'
    const retry = document.createElement('button')
    retry.type = 'button'
    retry.textContent = 'retry'
    retry.addEventListener('click', () => void showModal(blockId), { signal: request.signal })
    modalBody.append(retry)
    console.error(error)
    return
  } finally {
    if (session.modalRequest === request) modalBody.removeAttribute('aria-busy')
  }

  const sidebar = modalBody.querySelector('.arena-modal-sidebar') as HTMLElement | null
  const hasConnections = modalBody.querySelector('.arena-modal-connections') !== null
  const collapseBtn = modal?.querySelector('.arena-modal-collapse') as HTMLElement | null

  if (sidebar) {
    let shouldCollapse = !hasConnections || clonedContent.dataset.sidebarCollapsed === 'true'

    // Check for sidebar metadata
    if (session.arenaSearchData) {
      const blockData = session.arenaSearchData.blocks.find(b => b.id === blockId)
      if (typeof blockData?.metadata?.sidebar === 'string') {
        const sidebarValue = blockData.metadata.sidebar.toLowerCase().trim()
        if (sidebarValue === 'false' || sidebarValue === '0') {
          shouldCollapse = true
        }
      }
    }

    if (shouldCollapse) {
      sidebar.classList.add('collapsed')
      collapseBtn?.classList.add('active')
    } else {
      sidebar.classList.remove('collapsed')
      collapseBtn?.classList.remove('active')
    }
    collapseBtn?.setAttribute('aria-expanded', String(!shouldCollapse))
  }

  hydrateSubstackEmbeds(modalBody)
  session.externalEmbeds?.mount(modalBody)
  hydrateInternalHosts(modalBody)
  hydrateMapboxMaps(modalBody)
  mountPdfEmbeds(modalBody)
}

export function closeModal() {
  session.modalRequest?.abort()
  session.modalRequest = null
  const modal = document.getElementById('arena-modal')
  if (modal) {
    const modalBody = modal.querySelector('.arena-modal-body') as HTMLElement | null
    if (modalBody) {
      cleanupMaps(modalBody)
      cleanupPdfs(modalBody)
      session.externalEmbeds?.cleanup()
      modalBody.replaceChildren()
    }
    modal.classList.remove('active')
    unlockPageScroll()
    session.channelViews?.refresh()
  }
}

export async function navigateBlock(direction: number) {
  if (!document.getElementById('arena-modal')?.classList.contains('active')) return
  const newIndex = session.currentBlockIndex + direction
  if (newIndex < 0 || newIndex >= session.totalBlocks) return

  const blockId = session.channelViews?.blockIds[newIndex]
  if (blockId) {
    await showModal(blockId)
  }
}

function updateNavButtons() {
  const prevBtn = document.querySelector('.arena-modal-prev') as HTMLButtonElement
  const nextBtn = document.querySelector('.arena-modal-next') as HTMLButtonElement

  if (prevBtn) {
    prevBtn.disabled = session.currentBlockIndex === 0
    prevBtn.style.opacity = session.currentBlockIndex === 0 ? '0.3' : '1'
  }

  if (nextBtn) {
    nextBtn.disabled = session.currentBlockIndex >= session.totalBlocks - 1
    nextBtn.style.opacity = session.currentBlockIndex >= session.totalBlocks - 1 ? '0.3' : '1'
  }
}
